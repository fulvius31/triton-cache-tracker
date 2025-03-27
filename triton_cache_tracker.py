#!/usr/bin/env python3
"""
Triton Cache Tracker - A utility for monitoring and analyzing Triton kernel compilation cache.

This module provides tools to track cache hits/misses and analyze kernel compilation details
by hooking into Triton's JIT compilation process.
"""

import ast
import hashlib
import json
import os
import re
from functools import wraps
from pathlib import Path

from triton._C.libtriton import get_cache_invalidating_env_vars
from triton._utils import find_paths_if, get_iterable_path
from triton.compiler.compiler import make_backend, triton_key
from triton.runtime.cache import _base32
from triton.runtime.driver import driver
from triton.runtime.jit import JITFunction


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles tuples specially for serialization."""
    
    def default(self, obj):
        """Convert special objects to JSON serializable types."""
        if isinstance(obj, tuple):
            return {'__tuple__': True, 'items': [self.default(e) for e in obj]}
        return super().default(obj)

    def encode(self, obj):
        """Pre-process objects before encoding to handle tuples consistently."""
        def hint_tuples(item):
            if isinstance(item, list):
                return [hint_tuples(e) for e in item]
            if isinstance(item, dict):
                return {k: hint_tuples(v) for k, v in item.items()}
            if isinstance(item, tuple):
                return {'__tuple__': True, 'items': [hint_tuples(e) for e in item]}
            return item
            
        return super().encode(hint_tuples(obj))


def parse_options_dict_from_compile_key(compile_key: str) -> dict:
    """
    Extract the Python dictionary for the options from a compile key string.
    
    Args:
        compile_key: A string like "[('*fp16','D'), ('*fp16','D')]{'num_warps':8, 'num_stages':3, 'debug':False}"
    
    Returns:
        dict: The extracted options dictionary, or an empty dict if no options found.
    """
    match = re.search(r'(\{.*\})', compile_key)
    if not match:
        return {}
    dict_str = match.group(1)
    return ast.literal_eval(dict_str)


def format_metadata_as_cuda_options(metadata):
    """
    Convert KernelMetadata to a CUDAOptions-style string representation.
    
    Args:
        metadata: The kernel metadata object
        
    Returns:
        str: A formatted string representing the CUDA options
    """
    opts = [
        f"num_warps={metadata.num_warps}",
        f"num_ctas={metadata.num_ctas}",
        f"num_stages={metadata.num_stages}",
        f"maxnreg={metadata.maxnreg}",
        f"cluster_dims={metadata.cluster_dims}",
        f"ptx_version={metadata.ptx_version}",
        f"enable_fp_fusion={metadata.enable_fp_fusion}",
        f"launch_cooperative_grid={metadata.launch_cooperative_grid}",
        f"supported_fp8_dtypes={tuple(metadata.supported_fp8_dtypes)}",
        f"deprecated_fp8_dtypes={tuple(metadata.deprecated_fp8_dtypes)}",
        f"default_dot_input_precision={metadata.default_dot_input_precision!r}",
        f"allowed_dot_input_precisions={tuple(metadata.allowed_dot_input_precisions)}",
        f"max_num_imprecise_acc_default={metadata.max_num_imprecise_acc_default}",
        f"extern_libs={tuple(tuple(lib) for lib in metadata.extern_libs)}",
        f"debug={metadata.debug}",
        f"backend_name={metadata.backend_name!r}",
        f"sanitize_overflow={metadata.sanitize_overflow}",
        f"arch={metadata.arch!r}",
    ]
    return f"CUDAOptions({', '.join(opts)})"


def file_hash(path):
    """
    Compute SHA256 hash of a file, replicating Triton's file hashing.
    
    Args:
        path: Path to the file to hash
        
    Returns:
        str: Hexadecimal hash value
    """
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_cudaoptions_hash(metadata):
    """
    Replicate Triton's CUDAOptions hash computation.
    
    Args:
        metadata: The kernel metadata object
        
    Returns:
        str: Hexadecimal hash value
    """
    extern_libs = dict(metadata.extern_libs or {})
    
    if 'libdevice' not in extern_libs:
        default_libdir = Path(__file__).parent.parent / 'backends' / 'nvidia' / 'lib'
        libdevice_path = os.getenv("TRITON_LIBDEVICE_PATH", 
                                  str(default_libdir / 'libdevice.10.bc'))
        extern_libs['libdevice'] = libdevice_path

    hash_dict = {
        "allowed_dot_input_precisions": tuple(metadata.allowed_dot_input_precisions),
        "arch": metadata.arch,
        "backend_name": metadata.backend_name,
        "cluster_dims": metadata.cluster_dims,
        "debug": metadata.debug,
        "default_dot_input_precision": metadata.default_dot_input_precision,
        "deprecated_fp8_dtypes": tuple(metadata.deprecated_fp8_dtypes),
        "enable_fp_fusion": metadata.enable_fp_fusion,
        "extern_libs": tuple((k, file_hash(v)) for k, v in sorted(extern_libs.items())),
        "launch_cooperative_grid": metadata.launch_cooperative_grid,
        "max_num_imprecise_acc_default": metadata.max_num_imprecise_acc_default,
        "maxnreg": metadata.maxnreg,
        "num_ctas": metadata.num_ctas,
        "num_stages": metadata.num_stages,
        "num_warps": metadata.num_warps,
        "ptx_version": metadata.ptx_version,
        "sanitize_overflow": metadata.sanitize_overflow,
        "supported_fp8_dtypes": tuple(metadata.supported_fp8_dtypes),
    }
    
    # Generate hash key exactly as CUDAOptions.hash() does
    sorted_items = sorted(hash_dict.items(), key=lambda x: x[0])
    key = "_".join([f"{name}-{val}" for name, val in sorted_items])
    
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def serialize_specialization_data(name, signature, constants, attrs, options, key):
    """
    Serialize kernel specialization data to JSON.
    
    Args:
        name: Kernel function name
        signature: Function signature information
        constants: Constant expressions
        attrs: Attribute values
        options: Compilation options
        key: Compilation key
        
    Returns:
        str: JSON-encoded specialization data
    """
    # Convert KernelMetadata to CUDAOptions-like structure
    filtered_options = {
        'num_warps': options.num_warps,
        'num_ctas': options.num_ctas,
        'num_stages': options.num_stages,
        'maxnreg': options.maxnreg,
        'cluster_dims': options.cluster_dims,
        'ptx_version': options.ptx_version,
        'enable_fp_fusion': options.enable_fp_fusion,
        'launch_cooperative_grid': options.launch_cooperative_grid,
        'supported_fp8_dtypes': tuple(options.supported_fp8_dtypes),
        'deprecated_fp8_dtypes': tuple(options.deprecated_fp8_dtypes),
        'default_dot_input_precision': options.default_dot_input_precision,
        'allowed_dot_input_precisions': tuple(options.allowed_dot_input_precisions),
        'max_num_imprecise_acc_default': options.max_num_imprecise_acc_default,
        'extern_libs': tuple(tuple(lib) for lib in options.extern_libs),
        'debug': options.debug,
        'backend_name': options.backend_name,
        'sanitize_overflow': options.sanitize_overflow,
        'arch': options.arch
    }

    constants_processed = {
        k: str(v) if hasattr(v, '__class__') and v.__class__.__name__ == "dtype" 
        else v 
        for k, v in constants.items()
    }

    return json.dumps({
        'name': name,
        'signature': signature,
        'constant_keys': [list(x) for x in constants.keys()],
        'constant_vals': list(constants_processed.values()),
        'attrs_keys': [list(x) for x in attrs.keys()],
        'attrs_vals': list(attrs.values()),
        'options': filtered_options,
        'key': key
    }, cls=CustomJSONEncoder)


class JitFunctionInfo:
    """Helper class to wrap JITFunction information for hooks."""
    
    def __init__(self, module, name, jit_function):
        self.module = module
        self.name = name
        self.jit_function = jit_function


class TritonCacheTracker:
    """
    A utility for tracking and analyzing Triton kernel compilation cache.
    
    This class instruments Triton's JIT compilation process to track cache hits and misses,
    providing insights into kernel compilation behavior and performance.
    """
    
    def __init__(self):
        """Initialize a new tracker with zero counters."""
        self.total = 0
        self.hits = 0
        self.misses = 0
        self.per_kernel = {}
        self._original_methods = {}
        # Get the Triton cache directory
        self.cache_dir = os.environ.get("TRITON_CACHE_DIR", 
                                       os.path.expanduser("~/.triton/cache/"))
        # Create the cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        # Get the initial set of cached kernels
        self._initial_cache_keys = self._get_cached_keys()
    
    def install(self):
        """
        Install hooks to track Triton kernel compilation.
        
        This method:
        1. Sets JITFunction.compiled_hook to track compilation events
        2. Monkey-patches JITFunction.run to pass compiled kernels to hooks
        3. Wraps the run method to track hits and misses
        """
        self._original_methods = {
            'run': JITFunction.run,
            'call_hook': JITFunction._call_hook
        }
        
        JITFunction.compiled_hook = self._compilation_hook
        self._patch_jitfunction_call_hook()
        self._patch_jitfunction_run()
    
    def uninstall(self):
        """
        Remove hooks and restore original functionality.
        
        This restores the original methods that were monkey-patched during install.
        """
        if self._original_methods:
            JITFunction.run = self._original_methods.get('run', JITFunction.run)
            JITFunction._call_hook = self._original_methods.get('call_hook', JITFunction._call_hook)
            JITFunction.compiled_hook = None
    
    def _get_cached_keys(self):
        """
        Get the set of kernel keys that already exist in the Triton cache directory.
        
        Returns:
            set: Set of base32-encoded kernel keys
        """
        keys = set()
        if os.path.exists(self.cache_dir):
            for item in os.listdir(self.cache_dir):
                # Each subdirectory in the cache is a kernel
                item_path = os.path.join(self.cache_dir, item)
                if os.path.isdir(item_path):
                    keys.add(item)
        return keys

    def _compilation_hook(self, key, repr, fn, compile, **kwargs):
        """
        Hook called after Triton compiles a kernel.
        
        Args:
            key: Compilation key
            repr: String representation
            fn: JitFunctionInfo object
            compile: Dictionary with compilation details
            **kwargs: Additional arguments
            
        Returns:
            bool: Always returns False to continue normal processing
        """
        kernel = compile.get("kernel", None)
        if kernel is None:
            return False
            
        tk = triton_key()
        ast_hash = fn.jit_function.cache_key
        
        backend_obj = make_backend(kernel.metadata.target)
        backend_hash_str = str(backend_obj.hash())
        options_str = format_metadata_as_cuda_options(kernel.metadata)
        options_hash = get_cudaoptions_hash(kernel.metadata)
        
        env_vars = get_cache_invalidating_env_vars()
        env_vars_str = str(sorted(env_vars.items()))
        
        # Create the full cache key
        big_key_str = f"{tk}-{kernel.src.hash()}-{backend_hash_str}-{options_hash}-{env_vars_str}"
        final_sha = hashlib.sha256(big_key_str.encode("utf-8")).hexdigest()
        base32_key = _base32(final_sha)
        
        print(f"[TritonCacheTracker] Compiled kernel: {fn.name} using cache key {base32_key}")
        # Check if this kernel was in the cache when we started
        name = fn.name
        self.per_kernel.setdefault(name, {"miss": 0, "hit": 0})
        
        # Check if the kernel was in the cache before we started
        if base32_key in self._initial_cache_keys:
            # It was a hit - kernel was already in the cache
            self.hits += 1
            self.per_kernel[name]["hit"] += 1
            print(f"[TritonCacheTracker] HIT: {name} kernel using cached {base32_key}")
        else:
            # It was a miss - kernel had to be compiled
            self.misses += 1
            self.per_kernel[name]["miss"] += 1
            print(f"[TritonCacheTracker] MISS: {name} kernel compiled to {base32_key}")
            
        return False
    
    def _patch_jitfunction_call_hook(self):
        """
        Patch JITFunction._call_hook to pass kernel to the compiled_hook.
        
        This method creates a wrapper around the original _call_hook method
        that passes the compiled kernel to the hook for analysis.
        """
        original_call_hook = JITFunction._call_hook
        
        def inject_kernel_hook(self_jit, key, signature, device, constants, options, 
                               configs, is_warmup, before, kernel=None):
            """Enhanced _call_hook that can pass kernel to the user hook."""
            hook = JITFunction.cache_hook if before else JITFunction.compiled_hook
            if hook is None:
                return False

            name = self_jit.fn.__name__
            module = self_jit.fn.__module__
            arg_reprs = ", ".join([f"{p.name}: {ty}" for p, ty in zip(self_jit.params, key[1])])
            repr_str = (
                f"{name}[num_warps={options.num_warps}, num_ctas={options.num_ctas}, "
                f"num_stages={options.num_stages}, enable_fp_fusion={options.enable_fp_fusion}, "
                f"launch_cooperative_grid={options.launch_cooperative_grid}]({arg_reprs})"
            )

            specialization_data = serialize_specialization_data(
                name, signature, constants, configs[0], options, key
            )

            compile_dict = {
                "key": key,
                "signature": signature,
                "device": device,
                "constants": constants,
                "num_warps": options.num_warps,
                "num_ctas": options.num_ctas,
                "num_stages": options.num_stages,
                "enable_fp_fusion": options.enable_fp_fusion,
                "launch_cooperative_grid": options.launch_cooperative_grid,
                "extern_libs": options.extern_libs,
                "configs": configs,
                "specialization_data": specialization_data,
                "is_warmup": is_warmup,
                "kernel": kernel,  # Pass the compiled kernel
            }

            return hook(
                key=key,
                repr=repr_str,
                fn=JitFunctionInfo(module, name, self_jit),
                compile=compile_dict,
                is_manual_warmup=is_warmup,
                already_compiled=bool(kernel),
            )

        def call_hook_wrapper(self_jit, *args, kernel=None, **kwargs):
            """Wrapper that decides whether to use the original or enhanced hook."""
            if kernel is not None:
                return inject_kernel_hook(self_jit, *args, kernel=kernel, **kwargs)
            return original_call_hook(self_jit, *args, **kwargs)

        JITFunction._call_hook = call_hook_wrapper
    
    def _patch_jitfunction_run(self):
        """
        Patch JITFunction.run to track total kernel invocations and pass kernels to hooks.
        
        This replaces the run method with a version that:
        1. Counts total kernel invocations
        2. Passes the kernel to compiled_hook after compilation
        """
        original_run = JITFunction.run
        
        @wraps(original_run)
        def patched_run(self_jit, *args, grid, warmup, **kwargs):
            """Enhanced run method that tracks total invocations."""
            self.total += 1
            
            for hook_func in self_jit.pre_run_hooks:
                hook_func(*args, **kwargs)

            device = driver.active.get_current_device()
            stream = driver.active.get_current_stream(device)
            kernel_cache, target, backend, binder = self_jit.device_caches[device]
            bound_args, specialization, options_ = binder(*args, **kwargs)
            key = str(specialization) + str(options_)

            # Check if kernel is in cache
            kernel = kernel_cache.get(key, None)
            if kernel is None:
                options_ = backend.parse_options(kwargs)
                sigkeys = [x.name for x in self_jit.params]
                sigvals = [x[0] for x in specialization]
                signature = dict(zip(sigkeys, sigvals))

                constexprs = find_paths_if(sigvals, lambda _, val: val == "constexpr")
                constexprs = {path: get_iterable_path(list(bound_args.values()), path) for path in constexprs}
                
                attrvals = [x[1] for x in specialization]
                attrs = find_paths_if(attrvals, lambda _, x: isinstance(x, str))
                attrs = {k: backend.parse_attr(get_iterable_path(attrvals, k)) for k in attrs}

                if self_jit._call_hook(key, signature, device, constexprs, options_, [attrs], warmup, before=True):
                    return None

                src = self_jit.ASTSource(self_jit, signature, constexprs, attrs)
                kernel = self_jit.compile(src, target=target, options=options_.__dict__)
                kernel_cache[key] = kernel

                self_jit._call_hook(
                    key, signature, device, constexprs, options_, [attrs], warmup, 
                    before=False, kernel=kernel
                )

            if not warmup:
                if callable(grid):
                    grid = grid(bound_args)
                grid_size = len(grid)
                grid_0 = grid[0]
                grid_1 = grid[1] if grid_size > 1 else 1
                grid_2 = grid[2] if grid_size > 2 else 1
                launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
                kernel.run(
                    grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata,
                    launch_metadata, self_jit.CompiledKernel.launch_enter_hook,
                    self_jit.CompiledKernel.launch_exit_hook,
                    *bound_args.values()
                )
                
            return kernel

        JITFunction.run = patched_run
    
    def stats(self):
        """
        Get statistics about cache performance.
        
        Returns:
            dict: Dictionary with total invocations, hits, misses, and per-kernel statistics
        """
        # Get all current kernels in the cache
        current_cache_keys = self._get_cached_keys()
        
        # Newly added kernels during this run
        new_kernels = current_cache_keys - self._initial_cache_keys
        
        return {
            "total_invocations": self.total,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "per_kernel": self.per_kernel,
            "new_kernels_count": len(new_kernels),
            "new_kernels": list(new_kernels)
        }
    
    def summary(self):
        """Print a summary of cache statistics to stdout."""
        stats = self.stats()
        print(f"[TritonCacheTracker] Total invocations: {stats['total_invocations']}")
        print(f"[TritonCacheTracker] Cache directory: {self.cache_dir}")
        print(f"[TritonCacheTracker] Cache performance: {stats['cache_hits']} hits, "
              f"{stats['cache_misses']} misses")
        
        if stats['new_kernels_count'] > 0:
            print(f"[TritonCacheTracker] New kernels added to cache: {stats['new_kernels_count']}")
            for key in stats['new_kernels']:
                print(f" - {key}")
        
        for name, data in stats["per_kernel"].items():
            print(f" - {name}: {data['hit']} hits, {data['miss']} misses")
            
    def reset(self):
        """Reset all statistics counters to zero and refresh initial cache state."""
        self.total = 0
        self.hits = 0
        self.misses = 0
        self.per_kernel = {}
        # Reset the initial cache keys to current state
        self._initial_cache_keys = self._get_cached_keys()

