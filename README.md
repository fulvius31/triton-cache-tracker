# Triton Cache Tracker (TCT)
NOTE: IT IS A WIP PROJECT TESTED WITH TRITON 3.3.0

A lightweight utility for monitoring and analyzing Triton kernel compilation cache behavior.

## Overview

Triton Cache Tracker monitors Triton's kernel compilation process, tracking which kernels are loaded from cache versus compiled from scratch. By checking the Triton cache directory, it tells you if the Triton cache was hit or not.

## Installation

Simply copy the `triton_cache_tracker.py` file to your project directory:

```bash
# Clone the repo
git clone git@github.com:fulvius31/triton-cache-tracker.git

# Or just copy the file to your project
cp triton_cache_tracker.py /path/to/your/project/
```

## Quick Start

```python
from triton_cache_tracker import TritonCacheTracker
import triton

# Initialize and install the tracker
tracker = TritonCacheTracker()
tracker.install()

# Run your Triton kernels...
# (They will be automatically tracked)

# Print statistics
tracker.summary()

# Clean up when done (optional)
tracker.uninstall()
```

## How It Works: The Monkey-Patching Explained

Triton Cache Tracker uses monkey-patching to intercept Triton's compilation process without modifying Triton's source code. Here's how it works:

### 1. JITFunction Pipeline

To understand the monkey-patching, you need to know Triton's compilation pipeline:

- `JITFunction.run`: The entry point called when you execute a Triton kernel
- `JITFunction._call_hook`: Called before and after compilation
- `JITFunction.compiled_hook`: User-definable hook that's called after compilation

### 2. Cache Directory Inspection

When initialized, the tracker:
- Checks the `TRITON_CACHE_DIR` environment variable (defaults to `~/.triton/cache/`)
- Scans this directory to inventory all pre-existing cached kernels
- Uses this inventory as a baseline to determine cache hits vs. misses

### 3. The Three Key Patches

The tracker installs hooks at three critical points:

#### a) Compilation Hook

```python
JITFunction.compiled_hook = self._compilation_hook
```

This hook is called whenever a kernel is compiled. In our implementation:
- It calculates the exact same cache key that Triton uses (base32-encoded)
- It checks if that key existed in the cache directory at startup
- If it existed → It's a HIT (loaded from cache)
- If it didn't → It's a MISS (needed compilation)

#### b) Enhanced Call Hook

```python
original_call_hook = JITFunction._call_hook
JITFunction._call_hook = enhanced_version
```

The default `_call_hook` doesn't pass the compiled kernel to the `compiled_hook`. TCT enhanced version:
- Accepts an additional `kernel` parameter
- Passes this parameter to the `compiled_hook` after compilation
- This gives our hook access to the kernel's metadata (crucial for calculating cache keys)

#### c) Run Method Wrapper

```python
original_run = JITFunction.run
JITFunction.run = patched_run
```

The Triton Cache Tracker patched run method:
- Counts total kernel invocations
- Calls the original compilation process
- Ensures our `_call_hook` receives the compiled kernel
- Maintains all original functionality

### 4. Cache Key Calculation

The most intricate part is replicating Triton's exact cache key calculation:

1. TCT extracts the same components Triton uses:
   - Triton version hash (`triton_key()`)
   - AST hash (source code hash)
   - Backend hash (e.g., CUDA backend)
   - Options hash (compilation options)
   - Environment variables hash

2. TCT combines these exactly as Triton does and apply SHA256

3. TCT encode using base32, matching Triton's cache directory naming

This ensures our cache key matches the filesystem paths Triton uses.

## Usage Examples

### Basic Tracking

```python
from triton_cache_tracker import TritonCacheTracker

# Initialize and install
tracker = TritonCacheTracker()
tracker.install()

# Run your Triton code normally...

# Print a summary of statistics
tracker.summary()
```

### Resetting the Tracker

```python
# Reset all statistics counters and refresh the initial cache state
tracker.reset()
```

## Interpreting Results

When you run `tracker.summary()`, you'll see output like:

```
[TritonCacheTracker] Cache directory: /home/user/.triton/cache/
[TritonCacheTracker] Cache performance: 0 hits, 2 misses
[TritonCacheTracker] New kernels added to cache: 2
 - ABCDEFGHIJKLMNOPQRSTUVWXYZ234567
 - ZYXWVUTSRQPONMLKJIHGFEDCBA765432
 - matmul_kernel: 0 hits, 2 misses
```

This tells you:
- None of those executions used pre-cached kernels
- Two unique kernel configurations were compiled
- The exact base32-encoded cache keys for the new kernels
- All executions were for a kernel named "matmul_kernel"

This is typically what you'd see on the first run with a fresh cache. In subsequent runs, you should see more cache hits if your kernel configurations remain stable.

## License

MIT  [LICENSE](LICENSE)
