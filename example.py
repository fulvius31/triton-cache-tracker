import triton
from triton_cache_manager_tracker import TrackingCacheManager, _tracker
import torch
import triton.language as tl


# Just a simple add_kernel
@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Element-wise addition of two vectors.
    """
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)


def add_vectors(x: torch.Tensor, y: torch.Tensor):
    """
    Wrapper to launch the Triton add_kernel.
    """
    assert x.is_cuda and y.is_cuda and x.device == y.device
    output = torch.empty_like(x)
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


# Example use of the tracker
if __name__ == "__main__":
    cache_dir = triton.knobs.cache.dir
    triton.knobs.cache.manager_class = TrackingCacheManager

    size = 1024 * 128
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")

    print("\n--- Running Kernel ---")
    output1 = add_vectors(x, y)

    print("\n--- Cache Statistics ---")
    stats = _tracker.snapshot()

    totals = stats["totals"]
    per_key = stats["per_key"]

    print(f"Total cache accesses : {totals['access']}")
    print(f"   ├── hits          : {totals['hit']}")
    print(f"   └── misses        : {totals['miss']}\n")

    print("Per-kernel statistics")
    for key, data in per_key.items():
        name = data["name"]
        hits = data["hits"]
        misses = data["misses"]
        print(f"  • {name:<20}  (key={key[:10]}…):  hits={hits:3d}  misses={misses:3d}")
    # You can reset the stats if needed
    # _global_cache_tracker_stats.reset_stats()
