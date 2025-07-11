import time
import torch
import os
import shutil
from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
#from torch._inductor.utils import clear_caches
from torch._inductor.codecache import PyCodeCache


def timed(fn, *args, msg=""):
    torch.cuda.synchronize()
    start = time.time()
    result = fn(*args)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"{msg} took {elapsed:.4f} seconds")
    return result, elapsed

# Simple matmul function
def fn(x, y):
    return x @ y + 1

x = torch.randn(1024, 1024, device="cuda")
y = torch.randn(1024, 1024, device="cuda")

print("=== COLD START ===")
compiled_fn = fn #torch.compile(fn, mode="max-autotune", fullgraph=True)

out, cold_time = timed(compiled_fn, x, y, msg="First compiled run")

# # Save cache artifacts
# artifacts = torch.compiler.save_cache_artifacts()
# assert artifacts is not None, "No artifacts saved"
# artifact_bytes, cache_info = artifacts

# with open("artifacts.bin", "wb") as f:
#     f.write(artifact_bytes)

#print("Cache saved:", cache_info)

print("\n=== CLEARING CACHES ===")
# Clear in-memory caches
torch.compiler.reset()


# Clear on-disk cache
shutil.rmtree(os.path.expanduser("~/.cache/torch/inductor"), ignore_errors=True)
print("Caches cleared")

print("\n=== WARM START ===")
# Load artifacts back
# with open("artifacts.bin", "rb") as f:
#     loaded_bytes = f.read()

# torch.compiler.load_cache_artifacts(loaded_bytes)

compiled_fn_warm = fn #torch.compile(fn, mode="max-autotune", fullgraph=True)

out2, warm_time = timed(compiled_fn_warm, x, y, msg="Second compiled run with cache")

print("\n=== RESULTS ===")
print(f"Cold start time: {cold_time:.4f} seconds")
print(f"Warm start time: {warm_time:.4f} seconds")
print(f"Speedup: {cold_time/warm_time:.2f}x faster")