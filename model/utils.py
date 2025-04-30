import os
import sys
import psutil
import torch
from os.path import join


def debug_only(func):
    def wrapper(*args, **kwargs):
        if os.getenv("DEBUG"):
            return func(*args, **kwargs)
    return wrapper

@debug_only
def check_ram():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory used: {mem_info.rss / (1024 ** 3):.2f} GB")
check_ram()

def check_gpu():
    if torch.cuda.is_available():
        alloc  = torch.cuda.memory_allocated("cuda") / 1024**3
        reserved = torch.cuda.memory_reserved("cuda")   / 1024**3
        print(
            f"GPU Mem — Allocated: {alloc:.2f} GB, "
            f"Reserved: {reserved:.2f} GB"
        )