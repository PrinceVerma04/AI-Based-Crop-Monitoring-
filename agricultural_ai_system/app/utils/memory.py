import psutil
import gc
import logging

import torch

def memory_report():
    mem = psutil.virtual_memory()
    logging.info(
        f"Memory: {mem.used/1024/1024:.1f}MB used, "
        f"{mem.available/1024/1024:.1f}MB available"
    )
    
def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    memory_report()