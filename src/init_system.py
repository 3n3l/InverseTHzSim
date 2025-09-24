import mitsuba as mi
import drjit as dr
import numpy as np
import torch

from config import config

def mega_kernel(state):
    dr.set_flag(dr.JitFlag.LoopRecord, state)
    dr.set_flag(dr.JitFlag.VCallRecord, state)
    dr.set_flag(dr.JitFlag.VCallOptimize, state)

def comp_grad_r(flag: bool):
    config["render_grad"] = flag

    # Turn off Mega Kernel (check if helpful to leave on for false flag)
    mega_kernel(False)

    print(mi.variants())

    # Choose variant
    if flag:
        mi.set_variant('cuda_ad_mono')
    else:
        mi.set_variant('cuda_mono')

# Set Pytorch Device
config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch device: {config['device']}")

# Setup Pytorch Seed
torch.manual_seed(0)

# Setup numpy Seed
np.random.seed(0)