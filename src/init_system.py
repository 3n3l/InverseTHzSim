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

    print(f"\nAvailable variants: {mi.variants()}")

    # Choose variant
    variant_str = "cuda" if torch.cuda.is_available() else "llvm"
    variant_str += "_ad_mono" if flag else "_mono"
    mi.set_variant(variant_str)
    print(f"\nChosen variant: {variant_str}\n")


# Set Pytorch Device
device_str = "cuda" if torch.cuda.is_available() else "cpu"
config["device"] = device_str
print(f"\nTorch device: {device_str}")

# Setup Pytorch Seed
torch.manual_seed(0)

# Setup numpy Seed
np.random.seed(0)
