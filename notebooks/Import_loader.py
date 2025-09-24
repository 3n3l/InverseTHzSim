import sys
sys.path.append('../src/')

from config import config
import init_system
init_system.comp_grad_r(False)

import sampler as smp
smp.init_RNG()

import utils
import sparams
import scenes
import measurement
import reconstruction as recon
import losses
import optimisation as opt
import visualization as vis
import opt_utils
import geom_utils
import setup_utils

sparams.mimo_system.switch_off([2], 10)