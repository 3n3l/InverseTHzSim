import drjit as dr
import torch

import sparams
from config import config

if config['render_grad']:
    @dr.wrap_ad(source='drjit', target='torch')
    def torch_fft_l2(ref_real, ref_imag, msr_real, msr_imag):
        ref_cpx = torch.complex(ref_real, ref_imag).to(config['device'])
        msr_cpx = torch.complex(msr_real, msr_imag).to(config['device'])

        pad = torch.zeros(sparams.mimo_system.get_param('N_Tx'), sparams.mimo_system.get_param('N_Rx'), 192).to(config['device'])
        pad_cpx = torch.complex(pad, pad).to(config['device'])
        hamm_window = torch.hamming_window(64)
        hamm_window = hamm_window.repeat(sparams.mimo_system.get_param('N_Tx'), sparams.mimo_system.get_param('N_Rx'), 4).to(config['device'])

        ref_cpx = torch.cat((ref_cpx, pad_cpx), 2).to(config['device'])
        msr_cpx = torch.cat((msr_cpx, pad_cpx), 2).to(config['device'])

        ref_cpx = ref_cpx*hamm_window
        msr_cpx = msr_cpx*hamm_window

        ref_fft = torch.fft.fft(ref_cpx)
        msr_fft = torch.fft.fft(msr_cpx)

        return torch.sum((msr_fft - ref_fft)**2)
else:
    def torch_fft_l2(ref, msr):
        ref.normalize()
        msr.normalize()

        T_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        ref_cpx = ref.get_measurement().cpu().numpy()[T_indices, :, :]
        msr_cpx = msr.get_measurement().cpu().numpy()[T_indices, :, :]

        ref_cpx = torch.complex(torch.Tensor(ref_cpx.real), torch.Tensor(ref_cpx.imag)).to(config['device'])
        msr_cpx = torch.complex(torch.Tensor(msr_cpx.real), torch.Tensor(msr_cpx.imag)).to(config['device'])

        hamm_window = torch.hamming_window(64)
        hamm_window = hamm_window.repeat(len(T_indices), sparams.mimo_system.get_param('N_Rx'), 1).to(config['device'])

        msr_cpx = msr_cpx*hamm_window
        ref_cpx = ref_cpx*hamm_window

        msr_fft = torch.fft.fft(msr_cpx)
        ref_fft = torch.fft.fft(ref_cpx)

        abs_msr = torch.abs(msr_fft) 
        abs_ref = torch.abs(ref_fft)

        return torch.sum((abs_msr - abs_ref)**2)
    
def correlation_loss(ref_, msr_):
    ref_cpx = ref_.get_measurement()
    msr_cpx = msr_.get_measurement()

    return -1 * torch.abs(torch.sum(torch.conj(msr_cpx) * ref_cpx))
    
def torch_l1_loss(ref, msr):
    ref.normalize()
    msr.normalize()

    T_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    ref_cpx = ref.get_measurement().cpu().numpy()[T_indices, :, :]
    msr_cpx = msr.get_measurement().cpu().numpy()[T_indices, :, :]

    ref_cpx = torch.complex(torch.Tensor(ref_cpx.real), torch.Tensor(ref_cpx.imag)).to(config['device'])
    msr_cpx = torch.complex(torch.Tensor(msr_cpx.real), torch.Tensor(msr_cpx.imag)).to(config['device'])

    loss = torch.nn.L1Loss()

    return loss(ref_cpx, msr_cpx)

def torch_l2_loss(ref_, msr_):
    ref_.normalize()
    msr_.normalize()

    ref_cpx = ref_.get_measurement()
    msr_cpx = msr_.get_measurement()

    return torch.sum(torch.abs(msr_cpx - ref_cpx)**2)

def LogCoshLoss(ref, msr):
    ref_cpx = ref.get_measurement()
    msr_cpx = msr.get_measurement()

    abs_msr = torch.abs(msr_cpx) / torch.norm(torch.abs(msr_cpx))
    abs_ref = torch.abs(ref_cpx) / torch.norm(torch.abs(ref_cpx))

    return torch.mean(torch.log(torch.cosh(abs_msr - abs_ref)))

def CosSimLoss(ref, msr):
    ref_cpx = ref.get_measurement()
    msr_cpx = msr.get_measurement()

    ref_amp = torch.abs(ref_cpx)
    msr_amp = torch.abs(msr_cpx)

    return 1 - torch.sum(ref_amp * msr_amp) / (torch.norm(ref_amp) * torch.norm(msr_amp))


from scipy.stats import wasserstein_distance

def EMD_loss(ref, msr):
    ref = ref.get_measurement().cpu().detach().numpy()
    msr = msr.get_measurement().cpu().detach().numpy()

    dist = 0
    for t in range(sparams.mimo_system.get_param('N_Tx')):
        for r in range(sparams.mimo_system.get_param('N_Rx')):
            dist += wasserstein_distance(ref[t, r, :], msr[t, r, :])
    return torch.FloatTensor([dist]).to(config['device'])

def EMD_loss_FFT(ref, msr):
    ref = ref.get_measurement()
    msr = msr.get_measurement()

    dist = 0
    for t in range(sparams.mimo_system.get_param('N_Tx')):
        for r in range(sparams.mimo_system.get_param('N_Rx')):
            ref_fft = torch.fft.fft(ref[t, r, :]).cpu().detach().numpy()
            msr_fft = torch.fft.fft(msr[t, r, :]).cpu().detach().numpy()
            dist += wasserstein_distance(ref_fft, msr_fft)
    return torch.FloatTensor([dist]).to(config['device'])