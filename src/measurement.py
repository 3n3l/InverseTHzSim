import torch
import numpy as np
from config import config
import utils
import simulation as sim
import sparams
import reconstruction as recon
import matplotlib.pyplot as plt

class Measurement:
    def __init__(self, msr):
        self.msr = msr
        self.normalized = False

    def get_measurement(self):
        return self.msr
    
    def get_phase_only(self):
        phase = torch.angle(self.msr)
        return torch.exp(1j*phase)
    
    def normalize(self):
        if(self.normalized):
            return
        
        self.normalized = True

        amp = torch.abs(self.msr)

        normalized_amp = amp / torch.linalg.norm(amp, dim=2, keepdim=True)
        normalized_amp = torch.where(torch.isnan(normalized_amp), torch.zeros_like(normalized_amp), normalized_amp)

        phase = torch.angle(self.msr)
        self.msr = normalized_amp * torch.exp(1j*phase)
    
    def add_noise(self, mag = 0.1):
        np.random.seed(0)
        msr = self.get_measurement()
        ref_abs = torch.abs(msr) + mag * torch.rand(sparams.mimo_system.get_param('N_Tx'), sparams.mimo_system.get_param('N_Rx'), sparams.mimo_system.get_param('N_Fq')).to(config['device'])
        ref_angle = torch.angle(msr)
        self.msr = ref_abs * torch.exp(1j * ref_angle)

    def print_statistics(self):
        print('Max amplitude', torch.max(torch.abs(self.msr)).item())
        print('Min amplitude', torch.min(torch.abs(self.msr)).item())
        print('Mean amplitude', torch.mean(torch.abs(self.msr)).item())
        print('-----------------------------------')

    def plot_signal(self, t_idx=None, r_idx=None):
        msr = self.get_measurement()
        if t_idx is not None and r_idx is not None:
            amps = torch.abs(msr[t_idx, r_idx])
            phases = torch.angle(msr[t_idx, r_idx])
        elif t_idx is not None:
            amps = torch.abs(msr[t_idx]).mean(dim=-1)
            phases = torch.angle(msr[t_idx]).mean(dim=-1)
        elif r_idx is not None:
            amps = torch.abs(msr[:, r_idx]).mean(dim=-1)
            phases = torch.angle(msr[:, r_idx]).mean(dim=-1)
        else:
            amps = torch.abs(msr).mean(dim=(0,1))
            phases = torch.angle(msr).mean(dim=(0,1))

        plt.switch_backend('module://matplotlib_inline.backend_inline')
        plt.ioff()
        plt.figure(figsize=(10, 5))
        plt.suptitle('Signal per Frequency')
        plt.subplot(1, 2, 1)
        plt.plot(amps.cpu().numpy())
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.title('Amplitude')
        plt.subplot(1, 2, 2)
        plt.plot(phases.cpu().numpy())
        plt.xlabel('Frequency')
        plt.ylabel('Phase')
        plt.title('Phase')
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.suptitle('Signal per Tx')
        plt.plot(torch.abs(msr).mean(dim=-1).cpu().numpy())
        plt.xlabel('Tx')
        plt.ylabel('Amplitude')
        plt.title('Amplitude')
        plt.show()

        plt.switch_backend('module://ipympl.backend_nbagg')
        ax = plt.figure(figsize=(10, 5)).add_subplot(projection='3d')
        plt.suptitle('Signal per Tx-Rx')
        X = np.arange(msr.shape[0])
        Y = np.arange(msr.shape[1])
        X, Y = np.meshgrid(Y, X)
        Z = torch.abs(msr).mean(dim=-1).cpu().numpy()
        surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
        plt.title('Amplitude')
        plt.xlabel('Rx')
        plt.ylabel('Tx')
        plt.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        plt.switch_backend('agg')
        

    def calibrate(self, short='calibrated64/Calibration_files/Short', match=None):
        if config['cal_strat'] == 'self':
            cal = Measurement(self.msr)
        else:
            cal = Calibration(short, match)

        if not self.settings['random_phase']:
            starting_signal = cal.extract_original_signal()
            starting_phase_only = starting_signal.get_phase_only()
            self.msr = self.msr * starting_phase_only
        elif match is not None:
            self.msr = self.msr - cal.get_match().get_measurement()
        
        self.msr = self.msr / cal.get_phase_only()

    def reconstruct(self):
        rc_grid = recon.RecGrid()
        rc_grid.reconstruct(self.get_measurement())
        if config['recon_grid_normalize']:
            rc_grid.normalize()
        x_idx, y_idx, z_idx = rc_grid.get_max_index()
        rc_grid.print_slice(x_idx, y_idx, z_idx)
        rc_grid.plot_x_slice(y_idx, z_idx, config['recon_min_db'], config['recon_max_db'])
        rc_grid.plot_xz(y_idx, config['recon_min_db'], config['recon_max_db'])
        rc_grid.plot_xy(z_idx, config['recon_min_db'], config['recon_max_db'])
        return rc_grid

class Real(Measurement):
    def __init__(self, path):
        real, imag = utils.load_mat(path)
        self.msr = torch.complex(real, imag).to(config['device'])

        self.settings = {}
        self.settings['random_phase'] = True
        self.normalized = False

    def get_measurement(self, index=None):
        if len(self.msr.shape) == 3:
            return self.msr
        elif index is not None:
            return self.msr[index]
        return torch.mean(self.msr, axis=0)

    def print_statistics(self):
        N = self.msr.shape[0]
        print('Number of measurements: ', N)
        print('Max amplitude', torch.max(torch.abs(self.msr)).item())
        print('Min amplitude', torch.min(torch.abs(self.msr)).item())
        print('Mean amplitude', torch.mean(torch.abs(self.msr)).item())
        print('-----------------------------------')
        if N < 2:
            return
        print('Pairwise differences:')
        diffs = torch.zeros(N, N, 1).to(config['device']) # Pairwise differences

        max_size = 1000
        div = int(N**2 / max_size)
        n_batches = div if (N**2 % max_size == 0) else div + 1

        for batch in range(n_batches):
            start = batch * max_size
            end = min(N**2, (batch + 1) * max_size)
            i, j = np.unravel_index(np.arange(start, end), (N, N))
            diffs[i, j] = torch.sum(torch.abs(self.msr[i] - self.msr[j])/torch.linalg.norm(torch.abs(self.msr[i])), axis=(1, 2, 3)).reshape(-1, 1)

        mask = ~torch.eye(N, dtype=torch.bool)
        mask = mask.unsqueeze(-1)  # Shape: [N, N, 1]
        masked_diffs = diffs[mask].view(N, N-1, 1)

        max_diff = torch.max(masked_diffs)
        print("Max difference:", max_diff.item())

        min_diff = torch.min(masked_diffs)
        print("Min difference:", min_diff.item())

        avg_diff = torch.mean(masked_diffs)
        print("Average difference:", avg_diff.item())

        std_diff = torch.std(masked_diffs)
        print("Standard deviation of differences:", std_diff.item())

        print('-----------------------------------')

        avg_diffs_per_measurement = torch.mean(masked_diffs, dim=1)

        most_representative_idx = torch.argmin(avg_diffs_per_measurement)
        most_representative_diff = avg_diffs_per_measurement[most_representative_idx]

        mean_avg_diff = torch.mean(avg_diffs_per_measurement)
        std_avg_diff = torch.std(avg_diffs_per_measurement)
        outlier_threshold = mean_avg_diff + 2 * std_avg_diff
        outlier_indices = torch.where(avg_diffs_per_measurement > outlier_threshold)[0]

        print("Most representative measurement index:", most_representative_idx.item())
        print("Most representative measurement average difference:", most_representative_diff.item())
        print("Threshold for outliers (avg_diff + 2 * std): ", outlier_threshold.item())
        print("Number of outliers:", len(outlier_indices))
        print("Outlier measurement indices:", outlier_indices.tolist())
        print("Outlier measurement average differences:", avg_diffs_per_measurement[outlier_indices].tolist())
        print('-----------------------------------')

class RDM(Measurement):
    def __init__(self, path):
        rdm = np.array(utils.load_rdm(path))
        real, imag = utils.convert_rdm_to_data(rdm)
        real = torch.Tensor(real).to(config['device'])
        imag = torch.Tensor(imag).to(config['device'])
        self.msr = torch.complex(real, imag).to(config['device'])

        self.settings = {}
        self.settings['random_phase'] = (config['cal_strat'] == 'simulated' or config['cal_strat'] == 'self')
        self.normalized = False

    def print_statistics(self):
        print("Loaded Measurement")
        print("Random Phase: ", self.settings['random_phase'])
        print("-----------------------------------")
        print('Max amplitude', torch.max(torch.abs(self.msr)).item())
        print('Min amplitude', torch.min(torch.abs(self.msr)).item())
        print('Mean amplitude', torch.mean(torch.abs(self.msr)).item())
        print('-----------------------------------')

class Calibration(Measurement):
    def __init__(self, short='calibrated64/Calibration_files/Short', match=None):
        self.settings = {}
        self.settings['match'] = (match is not None)

        if config['cal_strat'] == 'simulated':
            self.short = RDM(short)
        else:
            self.short = Real(short)
        
        self.msr = self.short.get_measurement()

        if match is not None:
            self.match = Real(match)
            self.msr = self.msr - self.match.get_measurement()

    def extract_original_signal(self, path='../data/calibration/real/plane_approx.rdm'):
        approx = RDM(path).get_measurement()
        return Measurement(self.msr / approx)
    
    def get_match(self):
        if self.settings['match']:
            return self.match
        print("No match signal found")

    def get_short(self):
        return self.short
    
    def print_statistics(self):
        print("Calibration with the following settings:")
        print("Simulated: ", (config['cal_strat'] == 'simulated'))
        print("Use Match: ", self.settings['match'])
        print("-----------------------------------")
        print('Max amplitude', torch.max(torch.abs(self.msr)).item())
        print('Min amplitude', torch.min(torch.abs(self.msr)).item())
        print('Mean amplitude', torch.mean(torch.abs(self.msr)).item())
        print('-----------------------------------')

class Simulated(Measurement):
    def __init__(self, scene, ms=False):
        self.settings = {}
        self.settings['scene_name'] = scene.get_name()
        self.settings['random_phase'] = (config['cal_strat'] == 'simulated' or config['cal_strat'] == 'self')
        self.settings['ms'] = ms
        self.normalized = False

        if self.settings['random_phase']:
            sparams.mimo_system.randomize_starting_phase()

        self.ctx_args = {
            'scene': scene.get_scene()
        }

        real, imag = sim.run_simulation(self.ctx_args, use_ms=ms)
        self.msr = torch.complex(real, imag).to(config['device'])

    def save(self, path):
        utils.save_for_recon(self.msr.real, self.msr.imag, path)

    def print_statistics(self):
        print("Simulated measurement with the following settings:")
        print("Simulated Scene: ", self.settings['scene_name'])
        print("Random Phase: ", self.settings['random_phase'])
        print("Use Manifold Sampling: ", self.settings['ms'])
        print("-----------------------------------")
        print('Max amplitude', torch.max(torch.abs(self.msr)).item())
        print('Min amplitude', torch.min(torch.abs(self.msr)).item())
        print('Mean amplitude', torch.mean(torch.abs(self.msr)).item())
        print('-----------------------------------')





    
