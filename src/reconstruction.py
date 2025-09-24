import numpy as np
import drjit as dr
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator

from config import config
import sparams

def reconstruct(Tx, Rx, Fq, measurement, rc_grid):
    k = (2 * np.pi * Fq) / sparams.c

    reconstruction = torch.zeros(len(rc_grid), dtype=torch.complex64).to(config['device'])

    for t in range(len(Tx)):
        for r in range(len(Rx)):
            for i in range(len(rc_grid)):

                d1 = torch.sqrt(torch.sum((Tx[t] - rc_grid[i])**2))
                d2 = torch.sqrt(torch.sum((Rx[r] - rc_grid[i])**2))

                phase = torch.zeros([64], dtype=torch.complex64).to(config['device'])
                phase.real += torch.cos(1 * (d1 + d2) * k)
                phase.imag -= torch.sin(1 * (d1 + d2) * k)

                corr = torch.sum(measurement[t, r, :] * torch.conj(phase))
                reconstruction[i] += corr

    return reconstruction.real, reconstruction.imag

def fast_reconstruct(Tx, Rx, Fq, measurement, rc_grid):
    max_size = 1000
    div = int(len(rc_grid) / max_size)
    n_batches = div if (len(rc_grid) % max_size == 0) else div + 1
    start = 0
    end = len(rc_grid) if (n_batches == 1) else max_size

    reconstruction = torch.zeros(len(rc_grid), dtype=torch.complex64).to(config['device'])

    k = (2 * np.pi * Fq) / sparams.c
    k = k.reshape(1, 1, len(Fq), 1)

    for batch in range(n_batches):
        grid = rc_grid[start:end, :]

        d1 = torch.norm(Tx[:, None, :] - grid[None, :, :], dim=-1)
        d2 = torch.norm(Rx[:, None, :] - grid[None, :, :], dim=-1)

        d = d1[:, None]+d2

        phase = d[:, :, None, :] * k

        temp = torch.exp(-1j*phase).to(config['device'])
        corr = torch.sum(measurement[:, :, :, None] * torch.conj(temp), (0, 1, 2))
        reconstruction[start:end] += corr

        start = end
        end = len(rc_grid) if (len(rc_grid) - start < max_size) else start + max_size

    return reconstruction

def fast_reconstruct_cal(Tx, Rx, Fq, measurement, rc_grid, short=0.345):
    max_size = 1000
    div = int(len(rc_grid) / max_size)
    n_batches = div if (len(rc_grid) % max_size == 0) else div + 1
    start = 0
    end = len(rc_grid) if (n_batches == 1) else max_size

    reconstruction = torch.zeros(len(rc_grid), dtype=torch.complex64).to(config['device'])

    k = (2 * np.pi * Fq) / sparams.c
    k = k.reshape(1, 1, len(Fq), 1)

    for batch in range(n_batches):
        grid = rc_grid[start:end, :]

        d1 = torch.norm(Tx[:, None, :] - grid[None, :, :], dim=-1)
        d2 = torch.norm(Rx[:, None, :] - grid[None, :, :], dim=-1)

        hv = torch.norm(Tx[:, None, :] - Rx[None, :, :], dim=-1) / 2.0

        dc = 2 * torch.sqrt(hv**2 + short**2)
        dc = torch.reshape(dc, (len(Tx), len(Rx), 1))

        d = d1[:, None]+d2

        d = d - dc

        phase = d[:, :, None, :] * k

        temp = torch.exp(-1j*phase).to(config['device'])
        corr = torch.sum(measurement[:, :, :, None] * torch.conj(temp), (0, 1, 2))
        reconstruction[start:end] += corr

        start = end
        end = len(rc_grid) if (len(rc_grid) - start < max_size) else start + max_size

    return reconstruction

class RecGrid:
    def __init__(self):
        self.N_samples = config['recon_grid_samples']
        self.grid_lims = config['recon_grid_lims']
        self.__create_grid()

    def __create_grid(self):
        self.grid_x = torch.linspace(self.grid_lims[0], self.grid_lims[1], self.N_samples[0])
        self.grid_y = torch.linspace(self.grid_lims[2], self.grid_lims[3], self.N_samples[1])
        self.grid_z = torch.linspace(self.grid_lims[4], self.grid_lims[5], self.N_samples[2])

        zz, yy, xx = torch.meshgrid(self.grid_z, self.grid_y, self.grid_x, indexing='ij')
        self.orig_grid = torch.stack([xx, yy, zz], dim=-1).to(config['device'])

        self.rc_grid = self.orig_grid.reshape(self.N_samples[0] * self.N_samples[1] * self.N_samples[2], -1)

    def reconstruct(self, measurement):
        Tx = torch.Tensor(dr.ravel(sparams.mimo_system.get_param('Tx'))).reshape([sparams.mimo_system.get_param('N_Tx'), 3]).to(config['device'])
        Rx = torch.Tensor(dr.ravel(sparams.mimo_system.get_param('Rx'))).reshape([sparams.mimo_system.get_param('N_Rx'), 3]).to(config['device'])
        Fq = torch.Tensor(sparams.mimo_system.get_param('Fq')).to(config['device'])

        if config['recon_cal']:
            self.rec = fast_reconstruct_cal(Tx, Rx, Fq, measurement, self.rc_grid, config['recon_short'])
        else:
            self.rec = fast_reconstruct(Tx, Rx, Fq, measurement, self.rc_grid)
        self.reshaped = self.rec.reshape(self.orig_grid.shape[0:3], -1)

    def get_max_index(self):
        print(self.orig_grid.shape)
        idx = np.unravel_index(np.argmax(torch.abs(self.reshaped).cpu().numpy(), axis=None), self.reshaped.shape)
        return idx[2], idx[1], idx[0]
    
    def get_max_value(self, idx_y, idx_z):
        return torch.max(torch.abs(self.reshaped[:, idx_y, idx_z])).cpu().numpy()

    def print_slice(self, x, y, z):
        print(self.orig_grid[z, y, x, :].cpu().numpy())

    def get_slice(self, x, y, z):
        return self.orig_grid[z, y, x, :].cpu().numpy()

    def normalize(self):
        max_val = torch.max(torch.abs(self.reshaped))
        self.reshaped = self.reshaped / max_val

    def plot_x_slice(self, y, z, min_db, max_db):
        mag_x = torch.abs(self.reshaped[z, y, :])
        log_mag_x = 20*torch.log10(mag_x)

        mag_final_x = log_mag_x.cpu().numpy()
        
        fig, ax = plt.subplots()
        ax.plot(self.grid_x, mag_final_x,'g')
        ax.axis([self.grid_lims[0], self.grid_lims[1], min_db, max_db])
        plt.xlabel('X [m]')
        plt.ylabel('dB')
        plt.show()

    def plot_xz(self, y, min_db, max_db):
        mag_xz = torch.abs(self.reshaped[:, y, :])
        log_mag_xz = 20*torch.log10(mag_xz)

        mag_final_xz = log_mag_xz.cpu().numpy()

        fig, ax = plt.subplots()
        pos = ax.imshow(mag_final_xz,cmap=plt.cm.gray, vmin=min_db, vmax=max_db)
        xlabels = np.linspace(self.grid_lims[0], self.grid_lims[1], 9)
        xlabels = [round(label, 2).astype(str) for label in xlabels[1:-1]]
        ax.xaxis.set_major_locator(LinearLocator(numticks=9))
        ax.set_xticks(ax.get_xticks().tolist()[1:-1])
        ax.set_xticklabels(xlabels)
        plt.xlabel('X [m]')
        ylabels = np.linspace(self.grid_lims[4], self.grid_lims[5], 6)
        ax.yaxis.set_major_locator(LinearLocator(numticks=6))
        ax.set_yticks(ax.get_yticks().tolist())
        ax.set_yticklabels([round(label, 2).astype(str) for label in ylabels])
        plt.ylabel('Z [m]')

        plt.colorbar(pos, ax=ax, location="right", shrink=0.5)
        plt.show()

    def plot_xy(self, z, min_db, max_db):
        mag_xy = torch.abs(self.reshaped[z, :, :])
        log_mag_xy = 20*torch.log10(mag_xy)

        mag_final_xy = log_mag_xy.cpu().numpy()

        fig, ax = plt.subplots()
        pos = ax.imshow(mag_final_xy,cmap=plt.cm.gray, vmin=min_db, vmax=max_db)
        xlabels = np.linspace(self.grid_lims[0], self.grid_lims[1], 9)
        xlabels = [round(label, 2).astype(str) for label in xlabels[1:-1]]
        ax.xaxis.set_major_locator(LinearLocator(numticks=9))
        ax.set_xticks(ax.get_xticks().tolist()[1:-1])
        ax.set_xticklabels(xlabels)
        plt.xlabel('X [m]')
        ylabels = np.linspace(self.grid_lims[2], self.grid_lims[3], 9)
        ax.yaxis.set_major_locator(LinearLocator(numticks=9))
        ax.set_yticks(ax.get_yticks().tolist())
        ax.set_yticklabels([round(label, 2).astype(str) for label in ylabels])
        plt.ylabel('Y [m]')

        plt.colorbar(pos, ax=ax, location="right", shrink=0.5)
        plt.show()