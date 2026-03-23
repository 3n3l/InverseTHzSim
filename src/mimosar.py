import mitsuba as mi
import drjit as dr
import numpy as np
import torch

from config import config

class Antenna:
    def __init__(self, fwhm_azim, fwhm_elev):
        # Convert FWHM to standard deviations for azimuth and elevation
        self.sigma_azimuth = np.radians(fwhm_azim) / (2 * np.sqrt(2 * np.log(2)))
        self.sigma_elevation = np.radians(fwhm_elev) / (2 * np.sqrt(2 * np.log(2)))

        self.frame = mi.Frame3f(mi.Vector3f(1, 0, 0), mi.Vector3f(0, 1, 0), mi.Vector3f(0, 0, -1))

    def gaussian_pattern(self, incoming_dir):               
        theta = torch.acos(torch.Tensor(dr.dot(incoming_dir, mi.Vector3f(0, 0, -1))).to(config['device']))
        
        # Gaussian pattern for azimuth and elevation (normalized)
        attenuation_azimuth = torch.exp(-(theta ** 2) / (2 * self.sigma_azimuth ** 2))
        
        attenuation = attenuation_azimuth
        
        return attenuation
    
    def gaussian_pattern_2d(self, incoming_dir):
        incoming_dir = self.frame.to_local(incoming_dir)
                
        # Extract the azimuth and elevation angles from the vector
        azimuth = torch.arccos(torch.Tensor(incoming_dir.z).to(config['device']) / torch.sqrt(torch.Tensor(incoming_dir.x).to(config['device'])**2 + torch.Tensor(incoming_dir.z).to(config['device'])**2))  # Azimuth angle (phi) in radians
        elevation = torch.arccos(torch.Tensor(incoming_dir.z).to(config['device']) / torch.sqrt(torch.Tensor(incoming_dir.y).to(config['device'])**2 + torch.Tensor(incoming_dir.z).to(config['device'])**2))  # Elevation angle (theta) in radians
        
        # Gaussian pattern for azimuth and elevation (normalized)
        attenuation_azimuth = torch.exp(-(azimuth ** 2) / (2 * self.sigma_azimuth ** 2))
        attenuation_elevation = torch.exp(-(elevation ** 2) / (2 * self.sigma_elevation ** 2))
        
        # Combine the two Gaussian attenuations
        attenuation = attenuation_azimuth * attenuation_elevation
        
        return attenuation

class MIMOSAR:
    def __init__(self, path=None):
        self.params = {}
        if path:
            Tx, Rx, Fq = self.load_params(path)
            self.set_params(Tx, Rx, Fq)
        else:
            self.init_system_params()
            self.setup_mimosystem()
            self.generate_frequencies()
        self.rand_start_phase = False
        self.generate_starting_phase()

    def set_params(self, Tx, Rx, Fq):
        self.params = {'Tx': Tx, 'Rx': Rx, 'Fq': Fq, 'N_Tx': len(Tx), 'N_Rx': len(Rx), 'N_Fq': len(Fq)}

    def init_system_params(self, N_Tx=12, N_Rx=32, N_Fq=64):
        self._N_Tx = N_Tx         # Number of Transmitters
        self._N_Rx = N_Rx         # Number of Receivers
        self._N_Fq = N_Fq         # Number of Frequencies
        self.params['N_Tx'] = N_Tx
        self.params['N_Rx'] = N_Rx
        self.params['N_Fq'] = N_Fq
        self.Tx_mask = dr.full(dtype=mi.Bool, shape=self._N_Tx, value=True)
        self.Rx_mask = dr.full(dtype=mi.Bool, shape=self._N_Rx, value=True)

    def generate_frequencies(self, f_start=235E9, f_stop=270E9):
        self._Fq = dr.zeros(mi.Float, self._N_Fq)
        for f in range(self._N_Fq):
            dr.scatter(self._Fq, f_start + (f) * (f_stop-f_start)/(self._N_Fq-1), f)
        self.params['Fq'] = self._Fq

    def setup_mimosystem(self):
        self._Tx = dr.empty(mi.Vector3f, self._N_Tx)
        self._Tx.x = dr.linspace(mi.Float ,-0.187, 0.187, self._N_Tx)
        self._Tx.y = dr.ones(mi.Float, self._N_Tx) * 0.083
        self._Tx.z = dr.zeros(mi.Float, self._N_Tx)
        self.params['Tx'] = self._Tx

        self._Rx = dr.empty(mi.Vector3f, self._N_Rx)
        self._Rx.x = dr.linspace(mi.Float ,-0.143871, 0.143871, self._N_Rx)
        self._Rx.y = dr.zeros(mi.Float, self._N_Rx)
        self._Rx.z = dr.zeros(mi.Float, self._N_Rx)
        self.params['Rx'] = self._Rx

        self.params['antenna'] = Antenna(config['fwhm_azim'], config['fwhm_elev'])

    def generate_starting_phase(self):
        if self.rand_start_phase:
            np.random.seed(0)
            self.starting_phase = np.random.rand(self.params['N_Tx'], self.params['N_Fq']) * 2 * np.pi
        else:
            self.starting_phase = np.zeros((self.params['N_Tx'], self.params['N_Fq']))

    def randomize_starting_phase(self):
        self.rand_start_phase = True
        self.generate_starting_phase()

    def switch_off(self, Tx_idx=None, Rx_idx=None):
        if Tx_idx:
            for tidx in Tx_idx:
                if tidx < 0 or tidx >= self._N_Tx:
                    raise ValueError(f"Invalid Tx index: {tidx}")
                self.Tx_mask[tidx] = False
                self.params['Tx'] = dr.gather(mi.Vector3f, self._Tx, dr.compress(self.Tx_mask))
                self.params['N_Tx'] -= 1
                print("Switched off Tx: ", tidx)

        if Rx_idx:  
            self.Rx_mask[Rx_idx] = False
            self.params['Rx'] = dr.gather(mi.Vector3f, self._Rx, dr.compress(self.Rx_mask))
            self.params['N_Rx'] = (self._N_Rx - dr.count(~self.Rx_mask)).item()
            print("Switched off Rx: ", Rx_idx)

        self.generate_starting_phase()

    def get_param(self, param):
        return self.params[param]

    