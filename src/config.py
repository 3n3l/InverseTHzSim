config = {
    # Optimization Settings
    "op_param": "pos",  # optimization parameter: pos, radius, material
    "op_dim": 1,  # optimization dimension: 1D, 2D, 3D
    "ref_position": 4,  # reference position for the object (check sparams.py for the list of options)
    "ref_radius": 1,  # reference radius for the sphere (check sparams.py for the list of options)
    "use_sim_ref": False,  # simulate the reference measurement
    # ---------------------------------------------------------------
    # Simulation Settings
    "object": "sphere",  # object to be simulated: sphere, torus, plane
    "N_Paths": 1000000,  # number of rays/samples per transmitter
    "sim_depth": 1,  # number of ray bounces
    "sampling_strat": "shape",  # sampling strategy: shape, all_shapes, hemi, cone, disk
    "material": "phong",  # material type: phong, rough, diffuse
    "use_sms": False,  # use Specular Manifold Sampling
    "relax_physics": True,  # relax physical parameters for the simulation
    # Phong Material Settings
    "phong_kd": 0.03,  # diffuse reflectance for phong material
    "phong_ks": 0.97,  # specular reflectance for phong material
    "phong_ns": 1200,  # specular exponent for phong material
    # RoughConductor Material Settings
    "roughness": 0.2,  # roughness for roughconductor material
    # Antenna Settings
    "fwhm_azim": 40,  # 19.54                 # full width at half maximum for azimuth
    "fwhm_elev": 40,  # 19.99                 # full width at half maximum for elevation
    "apply_antenna": True,  # apply antenna pattern to the simulation
    # MIMO SAR System Settings
    "tx_mask": [2],  # transmitters to be switched off
    "rx_mask": [10],  # receivers to be switched off
    # --------------------------------------------------------------
    # Calibration Settings
    "cal_strat": "simulated",  # calibration strategy: self, simulated, injection
    "cal_tilt": 0,  # tilt applied to simulated calibration plane
    # ---------------------------------------------------------------
    # Reconstruction Settings
    "recon_grid_samples": [200, 200, 100],  # number of samples in the x, y, z directions
    "recon_grid_lims": [-0.187, 0.187, 0.0, 0.083, 0.25, 0.35],  # grid limits in the x, y, z directions
    "recon_cal": True,  # use calibrated reconstruction
    "recon_short": 0.345,  # short distance for calibrated reconstruction
    "recon_grid_normalize": True,  # normalize the reconstruction grid
    "recon_min_db": -15,  # max dB for the reconstruction plot
    "recon_max_db": 0,  # min dB for the reconstruction plot
    # ----------------------------------------------------------------
    # Runtime Settings
    "render_grad": False,  # calculate gradients while rendering
    "plot_sim": False,  # plot figures for testing
    "print_stats": False,  # print useful stats for debugging
    # -------------------------------------------------------------
}
