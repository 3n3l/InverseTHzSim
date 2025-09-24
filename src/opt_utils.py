import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi

import utils
import measurement
import setup_utils
import sparams
from config import config

def get_x_from_path(path):
    x_axis = utils.get_all_file_names(path)
    x_axis = [float(x) for x in x_axis]
    return [x_axis[i] for i in np.argsort(x_axis)]

def plot_1D_loss(x, y, title, x_label, y_label):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def compute_next_measurement(theta, scene, transform, randomize_phase, save, path, cal_path):
    transform(theta)

    msr = measurement.Simulated(scene, ms=False)
    if save:
        msr.save(f'{path}{theta}.rdm')
    if randomize_phase:
        msr.calibrate(cal_path)
    return msr

def loss_landscape_1D(ref, x_axis, scene, transform, losses, titles, x_label="x-axis (pos)", compute=True, save=True):
    path = setup_utils.save_path()
    cal_path = setup_utils.cal_path()
    hist = [[] for _ in range(len(losses))]
    randomize_phase = (config['cal_strat'] == 'simulated') or (config['cal_strat'] == 'self')
    for x in x_axis:
        if compute:
            msr = compute_next_measurement(x, scene, transform, randomize_phase, save, path, cal_path)
        else:
            msr = measurement.RDM(utils.make_rdm_path(x, path))
            if randomize_phase:
                msr.calibrate(cal_path)
            else:
                msr.calibrate(match='calibrated64/Calibration_files/Match')
        for i in range(len(losses)):
            hist[i].append(losses[i](ref, msr).item())

    for i in range(len(losses)):
        plot_1D_loss(x_axis, hist[i], titles[i], x_label, "Loss")
        if save:
            np.array(hist[i]).tofile(f'../data/loss_landscapes/1D/losses1D_{titles[i]}.dat')

def find_min_1D(title, x_axis):
    loaded = np.fromfile(f'../data/loss_landscapes/1D/losses1D_{title}.dat', dtype=float).reshape(len(x_axis))
    idx = np.unravel_index(np.argmin(loaded, axis=None), loaded.shape)
    return x_axis[idx[0]]

def print_min_1D(titles, x_axis):
    print("Minimum values for 1D loss landscapes:")
    print("===================================")
    for title in titles:
        min_x = find_min_1D(title, x_axis)
        print(f"{title}: {min_x}")

def plot_2D_loss(x, y, z, title, x_label, y_label):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    Z = np.array(z)
    Z = Z.reshape(X.shape)

    ax.plot_surface(X, Y, Z, cmap='jet')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel('Loss')
    fig.suptitle('2d loss surface')

    plt.show()

    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, levels=14, linewidths=0.5, colors='k')
    cntr = ax.contourf(X, Y, Z, levels=14, cmap='jet')
    fig.colorbar(cntr, ax=ax)
    fig.suptitle('2d loss contour')
    plt.title(title)
    plt.show()

def loss_landscape_2D(ref, x_axis, y_axis, scene, transform, losses, titles, x_label="X", y_label="Y", compute=True, save=False):
    path = setup_utils.save_path()
    cal_path = setup_utils.cal_path()
    hist = [[] for _ in range(len(losses))]
    randomize_phase = (config['cal_strat'] == 'simulated') or (config['cal_strat'] == 'self')
    for x in x_axis:
        for y in y_axis:
            theta = mi.Point2f(x, y)
            if compute:
                msr = compute_next_measurement(theta, scene, transform, randomize_phase, save, path, cal_path)
            else:
                m_path = utils.make_rdm_path(theta, path)
                msr = measurement.RDM(m_path)
                if randomize_phase:
                    msr.calibrate(cal_path)
                else:
                    msr.calibrate(match='calibrated64/Calibration_files/Match')
            for i in range(len(losses)):
                hist[i].append(losses[i](ref, msr).item())

    for i in range(len(losses)):
        plot_2D_loss(x_axis, y_axis, hist[i], titles[i], x_label, y_label)
        if save:
            np.array(hist[i]).tofile(f'../data/loss_landscapes/2D/losses2D_{titles[i]}.dat')

def find_min_2D(title, x_axis, y_axis):
    loaded = np.fromfile(f'../data/loss_landscapes/2D/losses2D_{title}.dat', dtype=float).reshape(len(x_axis), len(y_axis))
    idx = np.unravel_index(np.argmin(loaded, axis=None), loaded.shape)
    return x_axis[idx[0]], y_axis[idx[1]]

def print_min_2D(titles, x_axis, y_axis):
    print("Minimum values for 2D loss landscapes:")
    print("===================================")
    for title in titles:
        min_x, min_y = find_min_2D(title, x_axis, y_axis)
        print(f"{title}: {min_x}, {min_y}")

def loss_landscape_3D(ref, x_axis, y_axis, z_axis, params, ctx_args, transform, losses, titles, compute=True, save=True, path="../data/msr_3D/no_brdf/"):
    hist = [[] for _ in range(len(losses))]
    for x in x_axis:
        for y in y_axis:
            for z in z_axis:
                theta = mi.Point3f(x, y, z)
                if compute:
                    msr = compute_next_measurement(theta, params, ctx_args, transform, save, path)
                else:
                    msr = utils.load_measurement(theta, path)
                for i in range(len(losses)):
                    hist[i].append(losses[i](ref, msr).item())

    for i in range(len(losses)):
        if save:
            np.array(hist[i]).tofile(f'../data/loss_landscapes/3D/losses3D_{titles[i]}.dat')

def compute_loss_landscape(ref_cpx, scene, update_fn, loss_fns, titles, x_axis=None, y_axis=None, z_axis=None, x_label="X", y_label="Y", z_label="Z"):
    compute = utils.check_path_empty(setup_utils.save_path())
    print("Computing loss landscape...") if compute else print("Loading loss landscape...")
    if config['op_dim'] == 1:
        if x_axis is None:
            #x_axis = get_x_from_path(setup_utils.save_path())
            x_axis, x_label = get_x_axis()
        loss_landscape_1D(ref_cpx, x_axis, scene, update_fn, loss_fns, titles, x_label=x_label, compute=compute, save=True)
        print_min_1D(titles, x_axis)
    elif config['op_dim'] == 2:
        if x_axis is None:
            x_axis, x_label = get_x_axis()
        if y_axis is None:
            y_axis, y_label = get_y_axis()
        loss_landscape_2D(ref_cpx, x_axis, y_axis, scene, update_fn, loss_fns, titles, x_label=x_label, y_label=y_label, compute=compute, save=True)
        print_min_2D(titles, x_axis, y_axis)

def get_x_axis():
    if config['op_param'] == 'pos':
        if config['object'] == 'plane':
            x_axis = np.linspace(0.3, 0.4, 100)
            x_label = "Height (m)"
        else:
            x_axis = np.linspace(-0.2, 0.2, 100)
            for pos in sparams.measured_sphere_pos.values():
                x_axis = np.concatenate((x_axis[x_axis<pos[0]], [pos[0]], x_axis[x_axis>pos[0]]))
            x_label = "X-position (m)"
    elif config['op_param'] == 'pos_z':
        x_axis = np.linspace(0.3, 0.4, 100)
        x_label = "Height (m)"
        
    elif config['op_param'] == 'radius':
        x_axis = np.linspace(0.001, 0.1, 100)
        for rad in sparams.measured_sphere_rad.values():
            x_axis = np.concatenate((x_axis[x_axis<rad], [rad], x_axis[x_axis>rad]))
        x_label = "Radius (m)"
    elif config['op_param'] == 'material':
        if config['material'] == 'phong':
            if config['phong_ns'] == None:
                x_axis = np.linspace(1, 25, 25)
                x_label = "Phong Specular Exponent (Ns)"
            else:
                x_axis = np.linspace(0.0, 1.0, 100)
                x_label = "Phong Diffuse Weight (Kd)"
        elif config['material'] == 'rough':
            x_axis = np.linspace(0.001, 0.7, 70)
            x_label = "Roughness (alpha)"
    return x_axis, x_label

def get_y_axis():
    if config['op_param'] == 'pos':
        y_axis = np.linspace(-0.05, 0.1, 80)
        y_axis = np.concatenate((y_axis[y_axis<0], [0], y_axis[y_axis>0]))
        y_label = "Y-position (m)"
    elif config['op_param'] == 'material':
        if config['material'] == 'phong':
            y_axis = np.linspace(50, 500, 10)
            y_label = "Phong Specular Exponent (Ns)"
    elif config['op_param'] == 'radius':
        y_axis = np.linspace(0.150, 0.330, 10)
        for rad in sparams.measured_sphere_rad.values():
            y_axis = np.concatenate((y_axis[y_axis<330-rad], [330 - rad], y_axis[y_axis>330-rad]))
        y_label = "Z-position (m)"
    return y_axis, y_label