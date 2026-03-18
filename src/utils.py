import mitsuba as mi
import drjit as dr
import numpy as np
import torch
import losses as optim
import os
import scipy.io as sio
import measurement


def clean_dr():
    dr.kernel_history_clear()
    dr.flush_malloc_cache()
    dr.malloc_clear_statistics()
    dr.flush_kernel_cache()


def perturb_sample(pt, n_samples, mu=0.0, sigma=0.0015):
    x = np.random.normal(mu, sigma, n_samples)
    y = np.random.normal(mu, sigma, n_samples)
    np.append(x, 0.0)
    np.append(y, 0.0)
    pert = mi.Vector3f(x, y, 0.0)
    pts = pt + pert
    return pts


from array import array
import sparams


def save_for_recon(real, imag, path):
    with open(path, "wb") as file:
        binary_data = array("f")
        # FIXME: why does this have a different type all the time?????
        for Rx in range(sparams.mimo_system.get_param("N_Rx")):
            for Tx in range(sparams.mimo_system.get_param("N_Tx")):
                for fq in range(sparams.mimo_system.get_param("N_Fq")):
                    binary_data.append(real[Tx, Rx, fq])
                    binary_data.append(imag[Tx, Rx, fq])
        file.write(binary_data)


import struct


def load_rdm(path):
    with open(path, "rb") as f:
        data = array("f")
        data = f.read()

    if (len(data) % 4) > 0:
        assert "Data length not a multiple of 4"
    L = []
    for i in range(0, len(data), 4):
        L.append(struct.unpack("f", data[i : i + 4]))

    return L


def convert_rdm_to_data(rdm):
    print("shape = ", rdm.shape)
    N_Tx = sparams.mimo_system.get_param("N_Tx")
    N_Rx = sparams.mimo_system.get_param("N_Rx")
    N_Fq = sparams.mimo_system.get_param("N_Fq")
    real = np.zeros((N_Tx, N_Rx, N_Fq))
    imag = np.zeros((N_Tx, N_Rx, N_Fq))
    for Tx in range(N_Tx):
        for Rx in range(N_Rx):
            for fq in range(N_Fq):
                idx = Rx * N_Tx * N_Rx + Tx * N_Fq + fq
                real[Tx, Rx, fq] = rdm[2 * idx][0]
                imag[Tx, Rx, fq] = rdm[2 * idx + 1][0]
    return real, imag


def load_real_measurement(path, NY):
    rdm = np.array(load_rdm(path))
    real = np.zeros((NY, sparams.N_Tx, sparams.N_Rx, sparams.N_Fq))
    imag = np.zeros((NY, sparams.N_Tx, sparams.N_Rx, sparams.N_Fq))
    for Y in range(NY):
        for Tx in range(sparams.N_Tx):
            for Rx in range(sparams.N_Rx):
                for fq in range(sparams.N_Fq):
                    idx = (
                        Y * sparams.N_Rx * sparams.N_Tx * sparams.N_Fq
                        + Rx * sparams.N_Tx * sparams.N_Fq
                        + Tx * sparams.N_Fq
                        + fq
                    )
                    real[Y, Tx, Rx, fq] = rdm[2 * idx]
                    imag[Y, Tx, Rx, fq] = rdm[2 * idx + 1]
    return real, imag


def fread(fid, nelements, dtype):
    data_array = np.fromfile(fid, dtype, nelements)
    data_array.shape = (nelements, 1)

    return data_array


def load_rdm_temp(path):
    with open(path, "rb") as f:
        data = array("d")
        data = f.read()

    if (len(data) % 8) > 0:
        assert "Data length not a multiple of 8"
    L = []
    for i in range(0, len(data), 8):
        L.append(struct.unpack("d", data[i : i + 8]))

    return L


def load_real_measurement_temp(path, NY):
    fid = open(path, "rb")
    rdm = fread(fid, 2 * NY * 31 * 12 * 64, np.float32)
    real = np.zeros((NY, 31, 12, 64))
    imag = np.zeros((NY, 31, 12, 64))
    for Y in range(NY):
        for Tx in range(12):
            for Rx in range(31):
                for fq in range(64):
                    idx = Y * 31 * 12 * 64 + Rx * 12 * 64 + Tx * 64 + fq
                    real[Y, Rx, Tx, fq] = rdm[2 * idx]
                    imag[Y, Rx, Tx, fq] = rdm[2 * idx + 1]
    return real, imag


def load_mat(file_name="data", avg=True, idx=1):
    datf = sio.loadmat("../data/rdm/real/" + file_name + ".mat")
    dat = datf.get("data_complex_all")
    dat = dat.flatten()

    n_msrs = int(len(dat) / (31 * 12 * 64))
    data = np.array(dat, dtype=np.complex64)

    # receiver index 11 is broken, transmitter index 2 is broken
    data_reshaped = np.reshape(data, (n_msrs, 31, 12, 64))
    data_reshaped = np.swapaxes(data_reshaped, 1, 2)
    data_reshaped = data_reshaped[:, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11], :, :]

    return torch.Tensor(data_reshaped.real), torch.Tensor(data_reshaped.imag)


def save_and_test(msr, path):
    msr.save(path)

    loaded = measurement.RDM(path)

    if optim.torch_l1_loss(msr, loaded).data < 1e-5:
        print("Saved Successfully")
    else:
        print("Data Mismatch")


def get_all_file_names(dir):
    return [f.split(".rdm")[0] for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]


def check_path_empty(path):
    if os.path.exists(path) and not os.path.isfile(path):
        if len(os.listdir(path)) == 0:
            return True
        else:
            return False


def make_rdm_path(name, path):
    if isinstance(name, str):
        path = path + name + ".rdm"
    else:
        path = path + str(name) + ".rdm"
    return path


import json
import config


def save_config(file_name):
    with open("../data/configurations/" + file_name, "w") as f:
        json.dump(config.config, f, indent=4)


def load_config(file_name):
    with open("../data/configurations/" + file_name, "r") as f:
        new_conf = json.load(f)
        config.config.clear()
        config.config.update(new_conf)
