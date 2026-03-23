import mitsuba as mi
import drjit as dr
import numpy as np

import torch

import sparams
from config import config
import sampler as smp
import manifold as ms
import visualization as viz

def connect_to_Rx(si, scene, Rx, indices):
    temp = Rx - dr.gather(mi.Point3f, si.p, mi.UInt(dr.arange(mi.UInt, 0, config['N_Paths'])))
    dist_squared = dr.dot(temp, temp)
    dist_squared = dr.gather(mi.Float, dist_squared, indices)
    Rx_dist = dr.sqrt(dist_squared)

    wo = dr.normalize((temp))
    wi = dr.gather(mi.Vector3f, si.wi, indices)

    antenna_atten = sparams.mimo_system.get_param('antenna').gaussian_pattern_2d(wo)

    shad_ray = si.spawn_ray(wo)

    wo = si.to_local(wo)

    wo = dr.gather(mi.Vector3f, wo, indices)

    norms = dr.gather(mi.Vector3f, si.to_local(si.n), indices)

    throughput = dr.gather(mi.Float, mi.Float(antenna_atten), indices) 
    if not config['relax_physics']:
        throughput *= 1/(4*np.pi*dist_squared) * dr.select(dr.dot(wo, norms)>0.0, dr.dot(wo, norms), 0.0)

    if config['material'] == 'rough':
        ctx = mi.BSDFContext()
        bsdf = si.bsdf()
        bsdf_val = bsdf.eval(ctx, si, wo)
        throughput = throughput * dr.gather(mi.Float, bsdf_val.x, indices)
    elif config['material'] == 'diffuse':
        throughput = throughput * dr.dot(wi, norms)
    elif config['material'] == 'phong':
        r = ms.reflect(dr.gather(mi.Vector3f, si.wi, indices), dr.gather(mi.Vector3f, norms, indices))
        if config['relax_physics']:
            throughput = throughput * ((config['phong_kd'] * dr.dot(dr.gather(mi.Vector3f, si.wi, indices), dr.gather(mi.Normal3f, si.n, indices))) + config['phong_ks'] * dr.select((dr.dot(r, wo) > 0.0), (dr.power(dr.dot(r, wo), config['phong_ns'])), 0.0))
        else:
            diffuse = config['phong_kd'] / np.pi
            specular = config['phong_ks'] * (config['phong_ns'] + 2) / (2*np.pi) * dr.select((dr.dot(r, wo) > 0.0), (dr.power(dr.dot(r, wo), config['phong_ns'])), 0.0)
            cos_term = dr.select(dr.dot(wi, norms) > 0.0, dr.dot(wi, norms), 0.0)
            throughput = throughput * (diffuse + specular) * cos_term

    no_path = dr.gather(mi.Bool, scene.ray_test(shad_ray), indices)

    throughput[no_path] = 0.0

    return Rx_dist, throughput

@dr.wrap_ad(source='drjit', target='torch')
def get_measurement(real, imag, path_dists, path_amps, T_idx, R_idx, Fq):
    measurement = torch.complex(real, imag).to(config['device'])

    n_paths = len(path_dists)
    max_size = 5000000
    div = int(n_paths / max_size)
    n_batches = div if (n_paths % max_size == 0) else div + 1
    start = 0
    end = n_paths if (n_batches == 1) else max_size
    for batch in range(n_batches):
        d_batch = path_dists[start:end]
        a_batch = path_amps[start:end]

        if config['relax_physics']:
            sqrt_batch = torch.sqrt(a_batch)
            a_batch = torch.where(torch.isnan(sqrt_batch), 0.0, sqrt_batch)

        for f in range(sparams.mimo_system.get_param('N_Fq')):
            if not config['relax_physics']:
                sqrt_batch = torch.sqrt(a_batch * (sparams.c/Fq[f])**2)
                a_batch = torch.where(torch.isnan(sqrt_batch), 0.0, sqrt_batch)

            k = 2 * np.pi * Fq[f] / sparams.c
            phase = -1 * d_batch * k
            signal = a_batch * torch.exp(1j * (phase + sparams.mimo_system.starting_phase[T_idx, f]))
            measurement[T_idx, R_idx, f] = measurement[T_idx, R_idx, f] + torch.sum(signal)
        
        start = end
        end = n_paths if (n_paths - start < max_size) else start + max_size

    return measurement.real, measurement.imag

def intersect_scene(rays, scene):
    si = scene.ray_intersect(rays)
    if config['plot_sim']:
        if dr.count(si.is_valid()) > 0:
            connected_rays = dr.gather(mi.Ray3f, rays, dr.compress(si.is_valid()))
            viz.plot_lines(connected_rays.o, dr.gather(mi.Vector3f, si.p, dr.compress(si.is_valid())))

        if dr.count(~si.is_valid()) > 0:
            missed_rays = dr.gather(mi.Ray3f, rays, dr.compress(~si.is_valid()))
            viz.plot_rays(missed_rays.o, missed_rays.d)
        
        viz.show_plot()
    n_intersections = dr.count(si.is_valid())
    if config['print_stats']:
        print("Ray intersections: %3d" % n_intersections)
    return si, n_intersections

def get_bounce(si, tp, scene):
    indices = dr.compress(si.is_valid())

    wo, tp_pdf = smp.sample_dir(si.p, scene, mode=config['sampling_strat'])
    tp_pdf = dr.gather(mi.Color1f, tp_pdf, indices)
    new_tp = dr.gather(mi.Float, tp, indices) * tp_pdf

    wo_local = si.to_local(wo)
    wo_local = dr.gather(mi.Vector3f, wo_local, indices)
    if config['material'] == 'rough':
        ctx = mi.BSDFContext()
        bsdf = si.bsdf()
        bsdf_val = bsdf.eval(ctx, si, wo_local)
        new_tp = new_tp * dr.gather(mi.Float, bsdf_val.x, indices)
    elif config['material'] == 'diffuse':
        new_tp = new_tp * dr.dot(dr.gather(mi.Vector3f, si.wi, indices), dr.gather(mi.Vector3f, si.n, indices))
    elif config['material'] == 'phong':
        r = ms.reflect(dr.gather(mi.Vector3f, si.wi, indices), dr.gather(mi.Normal3f, si.n, indices))
        new_tp = new_tp * ((config['phong_kd'] * dr.dot(dr.gather(mi.Vector3f, si.wi, indices), dr.gather(mi.Normal3f, si.n, indices))) + config['phong_ks'] * dr.select((dr.dot(r, wo_local) > 0.0), (dr.power(dr.dot(r, wo_local), config['phong_ns'])), 0.0))

    rays = si.spawn_ray(wo)
    new_si, n_intersections = intersect_scene(rays, scene)

    if n_intersections <= 5:
        return None, tp
    return new_si, new_tp

# Run Simulation
def simulate_measurements_multi(ctx_args, depth = 1):
    scene = ctx_args['scene']

    smp.init_RNG(config['N_Paths'])
    real = dr.zeros(mi.TensorXf, shape=[sparams.mimo_system.get_param('N_Tx'), sparams.mimo_system.get_param('N_Rx'), sparams.mimo_system.get_param('N_Fq')])
    imag = dr.zeros(mi.TensorXf, shape=[sparams.mimo_system.get_param('N_Tx'), sparams.mimo_system.get_param('N_Rx'), sparams.mimo_system.get_param('N_Fq')])

    if config['render_grad']:
        dr.enable_grad(real)
        dr.enable_grad(imag)

    for T_i in range(sparams.mimo_system.get_param('N_Tx')):
        Ti = mi.Vector3f(dr.slice(sparams.mimo_system.get_param('Tx'), T_i))

        origins = mi.Vector3f(np.repeat(Ti.x, config['N_Paths']), np.repeat(Ti.y, config['N_Paths']), np.repeat(Ti.z, config['N_Paths']))

        ray_d, pdf = smp.sample_dir(Ti, scene, mode=config['sampling_strat'])
        rays = mi.Ray3f(o=origins, d=ray_d)
        si, n_intersections = intersect_scene(rays, scene)

        if n_intersections == 0:
            continue

        si_hist = []
        si_hist.append(si)

        wi = dr.gather(mi.Vector3f, si.wi, mi.UInt(dr.arange(mi.UInt, 0, config['N_Paths'])))
        norms = dr.gather(mi.Vector3f, si.to_local(si.n), mi.UInt(dr.arange(mi.UInt, 0, config['N_Paths'])))
 
        tp = dr.ones(mi.Float, config['N_Paths']) * (1/pdf) * mi.Float(sparams.mimo_system.get_param('antenna').gaussian_pattern_2d(dr.normalize(-ray_d)))
        if not config['relax_physics']:
            tp *= 1/(4*np.pi*si.t**2) * dr.select(dr.dot(wi, norms)> 0.0, dr.dot(wi, norms), 0.0)
        tp_hist = []
        tp_hist.append(tp)

        curr_depth = depth

        for i in range(depth-1):
            new_si, new_tp = get_bounce(si, tp, scene)
            tp_hist.append(new_tp)
            if new_si is None:
                curr_depth = i + 1
                break
            si_hist.append(new_si)
            si = new_si
            tp = new_tp

        if config['plot_sim']:
            viz.reset_plot()
            config['plot_sim'] = False

        for R_i in range(sparams.mimo_system.get_param('N_Rx')):
            Ri = mi.Vector3f(dr.slice(sparams.mimo_system.get_param('Rx'), R_i))

            di = dr.zeros(mi.Float, config['N_Paths'])
            ampi = dr.ones(mi.Float, config['N_Paths'])

            for i in range(curr_depth):
                sii = si_hist[i]
                tpi = tp_hist[i]

                indices = dr.compress(sii.is_valid())

                dir, tpir = connect_to_Rx(sii, scene, Ri, indices)

                di = dr.gather(mi.Float, di, indices) + dr.gather(mi.Float, sii.t, indices) + dir
                ampi = dr.gather(mi.Color1f, tpi, indices) * tpir
                
                dist = mi.TensorXf(dr.ravel(di))
                amps = mi.TensorXf(dr.ravel(ampi))
                Fq_t = mi.TensorXf(dr.ravel(sparams.mimo_system.get_param("Fq")))

                real, imag = get_measurement(real, imag, dist, amps, T_i, R_i, Fq_t)
                dr.flush_malloc_cache()

    return real, imag

def connect_to_Rx_dir(si, Rx, indices):
    temp = Rx - dr.gather(mi.Point3f, si.p, mi.UInt(dr.arange(mi.UInt, 0, sparams.N_Paths)))
    Rx_dist = dr.sqrt(dr.dot(temp,temp))
    Rx_dist = dr.gather(mi.Float, Rx_dist, indices)

    wo = dr.normalize((temp))

    wo = si.to_local(wo)
    wo = dr.gather(mi.Vector3f, wo, indices)

    ctx = mi.BSDFContext()
    bsdf = si.bsdf()
    
    bsdf_val = bsdf.eval(ctx, si, wo)

    throughput = dr.gather(mi.Float, bsdf_val.x, indices)

    return Rx_dist, throughput

# Return Distances and Amplitudes for each path
def calc_paths(si, Rx, N_Paths, depth=1):
    i = mi.UInt(0)
    dist = mi.Float(0)
    amp = mi.Float(1)

    loop = mi.Loop(name="", state=lambda: (i, dist, amp))

    while loop(si.is_valid() & (i < depth)):
        
        indices = mi.UInt(dr.arange(mi.UInt, 0, N_Paths))

        travel_dist = si.t

        dist += dr.gather(mi.Float, travel_dist, indices, si.is_valid())
        i += 1

    Rx_dist, Rx_amp = connect_to_Rx_dir(si, Rx, indices)
    dist += Rx_dist
    amp *= Rx_amp

    return dist, amp

# Run Simulation
def simulate_measurements_ms(ctx_args, N_Paths):
    scene = ctx_args['scene']
    n_paths = 50

    smp.init_RNG(n_paths)
    real = dr.zeros(mi.TensorXf, shape=[sparams.mimo_system.get_param('N_Tx'), sparams.mimo_system.get_param('N_Rx'), sparams.mimo_system.get_param('N_Fq')])
    imag = dr.zeros(mi.TensorXf, shape=[sparams.mimo_system.get_param('N_Tx'), sparams.mimo_system.get_param('N_Rx'), sparams.mimo_system.get_param('N_Fq')])

    if config['render_grad']:
        dr.enable_grad(real)
        dr.enable_grad(imag)

    for T_i in range(sparams.mimo_system.get_param('N_Tx')):
        Ti = mi.Vector3f(dr.slice(sparams.mimo_system.get_param('Tx'), T_i))

        for R_i in range(sparams.mimo_system.get_param('N_Rx')):
            Ri = mi.Vector3f(dr.slice(sparams.mimo_system.get_param('Rx'), R_i))

            ms_success = False
            ms_success, v_final = ms.sample_manifold(Ti, Ri, scene, 0.02, 1e-5, 250)
            if ms_success:
                
                pts = dr.slice(v_final.p, 0)
                n_paths = 1
            else:
                print("MS FAILED")
                pts = dr.slice(v_final.p, 0)
                n_paths = 1

            ray_d = dr.normalize(pts - Ti)
            rays = mi.Ray3f(o=Ti, d=ray_d)
            si, n_intersections = intersect_scene(rays, scene)
            if n_intersections == 0:
                continue

            result, amp = calc_paths(si, Ri, n_paths)

            dist = mi.TensorXf(dr.ravel(result), shape=[n_paths])
            amps = mi.TensorXf(dr.ones(mi.Float, n_paths), shape=[n_paths])
            Fq_t = mi.TensorXf(dr.ravel(sparams.mimo_system.get_param('Fq')), shape=[sparams.mimo_system.get_param('N_Fq')])

            real, imag = get_measurement(real, imag, dist, amps, T_i, R_i, Fq_t)

    return real, imag

def simulate_measurements(ctx_args, use_ms):
    if use_ms:
        return simulate_measurements_ms(ctx_args, config['N_Paths'])
    else:
        return simulate_measurements_multi(ctx_args, depth=config['sim_depth'])

if config['render_grad']:
    def run_simulation(ctx_args, use_ms=False):
        return simulate_measurements(ctx_args, use_ms)
else:
    @dr.wrap_ad(source='torch', target='drjit')
    def run_simulation(ctx_args, use_ms=False):
        return simulate_measurements(ctx_args, use_ms)
    
def apply_transformation(params, theta, key, ref):
    if isinstance(theta, torch.Tensor):
        theta = theta.tolist()

    trafo = mi.Transform4f.translate([theta, 0.0, 0.0])
    params[key] = trafo @ ref
    params.update()

def apply_transformation_s(params, theta, key, ref):
    if isinstance(theta, torch.Tensor):
        theta = theta.tolist()

    trafo = mi.Transform4f.scale([theta, theta, theta])
    params[key] = trafo @ ref
    params.update()

