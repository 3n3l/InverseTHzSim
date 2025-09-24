import mitsuba as mi
import drjit as dr

from config import config

# Setup Mitsuba RNG
sampler_ = mi.load_dict({
    'type': 'independent'
})

def init_RNG(size=config['N_Paths']):
    global sampler_
    sampler_.seed(0xBADCAFE, size)

def seed_RNG(seed, size):
    global sampler_
    sampler_.seed(seed, size)

# Sample towards a shape
def sample_pos_target_shape(scene, idx=0):
    sample = scene.shapes()[idx].sample_position(0, sampler_.next_2d())
    return sample.p

# Sample towards a shape
def sample_target_shape(eye, scene, idx=0):
    si = mi.SurfaceInteraction3f()
    si.p = eye
    sample = scene.shapes()[idx].sample_direction(si, sampler_.next_2d())
    return sample.d, sample.pdf

# Sample towards all shapes
def sample_all_shapes(eye, scene):
    tot_area = 0.0
    for shape in scene.shapes():
        tot_area += shape.surface_area()

    samples = sampler_.next_2d()

    shape_idx = len(scene.shapes())
    used_samples = 0
    d = dr.zeros(mi.Vector3f, config['N_Paths'])
    pdf = dr.zeros(mi.Float, config['N_Paths'])
    for shape in scene.shapes():
        shape_idx -= 1
        if shape_idx == 0:
            n_samples = config['N_Paths'] - used_samples
        else:
            n_samples = int(((shape.surface_area() / tot_area) * config['N_Paths']).numpy())
        shape_samples = dr.gather(mi.Vector2f, samples, dr.arange(mi.UInt, used_samples, used_samples + n_samples))
        si = mi.SurfaceInteraction3f()
        si.p = eye
        sample = shape.sample_direction(si, shape_samples)
        dr.scatter(d, sample.d, dr.arange(mi.UInt, used_samples, used_samples + n_samples))
        dr.scatter(pdf, sample.pdf, dr.arange(mi.UInt, used_samples, used_samples + n_samples))
        used_samples += n_samples
    
    return d, pdf

# Sample from a Hemisphere
def sample_hemi():
    return mi.warp.square_to_uniform_hemisphere(sampler_.next_2d()), 1.0

# Sample from a Cone
def sample_cone():
    return mi.warp.square_to_uniform_cone(sampler_.next_2d(), 0.75), 1.0

# Sample from a Disk
def sample_disk():
    return mi.warp.square_to_uniform_disk(sampler_.next_2d()), 1.0

def sample_bsdf(ctx, bsdf, si):
    sp, tp = bsdf.sample(ctx, si, sampler_.next_1d(), sampler_.next_2d())
    return sp.wo, tp

md = {
     "shape": sample_target_shape,
     "all_shapes": sample_all_shapes,
     "hemi": sample_hemi,
     "cone": sample_cone,
     "disk": sample_disk
}

# Generate Ray Directions
def sample_dir_old(n_rays, eye, scene, mode="shape"):
        fixed = (mode == "shape")

        dir_x = []
        dir_y = []
        dir_z = []
        for j in range(n_rays):
            if fixed:
                pos = md[mode](scene)
                dir = dr.normalize(pos-eye)
            else:
                dir = md[mode]()
               
            dir_x.append(dir.x)
            dir_y.append(dir.y)
            dir_z.append(dir.z)

        dir_x = [val for sublist in dir_x for val in sublist]
        dir_y = [val for sublist in dir_y for val in sublist]
        dir_z = [val for sublist in dir_z for val in sublist]

        return mi.Vector3f(dir_x, dir_y, dir_z)

def sample_dir(eye, scene, mode="shape"):
        fixed = (mode == "shape" or mode == "all_shapes")

        if fixed:
            dir, pdf = md[mode](eye, scene)
        else:
            dir, pdf = md[mode]()

        return dir, pdf

