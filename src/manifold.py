import mitsuba as mi
import drjit as dr
import math
from config import config
from sparams import mimo_system, measured_sphere_rad
import sparams
import sampler as smp


# helper class for manifold sampling
class MSVertex:
    def __init__(self, si: mi.SurfaceInteraction3f, mask_indices):
        """
        Params:
            si: mi.SurfaceInteraction3f or MSVertex
            mask_indices: list[int]
        """
        # self.p = dr.gather(mi.Vector3f, si.p, mask_indices)
        # TODO: why is this now a point? or has this always been a point in 3.6?
        self.p = dr.gather(mi.Point3f, si.p, mask_indices)
        self.dp_du = dr.gather(mi.Vector3f, si.dp_du, mask_indices)
        self.dp_dv = dr.gather(mi.Vector3f, si.dp_dv, mask_indices)
        self.shape = dr.gather(mi.ShapePtr, si.shape, mask_indices)
        # self.n = dr.gather(mi.Vector3f, si.n, mask_indices)
        # TODO: why is this now a normal? or has this always been a normal in 3.6?
        self.n = dr.gather(mi.Normal3f, si.n, mask_indices)
        self.dn_du = dr.gather(mi.Vector3f, si.dn_du, mask_indices)
        self.dn_dv = dr.gather(mi.Vector3f, si.dn_dv, mask_indices)

    def __str__(self):
        return f"""Manifold Sampling Vertex: {id(self)}\n, self.p = {self.p},\n self.dp_du = {self.dp_du},\n self.dp_dv = {self.dp_dv},\n self.shape = {self.shape},\n self.n = {self.n},\n self.dn_du = {self.dn_du},\n self.dn_dv = {self.dn_dv}"""

# Manifold Sampling Helpers
def reflect(w, n):
    return 2.0 * dr.dot(w, n) * n - w


def d_reflect(w, dw_du, dw_dv, n, dn_du, dn_dv):
    dot_w_n = dr.dot(w, n)
    dot_dwdu_n = dr.dot(dw_du, n)
    dot_dwdv_n = dr.dot(dw_dv, n)
    dot_w_dndu = dr.dot(w, dn_du)
    dot_w_dndv = dr.dot(w, dn_dv)
    dwr_du = 2.0 * ((dot_dwdu_n + dot_w_dndu) * n + dot_w_n * dn_du) - dw_du
    dwr_dv = 2.0 * ((dot_dwdv_n + dot_w_dndv) * n + dot_w_n * dn_dv) - dw_dv
    return dwr_du, dwr_dv


def sphcoords(w):
    theta = dr.safe_acos(w.z)
    phi = dr.atan2(w.y, w.x)

    # NOTE: original code compares whole phi against zero
    if phi[0] < 0.0:
    # NOTE: this does not make a difference
    # if phi[0] < 0.0 and phi[1] < 0.0:
        phi += 2.0 * math.pi

    return theta, phi


def d_sphcoords(w, dw_du, dw_dv):
    d_acos = -dr.rcp(dr.safe_sqrt(1.0 - w.z * w.z))
    d_theta = d_acos * mi.Vector2f(dw_du.z, dw_dv.z)

    yx = w.y / w.x
    d_atan = dr.rcp(1 + yx * yx)
    d_phi = d_atan * mi.Vector2f(w.x * dw_du.y - w.y * dw_du.x, w.x * dw_dv.y - w.y * dw_dv.x) * dr.rcp(w.x * w.x)

    # if w.x == 0.0:
    # if dr.any(w.x == 0.0):
    # print("w.x:", w.x)
    if dr.all(w.x == 0.0):
        d_phi.x, d_phi.y = 0.0, 0.0

    return d_theta.x, d_phi.x, d_theta.y, d_phi.y


def step_anglediff(v0, v, v2):
    # wi, wo and derivatives
    wi = v0 - v.p
    ili = dr.norm(wi)
    if ili[0] < 1e-3:
        # print("ili[0] < 1e-3:")
        return False, mi.Vector2f(math.inf, 0.0), mi.Vector2f(math.inf, 0.0)
    ili = dr.rcp(ili)
    wi *= ili

    dwi_du = -ili * (v.dp_du - wi * dr.dot(wi, v.dp_du))
    dwi_dv = -ili * (v.dp_dv - wi * dr.dot(wi, v.dp_dv))

    wo = v2 - v.p
    ilo = dr.norm(wo)
    if ilo[0] < 1e-3:
        # print("ilo[0] < 1e-3:")
        return False, mi.Vector2f(math.inf, 0.0), mi.Vector2f(math.inf, 0.0)
    ilo = dr.rcp(ilo)
    wo *= ilo

    dwo_du = -ilo * (v.dp_du - wo * dr.dot(wo, v.dp_du))
    dwo_dv = -ilo * (v.dp_dv - wo * dr.dot(wo, v.dp_dv))

    # Constraints
    C = mi.Vector2f(0.0)
    dC_dX = mi.Matrix2f(0.0)

    wio = reflect(wi, v.n)
    dwio_du, dwio_dv = d_reflect(wi, dwi_du, dwi_dv, v.n, v.dn_du, v.dn_dv)
    to, po = sphcoords(wo)
    tio, pio = sphcoords(wio)

    dt = to - tio
    dp = po - pio

    if dp[0] < -math.pi:
        dp += 2.0 * math.pi
    elif dp[0] > math.pi:
        dp -= 2.0 * math.pi

    C = mi.Vector2f(dt, dp)

    dto_du, dpo_du, dto_dv, dpo_dv = d_sphcoords(wo, dwo_du, dwo_dv)
    dtio_du, dpio_du, dtio_dv, dpio_dv = d_sphcoords(wio, dwio_du, dwio_dv)

    dC_dX[0, 0] = dto_du - dtio_du
    dC_dX[1, 0] = dpo_du - dpio_du
    dC_dX[0, 1] = dto_dv - dtio_dv
    dC_dX[1, 1] = dpo_dv - dpio_dv

    determinant = dr.det(dC_dX)
    if dr.abs(determinant)[0] < 1e-6:
        # print("dr.abs(determinant)[0] < 1e-6:")
        return False, mi.Vector2f(math.inf, 0.0), mi.Vector2f(math.inf, 0.0)

    dX_dC = dr.inverse(dC_dX)

    dX = dX_dC @ C

    # print(f"in step_anglediff(...):\nC = {C},\ndC_dX = {dC_dX},\ndX = {dX}\n\n")

    # TODO: C could be wrong?!
    return True, C, dX


# Newton Solver for Manifold Sampling
def newton_solver(v0, v_init, v2, scene, step_scale, threshold, max_iters):
    success = False
    beta = 1.0 # Initial beta of 1.0 might be too large???
    # beta = 0.5 # Initial beta of 1.0 might be too large???

    # print(f"step_scale = {step_scale}, threshold = {threshold}, max_iters = {max_iters}")
    si_current = mi.SurfaceInteraction3f()
    v = v_init

    for _ in range(max_iters):
        ### FIXME: v ERROR IN HERE?
        # print(f"calling step_anglediff(...) with v.p = {v.p}, beta = {beta}")
        step_success, C, dX = step_anglediff(v0, v, v2)
        # print(f"C (F) = {C}, dX (delta_q) = {dX}")
        # print()
        ### FIXME: ^ ERROR IN HERE?
        if not step_success:
            print("step_anglediff not successful!")
            break

        norm_C = dr.norm(C)
        # print(f"norm(C) = {norm_C}")
        # FIXME: norm_C is ~ 2.5 and never going down
        # FIXME: is C just wrong?!?
        # print(f"norm_C, threshold = {norm_C}, {threshold}")
        if dr.any(norm_C < threshold):
            # print(f"norm_C, threshold = ", norm_C, threshold)
            # print()
            success = True
            solution_mask = norm_C < threshold
            solution_idx = dr.compress(solution_mask)
            v = MSVertex(v, solution_idx)
            break

        ### FIXME: v ERROR IN HERE?
        # Make a PROPosal
        p_prop = v.p - step_scale * beta * (v.dp_du * dX.x + v.dp_dv * dX.y)
        # print(f"sphere_rad = {sphere_rad}, ref_radius = {config["ref_radius"]}")

        # # NOTE: original code from Mohamed, project the PROPosed ray against the center of the bbox
        # temp = mi.Vector3f(scene.shapes()[0].bbox().center()) - p_prop
        # # temp = p_prop - mi.Vector3f(scene.shapes()[0].bbox().center()) # NOTE: maybe wrong sign?
        # dist_c = dr.norm(temp)
        # sphere_rad = measured_sphere_rad[config["ref_radius"]]
        # d_prop = dr.select((dist_c > sphere_rad + 1e-5), dr.normalize(temp), dr.normalize(p_prop - v0))
        # ray_prop = dr.select((dist_c > sphere_rad + 1e-5), mi.Ray3f(p_prop, d_prop), mi.Ray3f(v0, d_prop))

        # # NOTE: based on SMS paper implementation, projects PROPosed ray against surface
        # # 1. 
        d_prop = dr.normalize(p_prop - v0)
        ray_prop = mi.Ray3f(v0, d_prop)
        # # 2.
        # d_prop = dr.normalize(mi.Vector3f(scene.shapes()[0].bbox().center()) - p_prop)
        # ray_prop = mi.Ray3f(p_prop, d_prop)

        si_current = scene.ray_intersect(ray_prop)
        mask = si_current.is_valid() & (si_current.shape == v.shape)

        # FIXME: this is missing the scene all the time?!
        # NOTE: yes, this is missing the scene all the time, which means that the MSVertex will never be updated,
        #       the iteration will just end with the first MSVertex and a beta approaching zero without finding
        #       the correct ray.

        # TODO: what could cause this???
        # FIXME: is this missing the scene or hitting a different shape??? or both???
        # print("HIT THE OBJECT" if dr.all(si_current.is_valid()) else "MISSED THE OBJECT", "HIT THE SAME SHAPE" if dr.all(si_current.shape == v.shape) else "MISSED THE SAME SHAPE", sep="\n", end="\n\n")
        # print("HIT THE OBJECT" if si_current.is_valid() else "MISSED THE OBJECT", sep="\n", end="\n\n")
        # print("HIT THE SAME SHAPE" if si_current.shape == v.shape else "MISSED THE SAME SHAPE", sep="\n", end="\n\n")

        if dr.none(mask):  # Missed scene completely or hit different shape (take smaller step)
            # print("missed scene, beta = ", beta)
            # FIXME: usually both of them are False, not just the conjunction:
            # print(f"si_current.shape == v.shape: {si_current.shape == v.shape}")
            # print(f"si_current.is_valid(): {si_current.is_valid()}")
            # print()

            # NOTE: what about changing the seed here? if the seed is bad a smaller step won't do anything
            # NOTE: original code does not reseed here
            beta *= 0.5
            continue

        beta = min(1.0, 2.0 * beta)
        # print(f"beta = {beta}")

        # NOTE: this does update the point, however all entries will be the same for all points???
        # print(f"v.p, before = {v.p}")
        v = MSVertex(si_current, dr.compress(mask))
        # v = MSVertex(si_current, mask)
        # print(f"v.p, after = {v.p}\n")

    # print()

    # if not success:
    #     return False, v

    # return True, v

    return success, v


# # helper function to verify perfect reflections
# def check_reflectance(v0, v1, v2, n):
#     wi = dr.normalize(v0 - v1)
#     wo = dr.normalize(v2 - v1)
#     wr = dr.normalize(reflect(wi, n))
#     return dr.all((dr.abs(wo - wr)) < 1e-5)


# def check_reflectance_dot(v0, v1, v2, n):
#     wi = dr.normalize(v0 - v1)
#     wo = dr.normalize(v2 - v1)
#     wr = dr.normalize(reflect(wi, n))
#     return dr.all((dr.abs(dr.dot(wo, wr) - 1.0)) < 1e-5)


def sample_manifold(Ti, Ri, scene, step_scale=0.04, threshold=1e-5, max_iters=50):
    # print(f"now sampling manifold, max_iters = {max_iters}")
    direction, _ = smp.sample_dir(Ti, scene, mode=config["sampling_strat"])
    # direction, _ = smp.sample_dir(Ti, scene, mode="shape")
    wi = mi.Ray3f(o=Ti, d=direction)
    # print(wi)
    si = scene.ray_intersect(wi)
    v_init = MSVertex(si, dr.compress(si.is_valid()))  # all valid intersections

    # print(f"si.p = {si.p}")
    print(f"v_init.p = {v_init.p}")
    print()

    # print("si:", si)
    # print("v_init:", v_init)
    success, v_final = newton_solver(Ti, v_init, Ri, scene, step_scale, threshold, max_iters)
    # print()
    return success, v_final
