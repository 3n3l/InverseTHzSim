import scenes
import measurement
from config import config
import sparams


def change_mat_param(value):
    if config["op_dim"] == 1:
        if config["material"] == "phong":
            if config["phong_ns"] == None:
                config["phong_ns"] = value
            else:
                config["phong_kd"] = value
                config["phong_ks"] = 1.0 - value
        elif config["material"] == "rough":
            config["roughness"] = value
    else:
        if config["material"] == "phong":
            kd = value[0].item()
            ns = value[1].item() if value[1].item() >= 0.01 else 0.01
            ns = value[1].item() * 100
            kd = kd if kd >= 0.0 else 9999  # penalize negative values
            config["phong_kd"] = kd
            config["phong_ks"] = 1 - kd
            config["phong_ns"] = ns


def setup_experiment(settings):
    # if check_errors(settings):
    #    return

    params = {}
    if settings["object"].lower() == "sphere":
        if settings["op_param"].lower() == "pos":
            params["scene"] = exp_sphere_pos(settings["ref_position"])
            if settings["op_dim"] == 1:
                params["update_fn"] = params["scene"].translate_x
            elif settings["op_dim"] == 2:
                params["update_fn"] = params["scene"].translate_xy
            elif settings["op_dim"] == 3:
                params["update_fn"] = params["scene"].translate_xyz
        elif settings["op_param"].lower() == "radius":
            params["scene"] = exp_sphere_radius(settings["ref_position"])
            if settings["op_dim"] == 1:
                params["update_fn"] = params["scene"].scale_rad
            elif settings["op_dim"] == 2:
                params["update_fn"] = params["scene"].scale_rad_z
        elif settings["op_param"].lower() == "material":
            params["scene"] = exp_sphere_material(settings["ref_position"])
            params["update_fn"] = change_mat_param
        elif settings["op_param"].lower() == "pos_z":
            params["scene"] = exp_sphere_height(settings["ref_position"])
            params["update_fn"] = params["scene"].translate_z
    elif settings["object"].lower() == "torus":
        if settings["op_param"].lower() == "pos":
            params["scene"] = exp_torus_pos(settings["ref_position"])
            params["update_fn"] = params["scene"].translate_x
        elif settings["op_param"].lower() == "material":
            params["scene"] = exp_torus_material(settings["ref_position"])
            params["update_fn"] = change_mat_param
    elif settings["object"].lower() == "plane" or settings["object"].lower() == "wood":
        if settings["op_param"].lower() == "pos":
            params["scene"] = exp_plane_height()
            params["update_fn"] = params["scene"].translate_z
        elif settings["op_param"].lower() == "material":
            params["scene"] = exp_plane_material()
            params["update_fn"] = change_mat_param
    elif settings["object"].lower() == "sphere_plane":
        if settings["op_param"].lower() == "pos":
            params["scene"] = exp_sphere_plane()
            params["update_fn"] = params["scene"].translate_sphere_x
        elif settings["op_param"].lower() == "material":
            params["scene"] = exp_sphere_plane_mat()
            params["update_fn"] = change_mat_param
    elif settings["object"].lower() == "two_spheres":
        params["scene"] = exp_two_spheres()
        params["update_fn"] = params["scene"].translate_x
    else:
        print("Invalid object type, please check the configuration file.")
        return

    params["ref_msr"] = measurement.Real(pos_path(settings["object"], settings["ref_position"], True))
    return params


def check_errors(settings):
    if sparams.measured_sphere_pos.get(settings["ref_position"]) is None:
        print("Invalid Reference Position, please check the configuration file.")
        return True


def exp_sphere_pos(pos=4):
    scene = scenes.OneSphere()
    if config["op_dim"] == 1:
        scene.set_position(0.0, sparams.measured_sphere_pos[pos][1], sparams.measured_sphere_pos[pos][2])
    elif config["op_dim"] == 2:
        scene.set_position(0.0, 0.0, sparams.measured_sphere_pos[pos][2])
    elif config["op_dim"] == 3:
        scene.set_position(0.0, 0.0, 0.0)
    scene.set_radius(0.02)  # use the ref sphere radius
    return scene


def exp_sphere_height(pos=4):
    scene = scenes.OneSphere()
    scene.set_position(sparams.measured_sphere_pos[4][0], sparams.measured_sphere_pos[4][1], 0.0)
    scene.set_radius(0.02)  # use the ref sphere radius
    return scene


def exp_sphere_radius(pos=4):
    scene = scenes.OneSphere()
    if config["op_dim"] == 1:
        scene.set_position(
            sparams.measured_sphere_pos[pos][0],
            sparams.measured_sphere_pos[pos][1],
            sparams.measured_sphere_pos[pos][2],
        )
    elif config["op_dim"] == 2:
        scene.set_position(sparams.measured_sphere_pos[pos][0], sparams.measured_sphere_pos[pos][1], 0.0)
    return scene


def exp_torus_pos(pos=4):
    scene = scenes.OneTorus()
    scene.set_position(0.0, sparams.measured_torus_pos[pos][1], sparams.measured_torus_pos[pos][2] - 0.01)
    return scene


def exp_plane_height():
    scene = scenes.OnePlane(distance=0.0)
    return scene


def exp_plane_material(pos=1):
    scene = scenes.OnePlane(distance=0.0)
    if config["object"].lower() == "wood":
        scene.set_height(sparams.measured_wood_pos[pos])
    else:
        scene.set_height(sparams.measured_plane_pos[pos])
    return scene


def exp_sphere_plane(pos=3):
    scene = scenes.SpherePlane()
    scene.set_sphere_position(0.0, sparams.measured_sphere_plane_pos[pos][1], sparams.measured_sphere_plane_pos[pos][2])
    scene.set_sphere_radius(0.02)
    return scene


def exp_sphere_plane_mat(pos=3):
    scene = scenes.SpherePlane()
    scene.set_sphere_position(
        sparams.measured_sphere_plane_pos[pos][0],
        sparams.measured_sphere_plane_pos[pos][1],
        sparams.measured_sphere_plane_pos[pos][2],
    )
    scene.set_sphere_radius(0.02)
    return scene


def exp_sphere_material(pos=4):
    scene = scenes.OneSphere()
    scene.set_position(
        sparams.measured_sphere_pos[pos][0], sparams.measured_sphere_pos[pos][1], sparams.measured_sphere_pos[pos][2]
    )
    scene.set_radius(0.02)
    return scene


def exp_torus_material(pos=4):
    scene = scenes.OneTorus()
    scene.set_position(
        sparams.measured_torus_pos[pos][0],
        sparams.measured_torus_pos[pos][1],
        sparams.measured_torus_pos[pos][2] - 0.01,
    )
    return scene


def exp_two_spheres(pos=4):
    scene = scenes.TwoSpheres()
    scene.set_position_1(0.0, sparams.measured_sphere_pos[pos][1], sparams.measured_sphere_pos[pos][2])
    scene.set_position_2(0.0, sparams.measured_sphere_pos[pos][1], sparams.measured_sphere_pos[pos][2])
    scene.set_radius_1(0.02)
    scene.set_radius_2(0.02)
    return scene


def pos_path(obj, pos, cal=True):
    path = "calibrated64/" if cal else "not_calibrated64/"
    if obj.lower() == "sphere":
        if config["op_param"] == "pos_z":
            path += f"sphere_40mm_pos_4_foam_64f_{int(sparams.measured_sphere_height[pos] * 1000)}_mm"
        elif config["ref_radius"] == 1:
            path += f"sphere_pos_{pos}_foam_64f_330_mm"
        else:
            path += f"sphere_{int(sparams.measured_sphere_rad[config['ref_radius']]*2000)}mm_pos_{pos}_foam_64f_330_mm"
    elif obj.lower() == "torus":
        path += f"torus_b_side_pos_{pos}_ref_4_foam_64f_330_mm"
    elif obj.lower() == "plane":
        path += f"alu_plate_64f_{int(sparams.measured_plane_pos[pos] * 1000)}_mm"
    elif obj.lower() == "sphere_plane":
        path += f"sphere_pos_{pos}_alu_plate_64f_345_mm"
    elif obj.lower() == "two_spheres":
        path += f"sphere_pos_2_and_{pos}_foam_64f_330_mm"
    elif obj.lower() == "wood":
        path += f"wood_board_16_6mm_64f_{int(sparams.measured_wood_pos[pos] * 1000)}_4_mm"
    print(f"Loaded Reference: {path}")
    return path


def cal_path():
    path = "../data/calibration/"
    if config["cal_strat"] == "simulated":
        if config["material"] == "phong":
            if config["relax_physics"]:
                path += "plane_1.0m_sd0_TxFq_0_tilt_phong_0.19_0.81_46.91"
            else:
                path += "plane_1m_sd0_TxFq_0_tilt_phong_0.0_1.0_13.83"
        elif config["material"] == "rough":
            path += "plane_10m_sd0_TxFq_0_tilt_rough_0.2"
        path += ".rdm"
    else:
        path = "calibrated64/Calibration_files/Short"
    return path


def save_path():
    path = f"../data/msr_{config['op_dim']}D/"
    if config["op_param"] == "pos":
        path += "x_axis/"
    elif config["op_param"] == "pos_z":
        path += "z_axis/"
    elif config["op_param"] == "radius":
        path += f"radius/"
    elif config["op_param"] == "material":
        path += f"material/"

    if config["cal_strat"] == "simulated" or config["cal_strat"] == "self":
        path += "rand_phase/"
    else:
        path += "phase_0/"

    path += f"{config['object']}/"

    if config["sim_depth"] > 1:
        path += "multi_bounce/"
    else:
        path += "single_bounce/"

    if config["relax_physics"]:
        path += "relaxed_physics/"

    if not config["op_param"] == "pos":
        path += f"pos_{config['ref_position']}/{config['material']}"
    else:
        path += f"{config['material']}"

    if not config["op_param"] == "material":
        if config["material"] == "phong":
            path += f"_{config['phong_kd']}_{config['phong_ks']}_{config['phong_ns']}/{config['N_Paths']}_samples/"
        elif config["material"] == "rough":
            path += f"_{config['roughness']}/"
    else:
        path += f"/"
        if config["material"] == "phong":
            if config["op_dim"] == 1:
                if config["phong_ns"] == None:
                    path += f"spec/kd_{config['phong_kd']}/"
                else:
                    path += f"weights/ns_{config['phong_ns']}/"

    return path


def get_gt():
    if config["op_param"] == "pos":
        if config["object"] == "sphere":
            return sparams.measured_sphere_pos[config["ref_position"]][0]
        elif config["object"] == "torus":
            return sparams.measured_torus_pos[config["ref_position"]][0]
        elif config["object"] == "plane":
            return sparams.measured_plane_pos[config["ref_position"]]
        elif config["object"] == "wood":
            return sparams.measured_wood_pos[config["ref_position"]]
        elif config["object"] == "sphere_plane":
            return sparams.measured_sphere_plane_pos[config["ref_position"]][0]
    elif config["op_param"] == "pos_z":
        if config["object"] == "sphere":
            return sparams.measured_sphere_height[config["ref_position"]]
    elif config["op_param"] == "radius":
        return sparams.measured_sphere_rad[config["ref_radius"]]
    else:
        raise ValueError("Invalid operation parameter for ground truth retrieval.")
