import numpy as np
import pyvista as pv

import sparams
import sampler as smp

base_actors = None
curr_actors = None

def initalize_plot(scene):
    n_shapes = len(scene.get_scene().shapes())
    plotter = pv.Plotter()

    viz_samples = 10000
    print(f"Using {viz_samples} samples for visualization")
    smp.init_RNG(viz_samples)
    
    for i in range(n_shapes):
        p = smp.sample_pos_target_shape(scene.get_scene(), i)
        pts = p.numpy()
        z_diff = np.max(pts[:, 2]) - np.min(pts[:, 2])
        cloud = pv.PolyData(pts)
        #cloud.plot(point_size=1)
        if scene.get_name() == "Sphere":
            surf = cloud.delaunay_3d()
        elif scene.get_name() == "Plane":
            surf = cloud.delaunay_2d()
        else:
            if z_diff < 0.01:
                surf = cloud.delaunay_2d()
            else:
                surf = cloud.delaunay_3d()
        plotter.add_mesh(surf, show_edges=False, name=f"shape_{i}")

    plotter.add_points(sparams.mimo_system.get_param('Rx').numpy(), color='darkorange', point_size=5, name="Rx")
    plotter.add_points(sparams.mimo_system.get_param('Tx').numpy(), color='green', point_size=5, name="Tx")

    global base_actors
    base_actors = plotter.actors
    global curr_actors
    curr_actors = base_actors

def plot_cloud(scene):
    n_shapes = len(scene.get_scene().shapes())
    viz_samples = 1000000
    print(f"Using {viz_samples} samples for visualization")
    smp.init_RNG(viz_samples)
    
    for i in range(n_shapes):
        p = smp.sample_pos_target_shape(scene.get_scene(), i)
        pts = p.numpy()
        cloud = pv.PolyData(pts)
        cloud.plot(point_size=1)

def reset_plot():
    global base_actors
    global curr_actors
    curr_actors = base_actors

def plot_rays(origins, ray_d):
    plotter = pv.Plotter()

    global curr_actors
    for actor in curr_actors.values():
        plotter.add_actor(actor)

    rays = [pv.Line(o, o+v*0.1) for o, v in zip(origins.numpy(), ray_d.numpy())]
    for ray in rays:
        plotter.add_mesh(ray, color='blue', line_width=1)
    curr_actors = plotter.actors

def plot_lines(origins, end_points):
    plotter = pv.Plotter()

    global curr_actors
    for actor in curr_actors.values():
        plotter.add_actor(actor)

    rays = [pv.Line(o, e) for o, e in zip(origins.numpy(), end_points.numpy())]
    for ray in rays:
        plotter.add_mesh(ray, color='green', line_width=1)
    curr_actors = plotter.actors

def show_plot():
    plotter = pv.Plotter()

    global curr_actors
    for actor in curr_actors.values():
        plotter.add_actor(actor)

    plotter.show()
