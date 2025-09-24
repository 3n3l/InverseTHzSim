import mitsuba as mi
import visualization as vis
import drjit as dr
import geom_utils as gu

class THZScene:
    def __init__(self):
        self.plot_init = False

    def get_scene(self):
        return self.scene
    
    def translate(self, theta):
        trafo = mi.Transform4f.translate([theta.x, theta.y, theta.z])

        self.params[self.key] = trafo @ self.ref_transform
        self.params.update()

    def scale(self, theta):
        trafo = mi.Transform4f.scale([theta, theta, theta])

        self.params[self.key] = self.ref_transform @ trafo
        self.params.update()

    def rotate(self, theta):
        trafo = mi.Transform4f.rotate([0.0, 1.0, 0.0], theta)

        self.params[self.key] = trafo @ self.ref_transform
        self.params.update()

    def translate_and_scale(self, t, s):
        trafo = mi.Transform4f.translate([t.x, t.y, t.z])
        ref = trafo @ self.ref_transform
        trafo = mi.Transform4f.scale([s, s, s])

        self.params[self.key] = ref @ trafo
        self.params.update()

    def update_ref(self):
        self.ref_transform = mi.Transform4f(self.params[self.key])

    def plot(self):
        if not self.plot_init:
            self.plot_init = True
            vis.initalize_plot(self)
        vis.show_plot()

    def plot_cloud(self):
        vis.plot_cloud(self)

    def get_name(self):
        return self.scene_name
    
    def print_mi_params(self):
        print(self.params)

    def print_opt_param(self):
        print(self.params[self.key])

class OneSphere(THZScene):
    def __init__(self, roughness = 0.2):
        super().__init__()

        self.scene_name = 'Sphere'

        self.scene = mi.load_dict({
            'type' : 'scene',
            'sphere1': {
                'type': 'sphere',
                'center': [0.0, 0.0, 0.0],
                'radius': 1.0,
                'bsdf': {
                    'type': 'roughconductor',
                    'alpha': roughness, # 0.001 - 0.01 (smooth), 0.1-0.3 (relatively rough), 0.3-0.7 (extremely rough)
                    'distribution': 'ggx'
                }
            }
        })

        self.params = mi.traverse(self.scene)
        self.key = 'sphere1.to_world'
        self.ref_transform = mi.Transform4f(self.params[self.key])

    def set_position(self, x, y, z):
        self.translate(mi.Vector3f(x, y, z))
        self.update_ref()

    def set_radius(self, rad):
        self.translate_and_scale(mi.Vector3f(0.0, 0.0, -rad), rad)
        self.update_ref()

    def translate_x(self, x):
        self.translate(mi.Vector3f(x, 0.0, 0.0))

    def translate_z(self, z):
        self.translate(mi.Vector3f(0.0, 0.0, z))

    def translate_xy(self, theta):
        self.translate(mi.Vector3f(theta[0].item(), theta[1].item(), 0.0))

    def translate_xyz(self, theta):
        self.translate(mi.Vector3f(theta[0].item(), theta[1].item(), theta[2].item()))

    def scale_rad(self, rad):
        if rad < 0.0001:
            rad = 0.0001
        self.translate_and_scale(mi.Vector3f(0.0, 0.0, -rad), rad)

    def scale_rad_z(self, theta):
        rad = theta[0][0]
        z = theta[1][0]
        if rad < 0.001:
            rad = 0.001

        if (z < 0) or (z - rad < 0.05):
            z = rad + 0.05

        self.translate_and_scale(mi.Vector3f(0.0, 0.0, z), rad)

class OnePlane(THZScene):
    def __init__(self, width=0.399, height=0.538, thickness=1, distance=0.345, roughness=0.2):
        super().__init__()

        self.scene_name = 'Plane'

        self.scene = mi.load_dict({
            'type' : 'scene',
            'rectangle1' : {
                'type': 'rectangle',
                'to_world': mi.ScalarTransform4f.scale([width, height, thickness]).translate([0.0, 0.0, distance]),
                'flip_normals': True,
                'bsdf': {
                    'type': 'roughconductor',
                    'alpha': roughness,
                    'distribution': 'ggx'
                }
            }
        })

        self.params = mi.traverse(self.scene)
        self.key = 'rectangle1.to_world'
        self.ref_transform = mi.Transform4f(self.params[self.key])

    def translate_z(self, z):
        self.translate(mi.Vector3f(0.0, 0.0, z))

    def set_height(self, z):
        self.translate_z(z)
        self.update_ref()

class SpherePlane(THZScene):
    def __init__(self, sphere_roughness=0.2, plane_width=0.399, plane_height=0.538, plane_thickness=1, plane_distance=0.345, plane_roughness=0.2):
        super().__init__()

        self.scene_name = 'Sphere_Plane'

        self.scene = mi.load_dict({
            'type' : 'scene',
            'sphere1': {
                'type': 'sphere',
                'center': [0.0, 0.0, 0.0],
                'radius': 1.0,
                'bsdf': {
                    'type': 'roughconductor',
                    'alpha': sphere_roughness,
                    'distribution': 'ggx'
                }
            },
            'rectangle1' : {
                'type': 'rectangle',
                'to_world': mi.ScalarTransform4f.scale([plane_width, plane_height, plane_thickness]).translate([0.0, 0.0, plane_distance]),
                'flip_normals': True,
                'bsdf': {
                    'type': 'roughconductor',
                    'alpha': plane_roughness,
                    'distribution': 'ggx'
                }
            }
        })

        self.params = mi.traverse(self.scene)
        self.key = 'sphere1.to_world'
        self.ref_transform = mi.Transform4f(self.params[self.key])

    def set_sphere_position(self, x, y, z):
        self.translate(mi.Vector3f(x, y, z))
        self.update_ref()

    def set_sphere_radius(self, rad):
        self.translate_and_scale(mi.Vector3f(0.0, 0.0, -rad), rad)
        self.update_ref()

    def translate_sphere_x(self, x):
        self.translate(mi.Vector3f(x, 0.0, 0.0))

class OneTorus(THZScene):
    def __init__(self, radius=0.02, thickness=0.01, roughness = 0.2):
        super().__init__()

        self.scene_name = 'Torus'

        gu.create_torus(radius, thickness)

        self.scene = mi.load_dict({
            'type' : 'scene',
            'torus': {
                'type': 'ply',
                'filename': f'../data/meshes/torus_{radius}_{thickness}.ply',
                'bsdf': {
                    'type': 'roughconductor',
                    'alpha': roughness,
                    'distribution': 'ggx'
                }
            }
        })

        self.params = mi.traverse(self.scene)
        self.key = 'torus.vertex_positions'
        self.ref_positions = dr.unravel(mi.Point3f, self.params['torus.vertex_positions'])

    def update_ref(self):
        self.ref_positions = dr.unravel(mi.Point3f, self.params['torus.vertex_positions'])

    def translate(self, theta):
        trafo = mi.Transform4f.translate([theta.x, theta.y, theta.z])

        self.params['torus.vertex_positions'] = dr.ravel(trafo @ self.ref_positions)
        self.params.update()

    def set_position(self, x, y, z):
        self.translate(mi.Vector3f(x, y, z))
        self.update_ref()

    def translate_x(self, x):
        self.translate(mi.Vector3f(x, 0.0, 0.0))

class TwoSpheres(THZScene):
    def __init__(self, roughness = 0.2):
        super().__init__()

        self.scene_name = 'TwoSpheres'

        self.scene = mi.load_dict({
            'type' : 'scene',
            'sphere1': {
                'type': 'sphere',
                'center': [0.0, 0.0, 0.0],
                'radius': 1.0,
                'bsdf': {
                    'type': 'roughconductor',
                    'alpha': roughness, # 0.001 - 0.01 (smooth), 0.1-0.3 (relatively rough), 0.3-0.7 (extremely rough)
                    'distribution': 'ggx'
                }
            },
            'sphere2': {
                'type': 'sphere',
                'center': [0.0, 0.0, 0.0],
                'radius': 1.0,
                'bsdf': {
                    'type': 'roughconductor',
                    'alpha': roughness, # 0.001 - 0.01 (smooth), 0.1-0.3 (relatively rough), 0.3-0.7 (extremely rough)
                    'distribution': 'ggx'
                }
            }
        })

        self.params = mi.traverse(self.scene)
        self.key1 = 'sphere1.to_world'
        self.ref_transform1 = mi.Transform4f(self.params[self.key1])
        self.key2 = 'sphere2.to_world'
        self.ref_transform2 = mi.Transform4f(self.params[self.key2])

    def set_position_1(self, x, y, z):
        self.key = self.key1
        self.ref_transform = self.ref_transform1
        self.translate(mi.Vector3f(x, y, z))
        self.update_ref()
        self.key = None
        self.ref_transform = None

    def set_position_2(self, x, y, z):
        self.key = self.key2
        self.ref_transform = self.ref_transform2
        self.translate(mi.Vector3f(x, y, z))
        self.update_ref()
        self.key = None
        self.ref_transform = None

    def set_radius_1(self, rad):
        self.key = self.key1
        self.ref_transform = self.ref_transform1
        self.translate_and_scale(mi.Vector3f(0.0, 0.0, -rad), rad)
        self.update_ref()
        self.key = None
        self.ref_transform = None

    def set_radius_2(self, rad):
        self.key = self.key2
        self.ref_transform = self.ref_transform2
        self.translate_and_scale(mi.Vector3f(0.0, 0.0, -rad), rad)
        self.update_ref()
        self.key = None

    def translate_x(self, theta):
        x = theta[0]
        self.key = self.key1
        self.ref_transform = self.ref_transform1
        self.translate(mi.Vector3f(x, 0.0, 0.0))
        x = theta[1]
        self.key = self.key2
        self.ref_transform = self.ref_transform2
        self.translate(mi.Vector3f(x, 0.0, 0.0))
        self.key = None
        self.ref_transform = None
