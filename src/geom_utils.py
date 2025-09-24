import trimesh
import os.path as osp

def create_torus(major_radius=0.03, minor_radius=0.01):
    if osp.isfile(f'../data/meshes/torus_{major_radius}_{minor_radius}.ply'):
        return
    
    # Generate a torus
    mesh = trimesh.creation.torus(major_radius=major_radius, minor_radius=minor_radius)
    
    # Save as a PLY file
    mesh.export(f'../data/meshes/torus_{major_radius}_{minor_radius}.ply')