#!/usr/bin/env python3
"""
Script to calculate contact area between scaled object and human mesh.

This script:
1. Loads SMPLX model with actual pose parameters from smplx_parameters.json
2. Loads object mesh from obj_pcd_h_align.obj and simplifies using quadric decimation
3. Applies cam_trans transformation to both human and object meshes
4. Scales object_mesh by 5%
5. Calculates the intersection/contact area between them
6. Exports contact.json and PLY visualization files

Note: Uses the same vertex ordering as the training code to ensure consistency.
"""

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import os
import json
import argparse
import torch
import smplx

def load_obj_mesh(obj_path, cam_trans=None, simplify=False, target_triangles=8000):
    """Load OBJ mesh using Open3D and optionally simplify it"""
    mesh = o3d.io.read_triangle_mesh(obj_path)
    
    if simplify:
        print(f"   Original: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
        print(f"   Simplified: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    vertices = np.asarray(mesh.vertices)
    
    # # Apply camera transformation (same as hoi.py line 246)
    # if cam_trans is not None:
    #     vertices = vertices + cam_trans.reshape(1, 3)
    #     print(f"   Applied cam_trans: {cam_trans}")
    
    triangles = np.asarray(mesh.triangles)
    return mesh, vertices, triangles

def load_smplx_vertices(data_path):
    """
    Load human vertices using SMPLX model with actual pose (same as training code).
    This ensures vertex ordering matches the training contact array.
    
    Args:
        data_path: Path to directory containing smplx_parameters.json
    
    Returns:
        vertices: [10475, 3] SMPLX vertices in actual pose
    """
    # Load SMPLX parameters
    smplx_path = os.path.join(data_path, 'smplx_parameters.json')
    if not os.path.exists(smplx_path):
        raise FileNotFoundError(f"SMPLX parameters not found at {smplx_path}")
    
    smplx_param = json.load(open(smplx_path))
    if type(smplx_param) == list:
        smplx_param = smplx_param[0]
    
    # Create SMPLX model (same as in hoi.py)
    model_folder = "./data/SMPLX_NEUTRAL.npz"
    layer_arg = {
        'create_global_orient': False, 'create_body_pose': False, 
        'create_left_hand_pose': False, 'create_right_hand_pose': False, 
        'create_jaw_pose': False, 'create_leye_pose': False, 
        'create_reye_pose': False, 'create_betas': False, 
        'create_expression': False, 'create_transl': False
    }
    
    smplx_model = smplx.create(
        model_folder, model_type='smplx',
        gender='neutral', num_betas=10,
        num_expression_coeffs=10, use_pca=False, 
        use_face_contour=True, **layer_arg
    )
    
    # Use actual pose parameters from smplx_parameters.json (same as hoi.py line 215-222)
    zero_pose = torch.zeros((1, 3)).float()
    
    output = smplx_model(
        betas=torch.tensor(smplx_param['shape']),
        body_pose=torch.tensor(smplx_param['body_pose']),
        global_orient=torch.tensor(smplx_param['root_pose']),
        right_hand_pose=torch.tensor(smplx_param['rhand_pose']),
        left_hand_pose=torch.tensor(smplx_param['lhand_pose']),
        jaw_pose=torch.tensor(smplx_param['jaw_pose']),
        leye_pose=zero_pose,
        reye_pose=zero_pose,
        expression=torch.tensor(smplx_param['expr'])
    )
    
    # Apply camera transformation (same as hoi.py line 224-226)
    cam_trans = np.asarray(smplx_param['cam_trans']).reshape(3, 1)
    xyz = output.vertices[0].detach().cpu().numpy()
    # xyz = xyz + cam_trans.reshape(1, 3)
    
    print(f"   SMPLX vertices loaded: {xyz.shape[0]} vertices (with actual pose)")
    
    return xyz

def scale_mesh(vertices, scale_factor=1.05):
    """
    Scale mesh by moving vertices away from center.
    
    Args:
        vertices: [N, 3] array of vertex positions
        scale_factor: scaling factor (1.05 for 5% increase)
    
    Returns:
        scaled_vertices: [N, 3] scaled vertex positions
    """
    center = np.mean(vertices, axis=0)
    scaled = (vertices - center) * scale_factor + center
    return scaled

def calculate_contact_area(human_verts, scaled_obj_verts, scale_factor=5.0):
    """
    Calculate contact area between human and scaled object.
    
    Uses KDTree to find nearest neighbors and identifies vertices within
    a certain distance threshold as contact points.
    
    Args:
        human_verts: [N, 3] human mesh vertices
        scaled_obj_verts: [M, 3] scaled object vertices
        scale_factor: distance threshold multiplier (default 5%)
    
    Returns:
        human_contact_indices: boolean array [N] marking contact points on human
        object_contact_indices: boolean array [M] marking contact points on object
        contact_distances: [N] distances from each human vertex to nearest object vertex
    """
    # Build KDTree for object vertices
    obj_tree = cKDTree(scaled_obj_verts)
    
    # For each human vertex, find nearest object vertex
    distances, indices = obj_tree.query(human_verts, k=1)
    
    # Calculate dynamic threshold based on mesh scale
    # Use mean distance as baseline for threshold
    mean_distance = np.mean(distances)
    threshold = mean_distance * (scale_factor / 100.0)
    
    # Identify contact points (within threshold distance)
    human_contact = distances < threshold
    
    # Build KDTree for human vertices to find contacted object vertices
    human_tree = cKDTree(human_verts)
    distances_obj, indices_obj = human_tree.query(scaled_obj_verts, k=1)
    object_contact = distances_obj < threshold
    
    print(f"Mean distance: {mean_distance:.6f}")
    print(f"Distance threshold: {threshold:.6f}")
    print(f"Human contact points: {np.sum(human_contact)} / {len(human_contact)}")
    print(f"Object contact points: {np.sum(object_contact)} / {len(object_contact)}")
    
    return human_contact, object_contact, distances

def create_contact_visualization_ply(vertices, contact_mask, output_path):
    """
    Create PLY file with contact points highlighted.
    
    Args:
        vertices: [N, 3] vertex positions
        contact_mask: [N] boolean array marking contact points
        output_path: path to save PLY file
    """
    # Create point cloud with colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    
    # Create colors: contact points in red, others in blue
    colors = np.zeros((len(vertices), 3))
    colors[~contact_mask] = [0.2, 0.2, 0.8]  # Blue for non-contact
    colors[contact_mask] = [1.0, 0.0, 0.0]   # Red for contact
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save as PLY
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"   Saved: {output_path}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Calculate contact area between human and object meshes")
    parser.add_argument("--path", type=str, required=True, 
                        help="Path to directory containing smplx_parameters.json and object_mesh.obj")
    args = parser.parse_args()
    
    # Construct paths
    object_mesh_path = os.path.join(args.path, "object_mesh.obj")
    smplx_param_path = os.path.join(args.path, "smplx_parameters.json")
    
    # Check if files exist
    if not os.path.exists(smplx_param_path):
        print(f"Error: SMPLX parameters not found at {smplx_param_path}")
        return
    if not os.path.exists(object_mesh_path):
        print(f"Error: Object mesh not found at {object_mesh_path}")
        return
    
    print("=" * 80)
    print("Contact Area Calculation Script")
    print("=" * 80)
    print(f"Input directory: {args.path}")
    
    # Load SMPLX parameters for cam_trans
    smplx_param = json.load(open(smplx_param_path))
    if type(smplx_param) == list:
        smplx_param = smplx_param[0]
    cam_trans = np.asarray(smplx_param['cam_trans']).reshape(3, 1)
    
    # Load meshes
    print("\n1. Loading meshes...")
    print("   Loading human vertices from SMPLX model...")
    h_verts = load_smplx_vertices(args.path)
    
    print("   Loading and simplifying object mesh...")
    obj_mesh, obj_verts, obj_faces = load_obj_mesh(
        object_mesh_path, 
        cam_trans=cam_trans, 
        simplify=True, 
        target_triangles=8000
    )
    
    # Scale object mesh by 5%
    print("\n2. Scaling object mesh by 5%...")
    scaled_obj_verts = scale_mesh(obj_verts, scale_factor=1.05)
    print(f"   Scaled object: {len(scaled_obj_verts)} vertices")
    
    # Calculate contact area
    print("\n3. Calculating contact area...")
    human_contact, obj_contact, distances = calculate_contact_area(
        h_verts, scaled_obj_verts, scale_factor=5.0
    )
    
    # Save contact.json
    print("\n4. Saving contact.json...")
    contact_list = np.concatenate([
        human_contact.astype(bool),
        obj_contact.astype(bool)
    ]).tolist()
    
    contact_output_path = os.path.join(args.path, "contact.json")
    with open(contact_output_path, 'w') as f:
        json.dump(contact_list, f)
    print(f"   Saved: {contact_output_path}")
    print(f"   Total vertices: {len(contact_list)}")
    print(f"   Contact points: {sum(contact_list)}")
    
    # Create visualizations
    print("\n5. Creating visualization files...")
    
    # Human mesh with contact areas
    human_vis_path = os.path.join(args.path, "h_contact_gt.ply")
    create_contact_visualization_ply(h_verts, human_contact, human_vis_path)
    
    # Scaled object mesh with contact areas
    obj_vis_path = os.path.join(args.path, "o_contact_gt.ply")
    create_contact_visualization_ply(scaled_obj_verts, obj_contact, obj_vis_path)
    
    print("\n" + "=" * 80)
    print("Contact area calculation complete!")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - contact.json: Boolean array of contact points (length: {len(contact_list)})")
    print(f"  - h_contact_gt.ply: Human mesh with contact visualization")
    print(f"  - o_contact_gt.ply: Object mesh with contact visualization")

if __name__ == "__main__":
    main()

