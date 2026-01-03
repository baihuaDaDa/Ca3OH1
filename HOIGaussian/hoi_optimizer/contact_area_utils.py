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
    Load human vertices and faces using SMPLX model with actual pose (same as training code).
    This ensures vertex ordering matches the training contact array.
    
    Args:
        data_path: Path to directory containing smplx_parameters.json
    
    Returns:
        vertices: [10475, 3] SMPLX vertices in actual pose
        faces: [N, 3] SMPLX face indices
    """
    # Load SMPLX parameters
    smplx_path = os.path.join(data_path, 'smplx_parameters.json')
    if not os.path.exists(smplx_path):
        raise FileNotFoundError(f"SMPLX parameters not found at {smplx_path}")
    
    smplx_param = json.load(open(smplx_path))
    if type(smplx_param) == list:
        smplx_param = smplx_param[0]
    
    # Create SMPLX model (same as in hoi.py)
    # Use absolute path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(script_dir, "..", "data", "SMPLX_NEUTRAL.npz")
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
    
    # Get faces from SMPLX model
    faces = smplx_model.faces.astype(np.int32)
    
    print(f"   SMPLX loaded: {xyz.shape[0]} vertices, {faces.shape[0]} faces (with actual pose)")
    
    return xyz, faces

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

def calculate_contact_area_kdtree(human_verts, scaled_obj_verts, scale_factor=5.0):
    """
    Calculate contact area between human and scaled object using KDTree distance method.
    (Backup method)
    
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

def calculate_contact_area_interior(human_mesh, human_verts, obj_mesh, obj_verts):
    """
    Calculate contact area using interior point detection only.
    (Backup method)
    
    Contact points are defined as:
    - Human vertices that are inside the object mesh
    - Object vertices that are inside the human mesh
    
    Args:
        human_mesh: Open3D TriangleMesh of human
        human_verts: [N, 3] human mesh vertices
        obj_mesh: Open3D TriangleMesh of object
        obj_verts: [M, 3] object vertices
    
    Returns:
        human_contact: boolean array [N] marking contact points on human
        object_contact: boolean array [M] marking contact points on object
    """
    # Create raycasting scene for both meshes
    human_scene = o3d.t.geometry.RaycastingScene()
    obj_scene = o3d.t.geometry.RaycastingScene()
    
    # Convert to tensor meshes and add to scenes
    human_tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(human_mesh)
    obj_tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(obj_mesh)
    
    human_scene.add_triangles(human_tensor_mesh)
    obj_scene.add_triangles(obj_tensor_mesh)
    
    # Check which human vertices are inside object mesh
    human_query_points = o3d.core.Tensor(human_verts.astype(np.float32), dtype=o3d.core.Dtype.Float32)
    human_occupancy = obj_scene.compute_occupancy(human_query_points)
    human_contact = human_occupancy.numpy() > 0.5
    
    # Check which object vertices are inside human mesh
    obj_query_points = o3d.core.Tensor(obj_verts.astype(np.float32), dtype=o3d.core.Dtype.Float32)
    obj_occupancy = human_scene.compute_occupancy(obj_query_points)
    object_contact = obj_occupancy.numpy() > 0.5
    
    print(f"  Interior detection - Human: {np.sum(human_contact)} / {len(human_contact)}")
    print(f"  Interior detection - Object: {np.sum(object_contact)} / {len(object_contact)}")
    
    return human_contact, object_contact

def build_vertex_to_neighbors_map(vertices, faces):
    """
    Build a mapping from each vertex to its neighboring vertices (vertices sharing faces).
    
    Args:
        vertices: [N, 3] vertex positions
        faces: [F, 3] face indices
    
    Returns:
        neighbors: dict mapping vertex_idx -> set of neighbor vertex indices
    """
    neighbors = {i: set() for i in range(len(vertices))}
    
    for face in faces:
        v0, v1, v2 = face
        # Each vertex in the face is a neighbor to the other two
        neighbors[v0].update([v1, v2])
        neighbors[v1].update([v0, v2])
        neighbors[v2].update([v0, v1])
    
    return neighbors

def calculate_contact_area_proximity(human_mesh, human_verts, obj_mesh, obj_verts, 
                                     distance_ratio=0.5):
    """
    Calculate contact area using mesh-topology-aware proximity detection.
    
    Contact is determined by:
    - For each human vertex, compute the minimum distance to its mesh neighbors (vertices sharing faces)
    - A human vertex is in contact if its distance to the nearest object vertex
      is less than distance_ratio * min_neighbor_distance
    - The corresponding nearest object vertex is also marked as contact
    
    Args:
        human_mesh: Open3D TriangleMesh of human
        human_verts: [N, 3] human mesh vertices
        obj_mesh: Open3D TriangleMesh of object
        obj_verts: [M, 3] object vertices
        distance_ratio: ratio threshold for contact detection (default: 0.5 = 50%)
                       Recommended values: 0.3-1.0 (30%-100% of local mesh resolution)
    
    Returns:
        human_contact: boolean array [N] marking contact points on human
        object_contact: boolean array [M] marking contact points on object
    """
    # Get faces
    human_faces = np.asarray(human_mesh.triangles)
    
    # Build neighbor maps for human mesh topology only
    print("  Building human vertex neighbor map...")
    human_neighbors = build_vertex_to_neighbors_map(human_verts, human_faces)
    
    # Calculate minimum neighbor distance for each human vertex (local mesh resolution)
    print("  Calculating human mesh local resolution...")
    human_min_neighbor_dist = np.zeros(len(human_verts))
    for i, neighbors in human_neighbors.items():
        if neighbors:
            distances = [np.linalg.norm(human_verts[i] - human_verts[j]) for j in neighbors]
            human_min_neighbor_dist[i] = min(distances)
        else:
            human_min_neighbor_dist[i] = np.inf
    
    # Build KDTree for efficient nearest neighbor search
    print("  Building KD-Tree for object...")
    obj_tree = cKDTree(obj_verts)
    
    # For each human vertex, find nearest object vertex and check distance
    print("  Checking human-to-object proximity...")
    human_contact = np.zeros(len(human_verts), dtype=bool)
    object_contact = np.zeros(len(obj_verts), dtype=bool)
    distances_h2o, nearest_obj_indices = obj_tree.query(human_verts, k=1)
    
    for i in range(len(human_verts)):
        # Dynamic threshold based on local mesh resolution
        threshold = distance_ratio * human_min_neighbor_dist[i]
        if distances_h2o[i] < threshold:
            human_contact[i] = True
            # Mark the corresponding nearest object vertex as contact
            object_contact[nearest_obj_indices[i]] = True
    
    print(f"  Proximity detection - Human: {np.sum(human_contact)} / {len(human_contact)}")
    print(f"  Proximity detection - Object: {np.sum(object_contact)} / {len(obj_verts)} (via human contact)")
    print(f"  Distance ratio: {distance_ratio:.4f} (ratio of local mesh resolution)")
    
    return human_contact, object_contact

def calculate_contact_area(human_mesh, human_verts, obj_mesh, obj_verts, 
                          distance_ratio=0.02):
    """
    Calculate contact area using combined interior and proximity detection.
    
    Contact points are defined as:
    1. Interior detection: vertices inside the other mesh
    2. Proximity detection: vertices close to the other mesh (based on local mesh resolution)
    
    Args:
        human_mesh: Open3D TriangleMesh of human
        human_verts: [N, 3] human mesh vertices
        obj_mesh: Open3D TriangleMesh of object
        obj_verts: [M, 3] object vertices
        distance_ratio: ratio threshold for proximity detection (default: 0.02 = 2%)
    
    Returns:
        human_contact: boolean array [N] marking contact points on human
        object_contact: boolean array [M] marking contact points on object
    """
    print("Calculating contact area (combined method)...")
    
    # Method 1: Interior detection
    print("\n1. Interior point detection:")
    human_interior, obj_interior = calculate_contact_area_interior(
        human_mesh, human_verts, obj_mesh, obj_verts
    )
    
    # Method 2: Proximity detection
    print("\n2. Proximity detection:")
    human_proximity, obj_proximity = calculate_contact_area_proximity(
        human_mesh, human_verts, obj_mesh, obj_verts, distance_ratio
    )
    
    # Combine results (union of both methods)
    human_contact = human_interior | human_proximity
    object_contact = obj_interior | obj_proximity
    
    print("\n3. Combined results:")
    print(f"  Human contact points: {np.sum(human_contact)} / {len(human_contact)} "
          f"({np.sum(human_contact)/len(human_contact)*100:.2f}%)")
    print(f"    - From interior: {np.sum(human_interior)}")
    print(f"    - From proximity: {np.sum(human_proximity)}")
    print(f"    - Overlap: {np.sum(human_interior & human_proximity)}")
    
    print(f"  Object contact points: {np.sum(object_contact)} / {len(object_contact)} "
          f"({np.sum(object_contact)/len(object_contact)*100:.2f}%)")
    print(f"    - From interior: {np.sum(obj_interior)}")
    print(f"    - From proximity: {np.sum(obj_proximity)}")
    print(f"    - Overlap: {np.sum(obj_interior & obj_proximity)}")
    
    return human_contact, object_contact

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
    print("   Loading human mesh from SMPLX model...")
    h_verts, h_faces = load_smplx_vertices(args.path)
    
    # Create human mesh from SMPLX vertices and faces
    human_mesh = o3d.geometry.TriangleMesh()
    human_mesh.vertices = o3d.utility.Vector3dVector(h_verts)
    human_mesh.triangles = o3d.utility.Vector3iVector(h_faces)
    
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
    scaled_obj_mesh = o3d.geometry.TriangleMesh()
    scaled_obj_mesh.vertices = o3d.utility.Vector3dVector(scaled_obj_verts)
    scaled_obj_mesh.triangles = obj_mesh.triangles
    print(f"   Scaled object: {len(scaled_obj_verts)} vertices")
    
    # Calculate contact area using interior detection
    print("\n3. Calculating contact area (interior detection method)...")
    human_contact, obj_contact = calculate_contact_area(
        human_mesh, h_verts, scaled_obj_mesh, scaled_obj_verts
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

