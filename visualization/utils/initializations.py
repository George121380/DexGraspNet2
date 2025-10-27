"""
Bimanual hand initialization utilities for grasp generation.
Provides functions to initialize hand poses and orientations around target objects.
"""

import math
import numpy as np
import torch
import transforms3d
import trimesh as tm
import pytorch3d.structures
import pytorch3d.ops

from utils.hand_model import HandModel
from utils.common import normalize_tensor

def initialize_convex_hull(left_hand_model, object_model, args):
    """
    Initialize grasp translation, rotation, joint angles, and contact point indices.
    
    Args:
        left_hand_model: HandModel instance for left hand
        object_model: ObjectModel instance containing target objects
        args: Configuration namespace with initialization parameters
        
    Returns:
        tuple: Normal vectors and sample points from object surface
    """
        
    device = left_hand_model.device
    n_objects = len(object_model.object_mesh_list)
    batch_per_obj = object_model.batch_size_each
    total_batch_size = n_objects * batch_per_obj

    # Initialize translation and rotation tensors
    translation = torch.zeros([total_batch_size, 3], dtype=torch.float, device=device)
    rotation = torch.zeros([total_batch_size, 3, 3], dtype=torch.float, device=device)

    if left_hand_model.handedness != 'left_hand':
        raise ValueError("This function should initialize the left hand model")

    for i in range(n_objects):
        # Get inflated convex hull
        mesh_origin = object_model.object_mesh_list[i].convex_hull
        vertices = mesh_origin.vertices.copy()
        faces = mesh_origin.faces
        vertices *= object_model.object_scale_tensor[i].max().item()
        mesh_origin = tm.Trimesh(vertices, faces)
        mesh_origin.faces = mesh_origin.faces[mesh_origin.remove_degenerate_faces()]
        vertices += 0.2 * vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        mesh = tm.Trimesh(vertices=vertices, faces=faces).convex_hull
        vertices = torch.tensor(mesh.vertices, dtype=torch.float, device=device)
        faces = torch.tensor(mesh.faces, dtype=torch.float, device=device)
        mesh_pytorch3d = pytorch3d.structures.Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))

        # Sample points from mesh surface
        dense_cloud = pytorch3d.ops.sample_points_from_meshes(mesh_pytorch3d, num_samples=100 * batch_per_obj)
        p = pytorch3d.ops.sample_farthest_points(dense_cloud, K=batch_per_obj)[0][0]
        closest_points, _, _ = mesh_origin.nearest.on_surface(p.detach().cpu().numpy())
        closest_points = torch.tensor(closest_points, dtype=torch.float, device=device)
        n = (closest_points - p) / (closest_points - p).norm(dim=1).unsqueeze(1)

        # Sample initialization parameters
        rand_vals = torch.rand([4, batch_per_obj], dtype=torch.float, device=device)
        distance = args.distance_lower + (args.distance_upper - args.distance_lower) * rand_vals[0]
        cone_angle = args.theta_lower + (args.theta_upper - args.theta_lower) * rand_vals[1]
        azimuth = 2 * math.pi * rand_vals[2]
        roll = 2 * math.pi * rand_vals[3]

        # Solve transformation matrices
        # hand_rot: rotate the hand to align its grasping direction with the +z axis
        # cone_rot: jitter the hand's orientation in a cone  
        # world_rot and translation: transform the hand to a position corresponding to point p sampled from the inflated convex hull
        cone_rot = torch.zeros([batch_per_obj, 3, 3], dtype=torch.float, device=device)
        world_rot = torch.zeros([batch_per_obj, 3, 3], dtype=torch.float, device=device)
        for j in range(batch_per_obj):
            cone_rot[j] = torch.tensor(
                transforms3d.euler.euler2mat(azimuth[j], cone_angle[j], roll[j], axes='rzxz'),
                dtype=torch.float, device=device
            )
            world_rot[j] = torch.tensor(
                transforms3d.euler.euler2mat(
                    math.atan2(n[j, 1], n[j, 0]) - math.pi / 2, -math.acos(n[j, 2]), 0, axes='rzxz'
                ), dtype=torch.float, device=device
            )
        start_idx = i * batch_per_obj
        end_idx = start_idx + batch_per_obj
        z_vec = torch.tensor([0, 0, 1], dtype=torch.float, device=device).reshape(1, -1, 1)
        translation[start_idx:end_idx] = p - distance.unsqueeze(1) * (world_rot @ cone_rot @ z_vec).squeeze(2)
        hand_rot = torch.tensor(
            transforms3d.euler.euler2mat(0, -np.pi / 3, 0, axes='rzxz'), dtype=torch.float, device=device
        )
        rotation[start_idx:end_idx] = world_rot @ cone_rot @ (-hand_rot)
    
    # Initialize joint angles using truncated normal distribution  
    # joint_angles_mu: hand-crafted canonicalized hand articulation
    joint_angles_mu = torch.tensor([
        0.1, 0, -0.6, 0, 0, 0, -0.6, 0, -0.1, 0, -0.6, 0,
        0, -0.2, 0, -0.6, 0, 0, -1.2, 0, -0.2, 0
    ], dtype=torch.float, device=device)
    joint_angles_sigma = args.jitter_strength * (left_hand_model.joints_upper - left_hand_model.joints_lower)
    joint_angles = torch.zeros([total_batch_size, left_hand_model.n_dofs], dtype=torch.float, device=device)
    for i in range(left_hand_model.n_dofs):
        torch.nn.init.trunc_normal_(
            joint_angles[:, i], joint_angles_mu[i], joint_angles_sigma[i],
            left_hand_model.joints_lower[i] - 1e-6, left_hand_model.joints_upper[i] + 1e-6
        )

    hand_pose = torch.cat([
        translation,
        rotation.transpose(1, 2)[:, :2].reshape(-1, 6),
        joint_angles
    ], dim=1)
    hand_pose.requires_grad_()

    # Initialize contact point indices
    # Handle both old and new parameter names for backward compatibility
    n_contact = getattr(args, 'num_contacts', getattr(args, 'n_contact', 4))
    contact_indices = torch.randint(
        left_hand_model.n_contact_candidates, size=[total_batch_size, n_contact], device=device
    )

    left_hand_model.set_parameters(hand_pose, contact_indices)
    return n, p


def initialize_dual_hand(right_hand_model, object_model, args):
    """
    Initialize both hands' positions and rotations to grasp an object symmetrically.
    
    Args:
        right_hand_model: HandModel instance for right hand
        object_model: ObjectModel instance containing target objects
        args: Configuration namespace with initialization parameters
        
    Returns:
        tuple: (left_hand_model, right_hand_model) with initialized poses
    """
    
    device = right_hand_model.device
    n_objects = len(object_model.object_mesh_list)
    batch_per_obj = object_model.batch_size_each
    total_batch_size = n_objects * batch_per_obj    
    
    # Create left hand model
    left_hand_model = HandModel(
        mjcf_path='mjcf/left_shadow_hand.xml',
        mesh_path='mjcf/meshes',
        contact_points_path='mjcf/left_hand_contact_points.json',
        penetration_points_path='mjcf/penetration_points.json',
        device=device,
        handedness='left_hand'
    )
    
    n, p = initialize_convex_hull(left_hand_model, object_model, args)

    # Compute the right hand's parameters based on the left hand's
    rotation_right = torch.zeros([total_batch_size, 3, 3], dtype=torch.float, device=device)
    translation_right = torch.zeros([total_batch_size, 3], dtype=torch.float, device=device)

    for i in range(n_objects):
        start_idx = i * batch_per_obj
        end_idx = start_idx + batch_per_obj
        # Mirror the normal vectors and points for symmetric grasp
        n[start_idx:end_idx, 0] = -n[start_idx:end_idx, 0]
        n[start_idx:end_idx, 1] = -n[start_idx:end_idx, 1]
        n[start_idx:end_idx, 2] = n[start_idx:end_idx, 2]

        p[start_idx:end_idx, 0] = -p[start_idx:end_idx, 0]
        p[start_idx:end_idx, 1] = -p[start_idx:end_idx, 1]
        p[start_idx:end_idx, 2] = p[start_idx:end_idx, 2]
        
        # Sample parameters for right hand
        rand_vals = torch.rand([4, batch_per_obj], dtype=torch.float, device=device)
        distance = args.distance_lower + (args.distance_upper - args.distance_lower) * rand_vals[0]
        cone_angle = args.theta_lower + (args.theta_upper - args.theta_lower) * rand_vals[1]
        azimuth = 2 * math.pi * rand_vals[2]
        roll = 2 * math.pi * rand_vals[3]

        # Solve transformation matrices for right hand
        # hand_rot: rotate the hand to align its grasping direction with the +z axis
        # cone_rot: jitter the hand's orientation in a cone
        # world_rot and translation: transform the hand to a position corresponding to point p sampled from the inflated convex hull
        cone_rot = torch.zeros([batch_per_obj, 3, 3], dtype=torch.float, device=device)
        world_rot = torch.zeros([batch_per_obj, 3, 3], dtype=torch.float, device=device)
        for j in range(batch_per_obj):
            cone_rot[j] = torch.tensor(
                transforms3d.euler.euler2mat(azimuth[j], cone_angle[j], roll[j], axes='rzxz'),
                dtype=torch.float, device=device
            )
            world_rot[j] = torch.tensor(
                transforms3d.euler.euler2mat(
                    math.atan2(n[j, 1], n[j, 0]) - math.pi / 2, -math.acos(n[j, 2]), 0, axes='rzxz'
                ), dtype=torch.float, device=device
            )
        z_vec = torch.tensor([0, 0, 1], dtype=torch.float, device=device).reshape(1, -1, 1)
        translation_right[start_idx:end_idx] = p - distance.unsqueeze(1) * (world_rot @ cone_rot @ z_vec).squeeze(2)
        hand_rot = torch.tensor(
            transforms3d.euler.euler2mat(0, -np.pi / 3, 0, axes='rzxz'), dtype=torch.float, device=device
        )
        rotation_right[start_idx:end_idx] = world_rot @ cone_rot @ hand_rot


    # Initialize right hand joint angles
    joint_angles_mu = torch.tensor([
        0.1, 0, 0.6, 0, 0, 0, 0.6, 0, -0.1, 0, 0.6, 0,
        0, -0.2, 0, 0.6, 0, 0, 1.2, 0, -0.2, 0
    ], dtype=torch.float, device=device)
    joint_angles_sigma = args.jitter_strength * (right_hand_model.joints_upper - right_hand_model.joints_lower)
    joint_angles = torch.zeros([total_batch_size, right_hand_model.n_dofs], dtype=torch.float, device=device)
    for i in range(right_hand_model.n_dofs):
        torch.nn.init.trunc_normal_(
            joint_angles[:, i], joint_angles_mu[i], joint_angles_sigma[i],
            right_hand_model.joints_lower[i] - 1e-6, right_hand_model.joints_upper[i] + 1e-6
        )

    # Assemble right hand pose
    hand_pose_right = torch.cat([
        translation_right,
        rotation_right.transpose(1, 2)[:, :2].reshape(-1, 6),
        joint_angles
    ], dim=1)
    hand_pose_right.requires_grad_()

    # Set parameters for right hand model
    n_contact = getattr(args, 'num_contacts', getattr(args, 'n_contact', 4))
    contact_indices = torch.randint(
        right_hand_model.n_contact_candidates, size=[total_batch_size, n_contact], device=device
    )
    right_hand_model.set_parameters(hand_pose_right, contact_indices)

    return left_hand_model, right_hand_model


def _parse_target_vec3(s: str, device):
    if s is None:
        return None
    parts = s.strip().split()
    if len(parts) != 3:
        raise ValueError(f"Target must be 'x y z', got: {s}")
    return torch.tensor([float(parts[0]), float(parts[1]), float(parts[2])], dtype=torch.float, device=device)


def _rotation_align_z_to_dir(direction: torch.Tensor, device):
    # direction: (3,), returns (3,3) rotation matrix mapping +z to this direction
    dir_n = direction / (direction.norm() + 1e-8)
    z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float, device=device)
    if torch.allclose(dir_n, z):
        return torch.eye(3, dtype=torch.float, device=device)
    if torch.allclose(dir_n, -z):
        R = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device))
        return R
    v = torch.cross(z, dir_n)
    s = torch.norm(v)
    c = torch.dot(z, dir_n)
    vx = torch.tensor([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]], dtype=torch.float, device=device)
    R = torch.eye(3, device=device, dtype=torch.float) + vx + (vx @ vx) * ((1 - c) / (s ** 2 + 1e-8))
    return R


def initialize_dual_hand_at_targets(right_hand_model, object_model, args):
    """
    Initialize both hands near specified target points on the object surface.
    Left near A, right near B, palms facing toward the targets.
    """
    device = right_hand_model.device
    n_objects = len(object_model.object_mesh_list)
    batch_per_obj = object_model.batch_size_each
    total_batch_size = n_objects * batch_per_obj

    # Left hand model (create anew like original)
    left_hand_model = HandModel(
        mjcf_path='mjcf/left_shadow_hand.xml',
        mesh_path='mjcf/meshes',
        contact_points_path='mjcf/left_hand_contact_points.json',
        penetration_points_path='mjcf/penetration_points.json',
        device=device,
        handedness='left_hand'
    )

    # Parse targets (object coords)
    A_obj = _parse_target_vec3(getattr(args, 'left_target', None), device)
    B_obj = _parse_target_vec3(getattr(args, 'right_target', None), device)

    # Distance/jitter params
    base_d = float(getattr(args, 'target_distance', 0.22))
    jitter_d = float(getattr(args, 'target_jitter_dist', 0.0))
    jitter_angle = float(getattr(args, 'target_jitter_angle', 0.0))

    # Allocate tensors
    translation_L = torch.zeros([total_batch_size, 3], dtype=torch.float, device=device)
    rotation_L = torch.zeros([total_batch_size, 3, 3], dtype=torch.float, device=device)
    translation_R = torch.zeros([total_batch_size, 3], dtype=torch.float, device=device)
    rotation_R = torch.zeros([total_batch_size, 3, 3], dtype=torch.float, device=device)

    z_vec = torch.tensor([0, 0, 1], dtype=torch.float, device=device).reshape(1, -1, 1)
    hand_rot = torch.tensor(transforms3d.euler.euler2mat(0, -np.pi / 3, 0, axes='rzxz'), dtype=torch.float, device=device)

    for i in range(n_objects):
        start_idx = i * batch_per_obj
        end_idx = start_idx + batch_per_obj

        # Determine scale for this object batch (use max or per-sample)
        # Use per-sample scale (batch_per_obj,) then expand to (batch,1)
        scales = object_model.object_scale_tensor[i]  # (batch_per_obj,)
        scales = scales.reshape(-1, 1)

        # Targets in world (object) coords after scale
        # Fallback: if only left_target provided, mirror to get right target
        if A_obj is None:
            raise ValueError('left_target must be provided when init_at_targets is enabled')
        A_scaled = (A_obj.unsqueeze(0) * scales).to(device)

        if B_obj is None:
            # mirror x,y to get a symmetric right target
            B_obj_mirror = A_obj.clone()
            B_obj_mirror[0] = -B_obj_mirror[0]
            B_obj_mirror[1] = -B_obj_mirror[1]
            B_scaled = (B_obj_mirror.unsqueeze(0) * scales).to(device)
        else:
            B_scaled = (B_obj.unsqueeze(0) * scales).to(device)

        # Sample jitter
        if jitter_d > 0:
            d_offsets = (torch.rand(batch_per_obj, 1, device=device) * 2 - 1) * jitter_d
        else:
            d_offsets = torch.zeros(batch_per_obj, 1, device=device)
        distances = base_d + d_offsets  # (batch,1)

        # Jitter direction in cone if requested
        def jitter_dir(dir_vec):
            dv = dir_vec
            if jitter_angle > 0:
                # Sample in cone around dv
                # Reuse sample_cone_uniform from common if available; otherwise simple axis-angle jitter
                axis = torch.randn(3, device=device)
                axis = axis / (axis.norm() + 1e-8)
                angle = (torch.rand(1, device=device) - 0.5) * 2 * jitter_angle
                K = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], device=device)
                Rj = torch.eye(3, device=device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
                dv = (Rj @ dv)
            return dv / (dv.norm() + 1e-8)

        # For each sample in this object batch
        for j in range(batch_per_obj):
            idx = start_idx + j
            A = A_scaled[j]
            B = B_scaled[j]

            # Approach directions: from hand to target => align hand +z toward target
            # Choose provisional hand positions by back-off along direction; compute dirs
            dir_A = A.clone(); dir_A = dir_A / (dir_A.norm() + 1e-8)
            dir_B = B.clone(); dir_B = dir_B / (dir_B.norm() + 1e-8)

            dir_A = jitter_dir(dir_A)
            dir_B = jitter_dir(dir_B)

            translation_L[idx] = A - distances[j] * dir_A
            translation_R[idx] = B - distances[j] * dir_B

            # Align palm (+z) toward object center
            base_center = torch.tensor(object_model.object_mesh_list[i].vertices.mean(axis=0), dtype=torch.float, device=device)
            center_j = base_center * scales[j][0]
            dir_center_A = center_j - translation_L[idx]
            dir_center_B = center_j - translation_R[idx]
            world_rot_A = _rotation_align_z_to_dir(dir_center_A, device)
            world_rot_B = _rotation_align_z_to_dir(dir_center_B, device)

            # Optional twist around approach axis (+z after alignment)
            twist_range = float(getattr(args, 'target_twist_range', 0.0))
            # Use the SAME twist for left and right to keep fingertip orientation symmetric
            if twist_range > 0:
                phi = (torch.rand(1, device=device).item() * 2 - 1) * twist_range  # [-range, range]
            else:
                phi = (torch.rand(1, device=device).item()) * 2 * np.pi            # [0, 2π)
            # Build left/right twist with optional mirror and constant offset for right
            right_offset = float(getattr(args, 'right_twist_offset', 0.0))
            if bool(getattr(args, 'twist_mirror', False)):
                phi_R = -phi + right_offset
            else:
                phi_R = phi + right_offset
            Rz_A = torch.tensor(transforms3d.euler.euler2mat(0, 0, phi, axes='rzxz'), dtype=torch.float, device=device)
            Rz_B = torch.tensor(transforms3d.euler.euler2mat(0, 0, phi_R, axes='rzxz'), dtype=torch.float, device=device)
            world_rot_A = world_rot_A @ Rz_A
            world_rot_B = world_rot_B @ Rz_B

            rotation_L[idx] = world_rot_A @ (-hand_rot)
            rotation_R[idx] = world_rot_B @ hand_rot

    # Initialize joint angles (same as original)
    joint_angles_mu_L = torch.tensor([
        0.1, 0, -0.6, 0, 0, 0, -0.6, 0, -0.1, 0, -0.6, 0,
        0, -0.2, 0, -0.6, 0, 0, -1.2, 0, -0.2, 0
    ], dtype=torch.float, device=device)
    sigma_L = args.jitter_strength * (left_hand_model.joints_upper - left_hand_model.joints_lower)
    joints_L = torch.zeros([total_batch_size, left_hand_model.n_dofs], dtype=torch.float, device=device)
    for k in range(left_hand_model.n_dofs):
        torch.nn.init.trunc_normal_(
            joints_L[:, k], joint_angles_mu_L[k], sigma_L[k],
            left_hand_model.joints_lower[k] - 1e-6, left_hand_model.joints_upper[k] + 1e-6
        )

    hand_pose_L = torch.cat([
        translation_L,
        rotation_L.transpose(1, 2)[:, :2].reshape(-1, 6),
        joints_L
    ], dim=1)
    hand_pose_L.requires_grad_()

    n_contact = getattr(args, 'num_contacts', getattr(args, 'n_contact', 4))
    contact_indices_L = torch.randint(left_hand_model.n_contact_candidates, size=[total_batch_size, n_contact], device=device)
    left_hand_model.set_parameters(hand_pose_L, contact_indices_L)

    # Right hand joints
    sigma_R = args.jitter_strength * (right_hand_model.joints_upper - right_hand_model.joints_lower)
    joint_angles_mu_R = torch.tensor([
        0.1, 0, 0.6, 0, 0, 0, 0.6, 0, -0.1, 0, 0.6, 0,
        0, -0.2, 0, 0.6, 0, 0, 1.2, 0, -0.2, 0
    ], dtype=torch.float, device=device)
    joints_R = torch.zeros([total_batch_size, right_hand_model.n_dofs], dtype=torch.float, device=device)
    for k in range(right_hand_model.n_dofs):
        torch.nn.init.trunc_normal_(
            joints_R[:, k], joint_angles_mu_R[k], joints_R[:, k].new_full((), sigma_R[k]),
            right_hand_model.joints_lower[k] - 1e-6, right_hand_model.joints_upper[k] + 1e-6
        )

    hand_pose_R = torch.cat([
        translation_R,
        rotation_R.transpose(1, 2)[:, :2].reshape(-1, 6),
        joints_R
    ], dim=1)
    hand_pose_R.requires_grad_()

    contact_indices_R = torch.randint(right_hand_model.n_contact_candidates, size=[total_batch_size, n_contact], device=device)
    right_hand_model.set_parameters(hand_pose_R, contact_indices_R)

    return left_hand_model, right_hand_model
