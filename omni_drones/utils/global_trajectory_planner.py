import torch
from omni_drones.utils.bspline import get_knots, splev_torch


def compute_global_trajectory(
    gate_positions,
    gate_normals,
    method="linear",
    num_points=100,
    tangent_scale=1.0,
    k=3,
):
    """
    Generate a global trajectory through the gates, matching gate normals.

    Args:
        gate_positions: (N_gates, 3) tensor of gate center positions.
        gate_normals: (N_gates, 3) tensor of gate normals (desired tangents).
        method: "linear" (default) or "spline" for smooth interpolation.
        num_points: Number of points in the trajectory.
        tangent_scale: Scaling factor for tangent magnitude at endpoints.
        k: Degree of B-spline (ignored if method is "linear").

    Returns:
        trajectory: (num_points, 3) tensor of waypoints.
    """
    if method == "linear":
        points = []
        for i in range(len(gate_positions) - 1):
            seg = torch.linspace(
                0,
                1,
                num_points // (len(gate_positions) - 1),
                device=gate_positions.device,
            ).unsqueeze(1)
            interp = (1 - seg) * gate_positions[i] + seg * gate_positions[i + 1]
            points.append(interp)
        trajectory = torch.cat(points, dim=0)
    elif method == "spline":
        # Use gate normals to set endpoint tangents
        n_ctps = gate_positions.shape[0]
        ctps = gate_positions.clone()

        # Adjust first and last control points to match normals
        # (Hermite-style: set first derivative at endpoints)
        if n_ctps >= 4:
            ctps[1] = ctps[0] + tangent_scale * gate_normals[0]
            ctps[-2] = ctps[-1] - tangent_scale * gate_normals[-1]

        knots = get_knots(n_ctps=n_ctps, k=k, device=gate_positions.device)
        t_vals = torch.linspace(
            knots[k], knots[-k - 1], num_points, device=gate_positions.device
        )
        trajectory = splev_torch(t_vals, knots, ctps, k)
    else:
        raise ValueError(f"Unknown method: {method}")
    return trajectory


def generate_trajectory_from_gate_poses(
    gate_positions,
    gate_orientations,
    method="spline",
    num_points=200,
    tangent_scale=1.0,
    k=3,
):
    """
    Helper to generate a global trajectory from gate positions and orientations (quaternions).
    Args:
        gate_positions: (N_gates, 3) tensor of gate center positions.
        gate_orientations: (N_gates, 4) tensor of gate orientations (quaternions, wxyz).
        method: "linear" or "spline".
        num_points: Number of points in the trajectory.
        tangent_scale: Scaling factor for tangent magnitude at endpoints.
        k: Degree of B-spline (ignored if method is "linear").
    Returns:
        trajectory: (num_points, 3) tensor of waypoints.
    """
    # Compute gate normals (x-axis in world frame for each gate)
    from omni_drones.utils.math import quat_rotate

    gate_local_x = torch.tensor(
        [1.0, 0.0, 0.0], device=gate_positions.device, dtype=gate_positions.dtype
    )
    gate_normals = []
    for q in gate_orientations:
        normal = quat_rotate(q.unsqueeze(0), gate_local_x.unsqueeze(0)).squeeze(0)
        gate_normals.append(normal)
    gate_normals = torch.stack(gate_normals)
    return compute_global_trajectory(
        gate_positions,
        gate_normals,
        method=method,
        num_points=num_points,
        tangent_scale=tangent_scale,
        k=k,
    )
