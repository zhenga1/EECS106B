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
    n_gates = gate_positions.shape[0]
    if n_gates < 2:
        raise ValueError("At least 2 gate positions are required")
    if int(num_points) < 2:
        raise ValueError("num_points must be >= 2")

    if method == "linear":
        # Piecewise-linear interpolation with exactly `num_points` samples.
        n_segments = n_gates - 1
        u = torch.linspace(
            0.0,
            float(n_segments),
            int(num_points),
            device=gate_positions.device,
            dtype=gate_positions.dtype,
        )
        seg_idx = torch.floor(u).long().clamp(max=n_segments - 1)
        alpha = (u - seg_idx.to(u.dtype)).unsqueeze(1)

        p0 = gate_positions[seg_idx]
        p1 = gate_positions[seg_idx + 1]
        trajectory = (1.0 - alpha) * p0 + alpha * p1
    elif method == "spline":
        # Use gate normals to set endpoint tangents
        n_ctps = n_gates
        ctps = gate_positions.clone()
        k_eff = min(int(k), n_ctps - 1)

        # Adjust first and last control points to match normals
        # (Hermite-style: set first derivative at endpoints)
        if n_ctps >= 4:
            ctps[1] = ctps[0] + tangent_scale * gate_normals[0]
            ctps[-2] = ctps[-1] - tangent_scale * gate_normals[-1]

        knots = get_knots(n_ctps=n_ctps, k=k_eff, device=gate_positions.device)
        t_vals = torch.linspace(
            knots[k_eff],
            knots[-k_eff - 1],
            int(num_points),
            device=gate_positions.device,
            dtype=gate_positions.dtype,
        )
        trajectory = splev_torch(t_vals, knots, ctps, k_eff)
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
