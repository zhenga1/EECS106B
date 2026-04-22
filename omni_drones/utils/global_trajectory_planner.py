import torch
from omni_drones.utils.bspline import get_knots, splev_torch


def compute_global_trajectory(
    gate_positions,
    gate_normals,
    method="catmull_rom",
    num_points=100,
    tangent_scale=1.0,
    k=3,
):
    """
    Generate a global trajectory through the gates, matching gate normals.

    Args:
        gate_positions: (N_gates, 3) tensor of gate center positions.
        gate_normals: (N_gates, 3) tensor of gate normals (desired tangents).
        method: interpolation method — see below.
        num_points: Number of points in the trajectory.
        tangent_scale: Scaling factor for tangent magnitude at endpoints
                       (only used by the legacy "spline" method).
        k: Degree of B-spline (only used by the legacy "spline" method).

    Supported methods
    -----------------
    "catmull_rom"
        Arc-length-parameterised cubic Hermite interpolating spline built with
        scipy.CubicSpline.  **Passes through every gate centre exactly.**
        Endpoint tangents are set from gate_normals so the drone enters/exits
        each terminal gate in the correct direction.

    "linear"
        Piecewise-linear interpolation.  Correct (passes through all gates)
        but produces sharp corners at each gate.
        
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

    elif method == "catmull_rom":
        # Interpolating cubic Hermite spline via scipy.CubicSpline.
        # Unlike the legacy "spline" method, this ALWAYS passes through every
        # gate centre.  Arc-length parameterisation gives uniform point
        # density along the path.
        import numpy as np
        from scipy.interpolate import CubicSpline as _CubicSpline

        gate_np = gate_positions.detach().cpu().numpy().astype(np.float64)  # (N, 3)
        normals_np = gate_normals.detach().cpu().numpy().astype(np.float64)  # (N, 3)

        # Arc-length parameter: t[i] = cumulative Euclidean distance to gate i.
        diffs = np.diff(gate_np, axis=0)                        # (N-1, 3)
        seg_lens = np.linalg.norm(diffs, axis=1)                # (N-1,)
        t_params = np.concatenate([[0.0], np.cumsum(seg_lens)]) # (N,)

        # Boundary conditions: match the gate's x-axis (normal) at each end.
        # scipy encodes BC as (derivative_order, value_at_endpoint).
        bc = [(1, normals_np[0]), (1, normals_np[-1])]

        cs = _CubicSpline(t_params, gate_np, bc_type=bc)

        t_samples = np.linspace(t_params[0], t_params[-1], int(num_points))
        traj_np = cs(t_samples).astype(np.float32)              # (num_points, 3)
        trajectory = torch.from_numpy(traj_np).to(gate_positions.device)

    else:
        raise ValueError(f"Unknown trajectory method '{method}'. "
                         f"Choose from: 'catmull_rom', 'linear'.")
    return trajectory


def generate_trajectory_from_gate_poses(
    gate_positions,
    gate_orientations,
    method="catmull_rom",
    num_points=200,
    tangent_scale=1.0,
    k=3,
):
    """
    Helper to generate a global trajectory from gate positions and orientations (quaternions).
    Args:
        gate_positions: (N_gates, 3) tensor of gate center positions.
        gate_orientations: (N_gates, 4) tensor of gate orientations (quaternions, wxyz).
        method: "linear" or "catmull_rom".
        num_points: Number of points in the trajectory.
        tangent_scale: Scaling factor for tangent magnitude at endpoints.
        k: Degree of B-spline (ignored if method is "linear").
    Returns:
        trajectory: (num_points, 3) tensor of waypoints.
    """
    # Compute gate normals (x-axis in world frame for each gate)
    from omni_drones.utils.torch import quat_rotate

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
