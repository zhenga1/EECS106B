# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.distributions as D
import einops
import torch.nn.functional as F

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import ArticulationView, RigidPrimView
from omni_drones.utils.torch import euler_to_quaternion, quat_axis
from omni_drones.sensors.camera import Camera
from omni_drones.sensors.config import PinholeCameraCfg

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import Unbounded, Composite, DiscreteTensorSpec

from isaacsim.core.utils.viewports import set_camera_view


class DepthNav(IsaacEnv):
    r"""
    This is a single-agent task where the agent is required to navigate a randomly
    generated cluttered environment. The agent needs to fly at a commanded speed
    along the positive direction while avoiding collisions with obstacles.

    The agent utilizes depth cameras to perceive its surroundings. The depth camera
    provides a depth image that shows the distance to objects in the scene.

    ## Observation

    The observation is given by a `Composite` containing the following values:

    - `"state"` (16 + `num_rotors`): The basic information of the drone
      (except its position), containing its rotation (in quaternion), velocities
      (linear and angular), heading and up vectors, and the current throttle.
    - `"depth"` (H, W) : The depth image from the camera, downsampled and flattened.
      The size is decided by the camera resolution and downsampling factor.

    ## Reward

    - `vel`: Reward computed from the position error to the target position.
    - `up`: Reward computed from the uprightness of the drone to discourage large tilting.
    - `survive`: Reward of a constant value to encourage collision avoidance.
    - `safety`: Reward computed from the depth image to encourage maintaining safe distances.

    The total reward is computed as follows:

    ```{math}
        r = r_\text{vel} + r_\text{up} + r_\text{survive} + r_\text{safety}
    ```

    ## Episode End

    The episode ends when the drone misbehaves, e.g., when the drone collides
    with the ground or obstacles, or when the drone flies out of the boundary:

    ```{math}
        d_\text{ground} < 0.2 \text{ or } d_\text{ground} > 4.0 \text{ or } v_\text{drone} > 2.5
    ```

    or when the episode reaches the maximum length.

    ## Config

    | Parameter         | Type  | Default   | Description                                                                                                                                                                                                                             |
    | ----------------- | ----- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `drone_model`     | str   | "firefly" | Specifies the model of the drone being used in the environment.                                                                                                                                                                         |
    | `depth_range`     | float | 4.0       | Specifies the maximum range of the depth camera.                                                                                                                                                                                       |
    | `camera_resolution` | tuple | (160, 120) | Specifies the resolution of the depth camera.                                                                                                                                                                                          |
    | `depth_downsample` | int   | 4         | Downsampling factor for the depth image to reduce observation size.                                                                                                                                                                     |
    | `time_encoding`   | bool  | True      | Indicates whether to include time encoding in the observation space. If set to True, a 4-dimensional vector encoding the current progress of the episode is included in the observation. If set to False, this feature is not included. |
    """
    def __init__(self, cfg, headless):
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.time_encoding = cfg.task.time_encoding
        self.randomization = cfg.task.get("randomization", {})
        self.has_payload = "payload" in self.randomization.keys()
        self.depth_range = cfg.task.depth_range
        self.camera_resolution = tuple(cfg.task.camera_resolution)
        self.depth_downsample = cfg.task.depth_downsample

        # Calculate downsampled depth image dimensions before super().__init__()
        # because _set_specs() is called during super().__init__() and needs this
        depth_h = self.camera_resolution[1] // self.depth_downsample
        depth_w = self.camera_resolution[0] // self.depth_downsample
        self.depth_dim = depth_h * depth_w

        super().__init__(cfg, headless)

        # Initialize camera after scene is cloned and reset
        # Get drone name from the drone that was created in _design_scene
        camera_prim_paths_expr = f"/World/envs/env_.*/{self.drone.name}_0/base_link/DepthCamera"
        self.depth_camera.initialize(camera_prim_paths_expr)

        self.drone.initialize()
        if "drone" in self.randomization:
            self.drone.setup_randomization(self.randomization["drone"])

        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        )

        with torch.device(self.device):
            self.target_pos = torch.zeros(self.num_envs, 1, 3)
            self.target_pos[:, 0, 0] = torch.linspace(-0.5, 0.5, self.num_envs) * 32.
            self.target_pos[:, 0, 1] = 24.
            self.target_pos[:, 0, 2] = 2.

        self.alpha = 0.8

        # Store depth dimensions for later use
        self.depth_h = self.camera_resolution[1] // self.depth_downsample
        self.depth_w = self.camera_resolution[0] // self.depth_downsample

    def _design_scene(self):
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )

        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.)])[0]

        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.terrains import (
            TerrainImporterCfg,
            TerrainImporter,
            TerrainGeneratorCfg,
            HfDiscreteObstaclesTerrainCfg,
        )

        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),
        )
        rot = euler_to_quaternion(torch.tensor([0., 0.1, 0.1]))
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos, rot)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)

        terrain_cfg = TerrainImporterCfg(
            num_envs=self.num_envs,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=0,
                size=(8.0, 8.0),
                border_width=20.0,
                num_rows=5,
                num_cols=5,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                use_cache=False,
                sub_terrains={
                    "obstacles": HfDiscreteObstaclesTerrainCfg(
                        size=(8.0, 8.0),
                        horizontal_scale=0.1,
                        vertical_scale=0.1,
                        border_width=0.0,
                        num_obstacles=40,
                        obstacle_height_mode="choice",
                        obstacle_width_range=(0.4, 0.8),
                        obstacle_height_range=(3.0, 4.0),
                        platform_width=1.5,
                    )
                },
            ),
            max_init_terrain_level=5,
            collision_group=-1,
            debug_vis=False,
        )
        terrain: TerrainImporter = terrain_cfg.class_type(terrain_cfg)

        # Create depth camera configuration
        camera_cfg = PinholeCameraCfg(
            sensor_tick=0,
            resolution=self.camera_resolution,
            data_types=["distance_to_camera"],
            usd_params=PinholeCameraCfg.UsdCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, self.depth_range),
            ),
        )
        
        # Spawn camera on template drone (will be cloned for all envs)
        drone_name = self.drone.name
        camera_prim_path = f"/World/envs/env_0/{drone_name}_0/base_link/DepthCamera"
        
        self.depth_camera = Camera(camera_cfg)
        self.depth_camera.spawn(
            [camera_prim_path],
            translations=[(0.0, 0.0, 0.0)],
            targets=[(1.0, 0.0, 0.0)]
        )
        
        return ["/World/ground"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        obs_dim = drone_state_dim + self.depth_dim

        self.observation_spec = Composite({
            "agents": Composite({
                "observation": Unbounded((1, obs_dim), device=self.device),
                "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device)
            })
        }).expand(self.num_envs).to(self.device)
        self.action_spec = Composite({
            "agents": Composite({
                "action": self.drone.action_spec.unsqueeze(0),
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = Composite({
            "agents": Composite({
                "reward": Unbounded((1,1))
            })
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics")
        )

        stats_spec = Composite({
            "return": Unbounded(1),
            "episode_len": Unbounded(1),
            "action_smoothness": Unbounded(1),
            "safety": Unbounded(1)
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)

        pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
        pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 32.
        pos[:, 0, 1] = -24.
        pos[:, 0, 2] = 2.

        rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(
            pos, rot, env_ids
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)

        self.stats[env_ids] = 0.

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)

    def _post_sim_step(self, tensordict: TensorDictBase):
        # Camera updates are handled automatically by the simulation
        pass

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state(env_frame=False)
        # relative position and heading
        self.rpos = self.target_pos - self.drone_state[..., :3]

        # Get depth images from camera
        images = self.depth_camera.get_images()  # (num_envs, ...)
        
        # Extract depth data (distance_to_camera)
        depth_images = images["distance_to_camera"]  # (num_envs, 1, H, W) or (num_envs, H, W)
        
        # Handle different tensor shapes
        if depth_images.dim() == 3:
            depth_images = depth_images.unsqueeze(1)  # Add channel dimension
        
        # Convert to positive depth values and clamp to range
        # distance_to_camera gives negative values, so we negate them
        depth_images = -torch.nan_to_num(depth_images, nan=self.depth_range)
        depth_images = depth_images.clamp(0.0, self.depth_range)
        
        # Downsample depth image
        if self.depth_downsample > 1:
            depth_images = F.avg_pool2d(
                depth_images, 
                kernel_size=self.depth_downsample, 
                stride=self.depth_downsample
            )
        
        # Flatten depth image: (num_envs, 1, H, W) -> (num_envs, 1, H*W)
        depth_flat = depth_images.flatten(start_dim=2)  # (num_envs, 1, depth_dim)
        
        # Normalize depth to [0, 1] range
        depth_flat = depth_flat / self.depth_range

        distance = self.rpos.norm(dim=-1, keepdim=True)
        rpos_clipped = self.rpos / distance.clamp(1e-6)
        state = torch.cat([rpos_clipped, self.drone_state[..., 3:]], dim=-1)  # (num_envs, 1, state_dim)
        obs = torch.cat([state, depth_flat], dim=-1)  # (num_envs, 1, obs_dim)

        if self._should_render(0):
            self.debug_draw.clear()
            x = self.drone.pos[0, 0]
            set_camera_view(
                eye=x.cpu() + torch.as_tensor(self.cfg.viewer.eye),
                target=x.cpu() + torch.as_tensor(self.cfg.viewer.lookat)
            )

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "intrinsics": self.drone.intrinsics,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        # pose reward
        distance = self.rpos.norm(dim=-1, keepdim=True)
        vel_direction = self.rpos / distance.clamp_min(1e-6)

        # Get depth images for safety reward
        images = self.depth_camera.get_images()
        depth_images = images["distance_to_camera"]
        if depth_images.dim() == 3:
            depth_images = depth_images.unsqueeze(1)
        depth_images = -torch.nan_to_num(depth_images, nan=self.depth_range)
        depth_images = depth_images.clamp(0.0, self.depth_range)
        
        # Downsample for reward computation
        if self.depth_downsample > 1:
            depth_images = F.avg_pool2d(
                depth_images, 
                kernel_size=self.depth_downsample, 
                stride=self.depth_downsample
            )
        
        # Safety reward: encourage maintaining safe distances
        # Reward is higher when depth values are larger (further from obstacles)
        min_depth = depth_images.view(self.num_envs, 1, -1).min(dim=-1)[0]  # (num_envs, 1)
        reward_safety = torch.log(min_depth.clamp_min(0.1) + 0.1) / torch.log(torch.tensor(self.depth_range + 0.1))
        
        reward_vel = (self.drone.vel_w[..., :3] * vel_direction).sum(-1).clip(max=2.0)
        reward_up = torch.square((self.drone.up[..., 2] + 1) / 2)

        reward = reward_vel + reward_up + 1. + reward_safety * 0.2

        # Check for collisions using depth data
        min_depth_check = depth_images.view(self.num_envs, 1, -1).min(dim=-1)[0]
        too_close = min_depth_check < 0.3

        misbehave = (
            (self.drone.pos[..., 2] < 0.2)
            | (self.drone.pos[..., 2] > 4.)
            | (self.drone.vel_w[..., :3].norm(dim=-1) > 2.5)
            | too_close.squeeze(-1)
        )
        hasnan = torch.isnan(self.drone_state).any(-1)

        terminated = misbehave | hasnan
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        self.stats["safety"].add_(reward_safety)
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1)
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
