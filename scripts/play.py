import logging
import os
import time

import hydra
import torch

from tqdm import tqdm
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from torchrl.data import CompositeSpec
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl import SyncDataCollector
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
    AttitudeController,
    RateController,
)
from omni_drones.utils.torchrl import AgentSpec, EpisodeStats
from omni_drones.learning import ALGOS

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose


FILE_PATH = os.path.dirname(__file__)

@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # If the task yaml specifies a ppo_cfg field, load that file from cfg/algo/
    # and merge it over the default algo config, allowing per-task PPO settings.
    # Only applied when actually using a PPO variant — skipped for SAC/TD3/etc.
    # so that ppo_cfg does not overwrite algo name and corrupt non-PPO runs.
    PPO_ALGOS = {"ppo", "ppo_rnn", "ppo_adapt", "mappo", "happo"}
    ppo_cfg_name = cfg.task.get("ppo_cfg", None)
    if ppo_cfg_name and cfg.algo.name.lower() in PPO_ALGOS:
        import pathlib
        algo_dir = pathlib.Path(__file__).parent.parent / "cfg" / "algo"
        ppo_cfg_path = algo_dir / ppo_cfg_name
        if not ppo_cfg_path.suffix:
            ppo_cfg_path = ppo_cfg_path.with_suffix(".yaml")
        if ppo_cfg_path.exists():
            logging.info(f"Loading task-specific PPO config: {ppo_cfg_path}")
            task_ppo_cfg = OmegaConf.load(ppo_cfg_path)
            cfg.algo = OmegaConf.merge(cfg.algo, task_ppo_cfg)
        else:
            logging.warning(f"ppo_cfg '{ppo_cfg_name}' not found at {ppo_cfg_path}, using default.")

    simulation_app = init_simulation_app(cfg)

    setproctitle(cfg.task.name)
    print(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]

    # a CompositeSpec is by deafault processed by a entity-based encoder
    # ravel it to use a MLP encoder instead
    if cfg.task.get("ravel_obs", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation"))
        transforms.append(transform)
    if cfg.task.get("ravel_obs_central", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation_central"))
        transforms.append(transform)

    # if cfg.task.get("history", False):
    #     # transforms.append(History([("info", "drone_state"), ("info", "prev_action")]))
    #     transforms.append(History([("agents", "observation")]))

    # optionally discretize the action space or use a controller
    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform == "rate":
            if not hasattr(base_env, "controller") or base_env.controller is None:
                raise ValueError(
                    "task.action_transform=rate requires cfg.task.drone_model.controller "
                    "to be set (e.g. RateController)."
                )
            transform = RateController(base_env.controller, action_key=("agents", "action"))
            transforms.append(transform)
        elif action_transform == "attitude":
            if not hasattr(base_env, "controller") or base_env.controller is None:
                raise ValueError(
                    "task.action_transform=attitude requires cfg.task.drone_model.controller "
                    "to be set (e.g. AttitudeController)."
                )
            transform = AttitudeController(base_env.controller, action_key=("agents", "action"))
            transforms.append(transform)
        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    # Render every `render_interval` seconds of simulation time (default 0.1 s).
    # The policy still runs every dt * substeps seconds — only the visual update
    # is throttled, so behaviour is identical to training.
    render_interval_sec = cfg.get("render_interval", 0.05)
    policy_dt = base_env.cfg.sim.dt * base_env.substeps
    render_every_n_steps = max(1, round(render_interval_sec / policy_dt))
    print(f"render_every_n_steps: {render_every_n_steps}")
    print(f"render_interval_sec: {render_interval_sec}")
    print(f"policy_dt: {policy_dt}")
    
    _step_counter = [0]
    _substeps = base_env.substeps

    def _render_fn(substep: int) -> bool:
        if substep == _substeps - 1:
            _step_counter[0] += 1
        return substep == _substeps - 1 and (_step_counter[0] % render_every_n_steps == 0)

    base_env.enable_render(_render_fn)

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    try:
        algo_cls = ALGOS[cfg.algo.name.lower()]
    except KeyError:
        raise NotImplementedError(f"Unknown algorithm: {cfg.algo.name}")

    # SAC and TD3 use the old AgentSpec API; PPO variants use the new
    # (observation_spec, action_spec, reward_spec) API.
    OLD_API_ALGOS = {"sac", "td3", "tdmpc"}
    if cfg.algo.name.lower() in OLD_API_ALGOS:
        agent_spec = AgentSpec(
            name="drone",
            n=1,
        )
        agent_spec._env = env
        policy = algo_cls(
            cfg.algo,
            agent_spec=agent_spec,
            device=base_env.device,
        )
    else:
        policy = algo_cls(
            cfg.algo,
            env.observation_spec,
            env.action_spec,
            env.reward_spec,
            device=base_env.device,
        )

    checkpoint_path = cfg.get("checkpoint_path", None)
    if checkpoint_path:
        policy.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded checkpoint from {checkpoint_path}")

    frames_per_batch = env.num_envs * 32

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=cfg.total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    pbar = tqdm(collector)
    env.train()
    for i, data in enumerate(pbar):
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        episode_stats.add(data.to_tensordict())

        if len(episode_stats) >= base_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))

        pbar.set_postfix({"rollout_fps": collector._fps, "frames": collector._frames})

    simulation_app.close()


if __name__ == "__main__":
    main()
