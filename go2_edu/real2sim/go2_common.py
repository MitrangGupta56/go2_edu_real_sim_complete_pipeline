"""Shared setup for the Go2 scripts: paths, policy loading and Isaac Lab env config.

Why this exists: the trained policies expect a specific observation vector, and the old
scripts relied on a hand-patched Isaac Lab source tree to get it. Here the observation
layout is chosen in code from the policy's input size, so stock Isaac Lab works.

  48 inputs  go2_policy/policy.pt        lin_vel, ang_vel, gravity, cmd, q, dq, last_action
  45 inputs  go2_policy_rough/policy.pt  same without lin_vel (deploy-friendly)
  235 inputs stock rough task            48 + 187-point height scan

Environment variables (all optional):
  GO2_POLICY            policy .pt to load (default: go2_policy/policy.pt)
  ROBOT_INTERFACE       network interface wired to the Go2 (default: enp6s0)
  UNITREE_SDK2_PYTHON   folder containing unitree_sdk2py (default: this folder)
"""
import os
import sys

GO2_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_POLICY = os.environ.get("GO2_POLICY", os.path.join(GO2_DIR, "go2_policy", "policy.pt"))
ROUGH_POLICY = os.path.join(GO2_DIR, "go2_policy_rough", "policy.pt")
ROBOT_INTERFACE = os.environ.get("ROBOT_INTERFACE", "enp6s0")

# ── Joint order ─────────────────────────────────────────────────────────────
# Hardware / SDK order (per leg): FL_hip FL_thigh FL_calf FR_hip ... RR_calf
# Isaac Lab order (per type):     FL_hip FR_hip RL_hip RR_hip FL_thigh ... RR_calf
ISAAC_TO_HW = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]   # hw = isaac[ISAAC_TO_HW]
HW_TO_ISAAC = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]   # isaac = hw[HW_TO_ISAAC]


def hw_to_isaac(x):
    return x[..., HW_TO_ISAAC]


def isaac_to_hw(x):
    return x[..., ISAAC_TO_HW]


def get_keyboard():
    """(input_interface, keyboard) for Kit keyboard events, or (None, None) when there is no
    app window (e.g. --headless), so scripts still run without keyboard control."""
    try:
        import carb
        import omni.appwindow
        window = omni.appwindow.get_default_app_window()
        if window is None:
            raise RuntimeError("no app window")
        return carb.input.acquire_input_interface(), window.get_keyboard()
    except (ImportError, RuntimeError) as e:
        print(f"[go2] Keyboard control unavailable ({e}); run with the GUI (--viz kit) to drive.")
        return None, None


_WRITE_METHODS = (
    "write_root_pose_to_sim", "write_root_velocity_to_sim", "write_root_state_to_sim",
    "write_joint_state_to_sim", "set_joint_position_target", "set_joint_velocity_target",
    "set_joint_effort_target",
)


def inputs_to_device(scene):
    """Isaac Lab 3 writes go through warp kernels that reject CPU tensors on a GPU sim
    (2.x copied them silently). Wrap the scene assets' write methods so tensors such as
    torch.zeros(1, 12) are moved to the asset's device automatically."""
    import torch

    assets = list(getattr(scene, "articulations", {}).values()) + list(getattr(scene, "rigid_objects", {}).values())
    for asset in assets:
        for name in _WRITE_METHODS:
            orig = getattr(asset, name, None)
            if orig is None:
                continue

            def wrapped(*args, _orig=orig, _asset=asset, **kwargs):
                device = _asset.device  # only valid once the sim has initialised the asset
                args = [a.to(device) if isinstance(a, torch.Tensor) else a for a in args]
                kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
                return _orig(*args, **kwargs)

            setattr(asset, name, wrapped)
    return scene


def as_torch(x):
    """Isaac Lab 3 returns warp-backed ProxyArrays from asset/sensor data; get the torch view."""
    return x.torch if hasattr(x, "torch") else x


def add_unitree_sdk_path():
    """Make `import unitree_sdk2py` work (bundled copy, or UNITREE_SDK2_PYTHON)."""
    sdk = os.environ.get("UNITREE_SDK2_PYTHON", GO2_DIR)
    if sdk not in sys.path:
        sys.path.insert(0, sdk)
    # cyclonedds (Python) was built against ROS Jazzy's CycloneDDS 0.10.5; it finds libddsc
    # through CYCLONEDDS_HOME, which must be set before the first `import cyclonedds`.
    shim = os.path.join(GO2_DIR, "third_party", "cyclonedds_ros_jazzy")
    if "CYCLONEDDS_HOME" not in os.environ and os.path.isdir(shim):
        os.environ["CYCLONEDDS_HOME"] = shim
        _preload_ros_iceoryx(os.path.join(shim, "lib"))


def _preload_ros_iceoryx(libdir):
    """ROS's libddsc needs iceoryx libs from the same folder, which isn't on the loader
    path unless ROS is sourced. Load them globally first (retrying until deps resolve)."""
    import ctypes
    import glob
    pending = sorted(glob.glob(os.path.join(libdir, "libiceoryx_*.so")))
    for _ in range(len(pending)):
        failed = []
        for lib in pending:
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(lib)
        if not failed or len(failed) == len(pending):
            break
        pending = failed


# ── Policy ──────────────────────────────────────────────────────────────────

def policy_input_dim(path):
    import torch
    m = torch.jit.load(path, map_location="cpu")
    for name, p in m.named_parameters():
        if name.endswith("weight") and p.dim() == 2:
            return int(p.shape[1])
    raise ValueError(f"Could not find an input layer in {path}")


def policy_layout(path):
    dim = policy_input_dim(path)
    if dim == 48:
        return {"dim": 48, "cmd_idx": 9, "lin_vel": True, "height_scan": False}
    if dim == 45:
        return {"dim": 45, "cmd_idx": 6, "lin_vel": False, "height_scan": False}
    if dim == 235:
        return {"dim": 235, "cmd_idx": 9, "lin_vel": True, "height_scan": True}
    raise ValueError(f"{path}: unsupported policy input size {dim} (expected 45, 48 or 235)")


def load_policy(path, device):
    import torch
    model = torch.jit.load(path, map_location=device).eval()

    def policy(obs):
        with torch.no_grad():
            x = obs["policy"] if not isinstance(obs, torch.Tensor) else obs
            return model(x)
    return policy


# ── Env config ──────────────────────────────────────────────────────────────

def resolve_presets(cfg):
    """Isaac Lab 3 configs hold Hydra presets; pick their defaults (no-op on 2.x)."""
    try:
        from isaaclab_tasks.utils.hydra import resolve_presets as _resolve
    except ImportError:
        return cfg
    return _resolve(cfg)


def make_env_cfg(layout):
    """Go2 velocity-task PLAY config whose policy observations match `layout`."""
    if layout["height_scan"]:
        from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
            UnitreeGo2RoughEnvCfg_PLAY as Cfg)
    else:
        from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
            UnitreeGo2FlatEnvCfg_PLAY as Cfg)
    cfg = resolve_presets(Cfg())
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 1_000_000
    cfg.curriculum = None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    if layout["height_scan"]:
        cfg.scene.height_scanner.mesh_prim_paths = ["/World/ground"]
    if not layout["lin_vel"]:
        cfg.observations.policy.base_lin_vel = None
    cfg.observations.policy.enable_corruption = False
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
    cfg.commands.base_velocity.ranges.heading = (-1.0, 1.0)
    return cfg


def make_env(cfg):
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    return RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=cfg))


def make_policy_env(policy_path=DEFAULT_POLICY, customize=None):
    """Build env + policy. `customize(cfg)` can add walls, LiDAR, etc. before creation."""
    layout = policy_layout(policy_path)
    cfg = make_env_cfg(layout)
    if customize is not None:
        customize(cfg)
    env = make_env(cfg)
    actual = env.get_observations()["policy"].shape[-1]
    if actual != layout["dim"]:
        raise RuntimeError(f"Env gives {actual} observations but policy expects {layout['dim']}")
    return env, load_policy(policy_path, env.unwrapped.device), layout


def set_command(env, obs, layout, vx, vy, wz):
    """Write a velocity command into the policy observation (and the command manager)."""
    import torch
    i = layout["cmd_idx"]
    with torch.inference_mode():
        obs["policy"][:, i:i + 3] = torch.tensor([vx, vy, wz], device=obs["policy"].device)
        term = env.unwrapped.command_manager.get_term("base_velocity")
        term.vel_command_b[:, 0] = vx
        term.vel_command_b[:, 1] = vy
        term.vel_command_b[:, 2] = wz
