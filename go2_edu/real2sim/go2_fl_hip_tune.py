import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import carb
from go2_common import get_keyboard
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from go2_common import inputs_to_device
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

SIM_STANDING = torch.tensor([0.1096, 0.0253, 0.0522, 0.0006, 0.9423, 0.6477,
                              1.1675, 1.1528, -1.6193, -1.7285, -1.4102, -1.5936])

@configclass
class Go2TuneSceneCfg(InteractiveSceneCfg):
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.0, 2.0, 1.5], target=[0.0, 0.0, 0.3])
    scene_cfg = Go2TuneSceneCfg(num_envs=args_cli.num_envs, env_spacing=5.0)
    scene = inputs_to_device(InteractiveScene(scene_cfg))
    sim.reset()

    joints = SIM_STANDING.clone()

    fl_hip = [joints[0].item()]

    input_iface, keyboard = get_keyboard()
    def on_key(event):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            step = 0.02
            if event.input == carb.input.KeyboardInput.U:
                fl_hip[0] += step
                print(f"FL_hip = {fl_hip[0]:.4f}")
            elif event.input == carb.input.KeyboardInput.J:
                fl_hip[0] -= step
                print(f"FL_hip = {fl_hip[0]:.4f}")
        return True
    if keyboard is not None:
        input_iface.subscribe_to_keyboard_events(keyboard, on_key)

    print("Click viewport, then U = increase FL_hip, J = decrease FL_hip")
    print(f"Starting FL_hip = {fl_hip[0]:.4f}")

    root_state = scene["robot"].data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    scene["robot"].write_root_pose_to_sim(root_state[:, :7])
    scene["robot"].write_root_velocity_to_sim(torch.zeros(1, 6))

    while simulation_app.is_running():
        joints[0] = fl_hip[0]
        scene["robot"].write_joint_state_to_sim(joints.unsqueeze(0), torch.zeros(1, 12))
        scene["robot"].set_joint_position_target(joints.unsqueeze(0))

        # Keep root fixed
        root_state = scene["robot"].data.root_state_w.clone()
        root_state[0, :3] = torch.tensor([0.0, 0.0, 0.35])
        root_state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        root_state[0, 7:13] = 0.0
        scene["robot"].write_root_pose_to_sim(root_state[:, :7])
        scene["robot"].write_root_velocity_to_sim(root_state[:, 7:13])

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())


if __name__ == "__main__":
    main()
    simulation_app.close()
