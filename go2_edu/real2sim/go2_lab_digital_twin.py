import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Go2 Lab Digital Twin - final layout, real2sim mirroring, LiDAR.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from go2_common import hw_to_isaac
import numpy as np
import socket
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from go2_common import inputs_to_device
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

FT = 0.3048
AREA_X = 15 * FT
AREA_Y = 10 * FT

# Calibration (proven working from go2_digital_twin_walk_debug.py)
SIM_STANDING = torch.tensor([0.1096, 0.0253, 0.0522, 0.0006, 0.9423, 0.6477,
                              1.1675, 1.1528, -1.6193, -1.7285, -1.4102, -1.5936])
REAL_STANDING = torch.tensor([-0.0137, 0.7486, -1.5539, -0.0592, 0.8886, -1.5863,
                               0.1728, 0.7593, -1.4299, -0.0432, 0.7585, -1.5323])

# Position calibration (from earlier real2sim work)
REAL_ORIGIN_X = 0.2473
REAL_ORIGIN_Y = -0.0867

# Robot spawn — parallel to the big red cylinder at (4.0097, 2.7163, 0.2324)
# Offset along X by cylinder radius (0.25) + clearance, same Y, facing +X
BIG_CYL_POS = (4.0097, 2.7163)
# Spawn on the opposite (west) corner along the short 10ft side, near the boundary
ROBOT_SPAWN_X = 0.5
ROBOT_SPAWN_Y = BIG_CYL_POS[1]
ROBOT_SPAWN_Z = 0.35


@configclass
class Go2LabDigitalTwinSceneCfg(InteractiveSceneCfg):
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    boundary_n = RigidObjectCfg(
        prim_path="/World/Boundary/N",
        spawn=sim_utils.CuboidCfg(size=(AREA_X, 0.05, 0.005),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(AREA_X/2, AREA_Y, 0.003)),
    )
    boundary_s = RigidObjectCfg(
        prim_path="/World/Boundary/S",
        spawn=sim_utils.CuboidCfg(size=(AREA_X, 0.05, 0.005),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(AREA_X/2, 0.0, 0.003)),
    )
    boundary_e = RigidObjectCfg(
        prim_path="/World/Boundary/E",
        spawn=sim_utils.CuboidCfg(size=(0.05, AREA_Y, 0.005),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(AREA_X, AREA_Y/2, 0.003)),
    )
    boundary_w = RigidObjectCfg(
        prim_path="/World/Boundary/W",
        spawn=sim_utils.CuboidCfg(size=(0.05, AREA_Y, 0.005),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, AREA_Y/2, 0.003)),
    )

    table = RigidObjectCfg(
        prim_path="/World/Objects/Table",
        spawn=sim_utils.CuboidCfg(size=(1.2, 0.74, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.0), metallic=0.3)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.9500, 1.5370, -0.0520)),
    )

    cyl_0 = RigidObjectCfg(
        prim_path="/World/Objects/Cylinder",
        spawn=sim_utils.CylinderCfg(radius=0.05, height=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.15))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.9075, 1.5201, 0.1058)),
    )
    cyl_1 = RigidObjectCfg(
        prim_path="/World/Objects/Cylinder_01",
        spawn=sim_utils.CylinderCfg(radius=0.05, height=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.15))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6051, 0.6078, 0.1058)),
    )
    cyl_2 = RigidObjectCfg(
        prim_path="/World/Objects/Cylinder_02",
        spawn=sim_utils.CylinderCfg(radius=0.05, height=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.15))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.9075, 2.1519, 0.1058)),
    )
    cyl_3 = RigidObjectCfg(
        prim_path="/World/Objects/Cylinder_03",
        spawn=sim_utils.CylinderCfg(radius=0.05, height=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.4, 1.0))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0210, 0.9255, 0.1058)),
    )

    big_cylinder = RigidObjectCfg(
        prim_path="/World/Objects/BigCylinder",
        spawn=sim_utils.CylinderCfg(radius=0.25, height=0.5,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.2))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.0097, 2.7163, 0.2324)),
    )

    robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=True,
        linear_damping=10.0,
        angular_damping=10.0,
    )
    robot.spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15))

    ray_caster = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=1/10,
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.3)),
        mesh_prim_paths=[
            "/World/Boundary/N", "/World/Boundary/S",
            "/World/Boundary/E", "/World/Boundary/W",
            "/World/Objects/Table",
            "/World/Objects/Cylinder", "/World/Objects/Cylinder_01",
            "/World/Objects/Cylinder_02", "/World/Objects/Cylinder_03",
            "/World/Objects/BigCylinder",
        ],
        pattern_cfg=patterns.LidarPatternCfg(
            channels=16,
            vertical_fov_range=[-15.0, 15.0],
            horizontal_fov_range=[-180.0, 180.0],
            horizontal_res=1.0,
        ),
        debug_vis=False,
    )


def spawn_checker_floor():
    n_x, n_y = 15, 10
    for ix in range(n_x):
        for iy in range(n_y):
            is_light = (ix + iy) % 2 == 0
            color = (1.0, 1.0, 1.0) if is_light else (0.02, 0.02, 0.02)
            prim_path = f"/World/Floor/Tile_{ix}_{iy}"
            cfg = sim_utils.CuboidCfg(
                size=(FT, FT, 0.01),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            cfg.func(prim_path, cfg, translation=(ix*FT + FT/2, iy*FT + FT/2, -0.005))


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[8.0, 5.0, 6.0], target=[7.5, 5.0, 0.0])
    scene_cfg = Go2LabDigitalTwinSceneCfg(num_envs=args_cli.num_envs, env_spacing=20.0)
    scene = inputs_to_device(InteractiveScene(scene_cfg))
    sim.reset()

    spawn_checker_floor()

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
    udp_sock.bind(('127.0.0.1', 5005))
    udp_sock.setblocking(False)
    print("[INFO]: Listening on UDP port 5005...")
    print("[INFO]: Run ~/go2_real_bridge_debug.py in another terminal!")

    # Initial spawn: parallel to the big red cylinder
    root_state = scene["robot"].data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    root_state[0, 0] = ROBOT_SPAWN_X
    root_state[0, 1] = ROBOT_SPAWN_Y
    root_state[0, 2] = ROBOT_SPAWN_Z
    scene["robot"].write_root_pose_to_sim(root_state[:, :7])
    scene["robot"].write_root_velocity_to_sim(torch.zeros(1, 6))
    scene["robot"].write_joint_state_to_sim(SIM_STANDING.unsqueeze(0), torch.zeros(1, 12))
    scene.reset()

    print(f"[INFO]: Robot spawned at ({ROBOT_SPAWN_X:.3f}, {ROBOT_SPAWN_Y:.3f}, {ROBOT_SPAWN_Z:.3f}) — parallel to big red cylinder")

    state = None
    all_hits = []
    count = 0
    origin_pos = [None, None]  # captured on first real robot packet

    while simulation_app.is_running():
        try:
            while True:
                data, _ = udp_sock.recvfrom(36 * 4)
                state = np.frombuffer(data, dtype=np.float32).copy()
        except BlockingIOError:
            pass

        if state is not None:
            real_joints = torch.tensor(state[:12], dtype=torch.float32)
            sim_joints = SIM_STANDING + hw_to_isaac(real_joints - REAL_STANDING)  # real data is hardware order
            scene["robot"].write_joint_state_to_sim(
                sim_joints.unsqueeze(0), torch.zeros(1, 12))

            root_state = scene["robot"].data.root_state_w.clone()
            # XY from rt/sportmodestate (36-float packets from go2_real_bridge_debug.py);
            # 30-float packets from go2_real_bridge.py carry no position, so hold the spawn XY.
            has_pos = len(state) >= 32
            real_pos_x = float(state[30]) if has_pos else 0.0
            real_pos_y = float(state[31]) if has_pos else 0.0
            if origin_pos[0] is None:
                origin_pos[0] = real_pos_x
                origin_pos[1] = real_pos_y
            root_state[0, 0] = ROBOT_SPAWN_X + (real_pos_x - origin_pos[0])
            root_state[0, 1] = ROBOT_SPAWN_Y + (real_pos_y - origin_pos[1])
            root_state[0, 2] = ROBOT_SPAWN_Z
            root_state[0, 3] = float(state[26])  # w
            root_state[0, 4] = float(state[27])  # x
            root_state[0, 5] = float(state[28])  # y
            root_state[0, 6] = float(state[29])  # z
            root_state[0, 7:13] = 0.0
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:13])

        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim.get_physics_dt())

        if count % 20 == 0:
            try:
                hits = scene["ray_caster"].data.ray_hits_w[0].cpu().numpy()
                valid_mask = hits[:, 2] > -1e5
                valid_hits = hits[valid_mask]
                if len(valid_hits) > 0:
                    all_hits.append(valid_hits[:, :2])
                robot_pos = scene["robot"].data.root_pos_w[0].cpu().numpy()
                if len(all_hits) > 0:
                    np.save('/tmp/go2_lidar_map.npy', np.vstack(all_hits[-100:]))
                    np.save('/tmp/go2_robot_pos.npy', robot_pos[:2])
                if count % 500 == 0:
                    print(f"[{count}] UDP:{'OK' if state is not None else 'waiting'} | LiDAR:{valid_mask.sum()}")
            except Exception as e:
                pass


if __name__ == "__main__":
    main()
    simulation_app.close()
