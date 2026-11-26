import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
import socket
import threading
import math
import time

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab.sim as sim_utils
from go2_common import hw_to_isaac, DEFAULT_POLICY, policy_layout, make_env_cfg, make_env, load_policy, set_command

# Offsets are reported in Isaac Lab joint order (4 hips, 4 thighs, 4 calves)
JOINTS = ['FL_hip','FR_hip','RL_hip','RR_hip','FL_thigh','FR_thigh','RL_thigh','RR_thigh',
          'FL_calf','FR_calf','RL_calf','RR_calf']

def main():
    layout = policy_layout(DEFAULT_POLICY)
    env_cfg = make_env_cfg(layout)

    env = make_env(env_cfg)
    device = env.unwrapped.device
    policy = load_policy(DEFAULT_POLICY, device)

    # UDP to receive real robot state
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.bind(('127.0.0.1', 5005))
    udp_sock.setblocking(False)
    real_state = [None]

    obs, _ = env.reset()
    set_command(env, obs, layout, 0.0, 0.0, 0.0)

    datasets = []

    def collect_data():
        print("\nCalibration collector ready!")
        print("Keep robot standing. Press Enter to collect a reading (5 total).")
        print("Run go2_real_bridge.py in another terminal first!")
        print("="*50)
        for i in range(5):
            input(f"\nPress Enter for reading {i+1}/5...")
            sim_jp = env.unwrapped.scene["robot"].data.joint_pos[0].cpu().numpy().copy()
            # real joints arrive in hardware order; compare in Isaac order
            real_jp = hw_to_isaac(real_state[0][:12].copy()) if real_state[0] is not None else np.zeros(12)
            datasets.append((sim_jp, real_jp))
            print(f"Sim:  {sim_jp.round(4).tolist()}")
            print(f"Real: {real_jp.round(4).tolist()}")

        sim_avg = np.mean([d[0] for d in datasets], axis=0)
        real_avg = np.mean([d[1] for d in datasets], axis=0)
        offsets = real_avg - sim_avg

        print("\n" + "="*50)
        print("CALIBRATION RESULTS:")
        print(f"\nSim avg:   {sim_avg.round(4).tolist()}")
        print(f"Real avg:  {real_avg.round(4).tolist()}")
        print(f"\nOffsets (real - sim):")
        for j in range(12):
            print(f"  [{j}] {JOINTS[j]:12s}: {offsets[j]:+.4f} rad ({math.degrees(offsets[j]):+.2f} deg)")
        print(f"\nOffset array: {offsets.round(4).tolist()}")
        np.save('/tmp/go2_calibration_offsets.npy', offsets)
        print("\nSaved to /tmp/go2_calibration_offsets.npy")

    threading.Thread(target=collect_data, daemon=True).start()

    count = 0
    while simulation_app.is_running():
        # Read UDP
        try:
            data, _ = udp_sock.recvfrom(30 * 4)
            real_state[0] = np.frombuffer(data, dtype=np.float32).copy()
        except BlockingIOError:
            pass

        with torch.inference_mode():
            action = policy(obs)
            obs, _, _, _ = env.step(action)
        set_command(env, obs, layout, 0.0, 0.0, 0.0)
        count += 1

if __name__ == "__main__":
    main()
    simulation_app.close()
