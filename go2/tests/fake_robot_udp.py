"""Pretend to be go2_real_bridge_debug.py: stream 36-float robot-state packets to UDP 5005.

Joints (hardware order) sway gently around the measured standing pose, the IMU quaternion
is level, and XY drifts slowly, so real2sim scripts can be exercised without the robot.

    python tests/fake_robot_udp.py [--seconds 600] [--short]   # --short: 30-float packets
"""
import argparse
import math
import socket
import time

import numpy as np

STAND_HW = np.array([-0.0137, 0.7486, -1.5539, -0.0592, 0.8886, -1.5863,
                     0.1728, 0.7593, -1.4299, -0.0432, 0.7585, -1.5323], dtype=np.float32)

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=600)
parser.add_argument("--port", type=int, default=5005)
parser.add_argument("--short", action="store_true", help="send 30-float packets like go2_real_bridge.py")
args = parser.parse_args()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
t0 = time.time()
n = 0
while time.time() - t0 < args.seconds:
    t = time.time() - t0
    s = np.zeros(36, dtype=np.float32)
    s[:12] = STAND_HW
    s[[1, 4, 7, 10]] += 0.1 * math.sin(2 * math.pi * 0.5 * t)  # thighs sway
    s[26] = 1.0  # quaternion w (level)
    s[30], s[31] = 0.2473 + 0.02 * t, -0.0867
    sock.sendto((s[:30] if args.short else s).tobytes(), ("127.0.0.1", args.port))
    n += 1
    time.sleep(0.005)
print(f"sent {n} packets")
