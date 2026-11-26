import numpy as np
import socket
import sys
import os
import threading
from go2_common import add_unitree_sdk_path, ROBOT_INTERFACE
add_unitree_sdk_path()

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
ISAAC_IP = '127.0.0.1'
ISAAC_PORT = 5005

latest = [None]
lock = threading.Lock()

def handler(msg):
    state = np.zeros(30, dtype=np.float32)
    for i in range(12):
        state[i] = msg.motor_state[i].q
        state[12+i] = msg.motor_state[i].dq
    state[24] = msg.imu_state.gyroscope[0]
    state[25] = msg.imu_state.gyroscope[1]
    state[26] = msg.imu_state.quaternion[0]
    state[27] = msg.imu_state.quaternion[1]
    state[28] = msg.imu_state.quaternion[2]
    state[29] = msg.imu_state.quaternion[3]
    sock.sendto(state.tobytes(), (ISAAC_IP, ISAAC_PORT))
    with lock:
        latest[0] = state

ChannelFactoryInitialize(0, ROBOT_INTERFACE)
sub = ChannelSubscriber('rt/lowstate', LowState_)
sub.Init(handler, 1)

print(f"Bridge running at maximum rate → {ISAAC_IP}:{ISAAC_PORT}")
import time
count = 0
while True:
    time.sleep(1)
    count += 1
    with lock:
        if latest[0] is not None and count % 5 == 0:
            print(f"Joint0={latest[0][0]:.3f} quat_w={latest[0][26]:.3f}")
