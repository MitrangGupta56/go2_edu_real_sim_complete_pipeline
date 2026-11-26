"""Pretend to be the Go2 on DDS: publish rt/lowstate (and rt/sportmodestate) on an interface.

Lets the real-robot scripts be exercised without the dog, e.g. on loopback:
    python tests/fake_go2_dds.py lo &
    ROBOT_INTERFACE=lo python go2_real_bridge.py
Also prints any rt/lowcmd it receives (what a controller would send to the motors).
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from go2_common import add_unitree_sdk_path  # noqa: E402

add_unitree_sdk_path()
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.default import (  # noqa: E402
    unitree_go_msg_dds__LowState_, unitree_go_msg_dds__SportModeState_)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, SportModeState_  # noqa: E402

STAND_HW = [-0.0137, 0.7486, -1.5539, -0.0592, 0.8886, -1.5863,
            0.1728, 0.7593, -1.4299, -0.0432, 0.7585, -1.5323]

iface = sys.argv[1] if len(sys.argv) > 1 else "lo"
seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
ChannelFactoryInitialize(0, iface)
pub = ChannelPublisher("rt/lowstate", LowState_)
pub.Init()
sport_pub = ChannelPublisher("rt/sportmodestate", SportModeState_)
sport_pub.Init()

cmds = {"n": 0, "last": None}


def on_cmd(msg):
    cmds["n"] += 1
    cmds["last"] = [round(msg.motor_cmd[i].q, 3) for i in range(12)]


sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
sub.Init(on_cmd, 10)

t0 = time.time()
n = 0
while time.time() - t0 < seconds:
    t = time.time() - t0
    st = unitree_go_msg_dds__LowState_()
    for i in range(12):
        st.motor_state[i].q = STAND_HW[i] + (0.1 * math.sin(math.pi * t) if i % 3 == 1 else 0.0)
    st.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
    pub.Write(st)
    sp = unitree_go_msg_dds__SportModeState_()
    sp.position = [0.01 * t, 0.0, 0.3]
    sport_pub.Write(sp)
    n += 1
    if n % 500 == 0:
        print(f"[FAKE GO2] t={t:.0f}s published {n} lowstate | lowcmd received {cmds['n']} last q={cmds['last']}",
              flush=True)
    time.sleep(0.002)
print(f"[FAKE GO2] done: {n} lowstate sent, {cmds['n']} lowcmd received", flush=True)
