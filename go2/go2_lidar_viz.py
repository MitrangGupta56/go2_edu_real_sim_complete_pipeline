import numpy as np
import matplotlib.pyplot as plt
import os

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
plt.ion()

print("Go2 LiDAR Map Visualizer")
print("Waiting for data from Isaac Sim...")

while True:
    try:
        if os.path.exists('/tmp/go2_lidar_map.npy') and os.path.exists('/tmp/go2_robot_pos.npy'):
            points = np.load('/tmp/go2_lidar_map.npy')
            robot = np.load('/tmp/go2_robot_pos.npy')

            ax.clear()
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_aspect('equal')
            ax.set_facecolor('black')
            fig.patch.set_facecolor('#1a1a2e')

            ax.scatter(points[:, 0], points[:, 1], s=1, c='lime', alpha=0.5)
            ax.plot(robot[0], robot[1], 'ro', markersize=10)
            circle = plt.Circle((robot[0], robot[1]), 0.3, color='red', fill=False, linewidth=2)
            ax.add_patch(circle)

            ax.set_title('Go2 LiDAR Map', color='white')
            ax.grid(True, color='gray', alpha=0.3)
            ax.tick_params(colors='white')

        plt.pause(0.1)
    except KeyboardInterrupt:
        break
    except Exception as e:
        plt.pause(0.3)
