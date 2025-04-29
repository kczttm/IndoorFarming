import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the saved poses
poses = np.load("teleop_camera_poses.npy")  # or whatever filename you used

# Take the first pose as initial camera position
H_start = poses[0]
camera_position = H_start[:3, 3]
camera_orientation = H_start[:3, :3]

# Shift along camera +Z axis (forward)
flower_center = camera_position + 0.10 * camera_orientation[:, 2]  # +Z axis direction

print(f"Estimated flower center in world frame: {flower_center}")


# # Define the sphere center (make sure it matches what you used during teleop)
# center = np.array([0.0, 0.0, 0.1])  # Adjust if necessary

# Calculate distances
distances = []
positions = []

for H in poses:
    position = H[:3, 3]
    positions.append(position)
    distance = np.linalg.norm(position - flower_center)
    distances.append(distance)

# Print distances
for i, d in enumerate(distances):
    print(f"Pose {i+1}: Distance to center = {d*100:.2f} cm")

# Plot distances
plt.figure()
plt.plot(distances, marker='o')
plt.title('Distance from Camera to Flower Center')
plt.xlabel('Pose Index')
plt.ylabel('Distance (m)')
plt.grid(True)
plt.show()

# Plot 3D scatter plot of camera positions
positions = np.array(positions)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', marker='o')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Camera Positions in 3D Space')

plt.show()
