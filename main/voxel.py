import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_cube(ax, center, size):
    # Define vertices of the cube relative to the center point
    vertices = np.array([[ 1,  1,  1],
                         [ 1,  1, -1],
                         [ 1, -1,  1],
                         [ 1, -1, -1],
                         [-1,  1,  1],
                         [-1,  1, -1],
                         [-1, -1,  1],
                         [-1, -1, -1]]) * size / 2

    # Translate vertices to the center point
    vertices += center

    # Define edges of the cube
    edges = [[0, 1], [1, 3], [3, 2], [2, 0],
             [4, 5], [5, 7], [7, 6], [6, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    # Plot edges
    for edge in edges:
        ax.plot3D(*vertices[edge, :].T, color='r')

# Example usage
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

center = np.array([0, 0, 0])  # Center of the cube
size = 0.001  # Size of the cube
plot_cube(ax, center, size)

# Plot the point
ax.scatter(*center, color='r', s=1)

# Set axis limits
ax.set_xlim([-0.1, 0.1])
ax.set_ylim([-0.1, 0.1])
ax.set_zlim([-0.1, 0.1])

plt.show()