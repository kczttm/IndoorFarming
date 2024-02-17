import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_cylinder_along_y(x_center, z_center, radius, height):
    # Generate cylindrical coordinates
    theta = np.linspace(0, 2 * np.pi, 100)
    y = np.linspace(-height / 2, height / 2, 10)
    theta_grid, y_grid = np.meshgrid(theta, y)
    x_grid = radius * np.cos(theta_grid) + x_center
    z_grid = radius * np.sin(theta_grid) + z_center

    # Plot cylinder
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_grid, y_grid, z_grid, color='b', alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Cylinder along Y-axis')
    plt.show()

# Example usage
x_center = 0
z_center = 0
radius = 1
height = 2
plot_cylinder_along_y(x_center, z_center, radius, height)