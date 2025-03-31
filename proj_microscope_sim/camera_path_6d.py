import numpy as np
import matplotlib.pyplot as plt


def generate_upper_hemisphere_path_with_orientation(radius=1.0, num_points=10):
    """
    Generate 6D points (3D position + 3D orientation) along the upper hemisphere of a sphere.

    Parameters:
        radius (float): Radius of the sphere.
        num_points (int): Number of points along the path.

    Returns:
        np.ndarray: Array of 6D points [x, y, z, roll, pitch, yaw].
    """

    # Generate points evenly distributed along the upper hemisphere
    theta = np.linspace(-np.pi / 2, np.pi / 2, num_points)  # Elevation angles
    phi = 0                                                 # Azimuthal angles (Rotation about the Z-axis)


    # Convert spherical to Cartesian coordinates
    # Case 1: Rotation about the X-axis
    # x = radius * np.sin(theta) * np.sin(phi)
    # y = radius * np.sin(theta) * np.cos(phi)
    # z = radius * np.cos(theta)

    # Case 2: Rotation about the Y-axis
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    # Case 3: Rotation about the Z-axis
    # x = radius * np.cos(theta)
    # y = radius * np.sin(theta) * np.cos(phi)
    # z = radius * np.sin(theta) * np.sin(phi)


    # Calculate global orientation (yaw)
    yaw = np.arctan2(y, x)                       # Azimuthal angle 

    points = np.column_stack((x, y, z, yaw))
    return points


def visualize_sphere_with_path(radius=1.0, path_points=None):
    """
    Visualize the sphere, path, and points along the upper hemisphere.

    Parameters:
        radius (float): Radius of the sphere.
        path_points (np.ndarray): Array of 5D points along the path.
    """

    # Create the sphere
    phi = np.linspace(0, 2 * np.pi, 30)
    theta = np.linspace(0, np.pi, 30)
    phi, theta = np.meshgrid(phi, theta)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    # Plot the sphere and the center
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, color='white', alpha=0.3, edgecolor='gray')
    ax.scatter(0, 0, 0, color='black', s=20, label="Center (Object)")

    # Plot the path
    if path_points is not None:
        path_x, path_y, path_z, yaw = path_points.T
        ax.scatter(path_x, path_y, path_z, color='red', s=20, label="Path Points")
        ax.plot(path_x, path_y, path_z, color='red', label="Path Trajectory")

        # Initialize rolling frame
        prev_x_axis = np.array([0, 1, 0])  # Initial reference x-axis

        # Visualize axes at each point
        for i in range(len(path_points)):
            point = np.array([path_x[i], path_y[i], path_z[i]])
            norm = np.linalg.norm(point)

            # Force local z-axis to point toward center (assume (0, 0, 0))
            local_z_axis = -point / np.linalg.norm(point)
            
            # Get the direction of forward motion (tangent to the path)
            if i < len(path_points) - 1:
                forward = path_points[i+1, :3] - path_points[i, :3]
            else:
                forward = path_points[i, :3] - path_points[i-1, :3]

            forward /= np.linalg.norm(forward)
            local_x_axis = forward

            # Fix axis flipping with previous direction
            if np.dot(local_x_axis, prev_x_axis) < 0:
                local_x_axis = -local_x_axis

            # Update rolling reference
            prev_x_axis = local_x_axis

            # Construct the remaining local y-axis
            local_y_axis = np.cross(local_z_axis, local_x_axis)
            local_y_axis /= np.linalg.norm(local_y_axis)

            # Re-orthogonalize x to ensure numerical stability
            local_x_axis = np.cross(local_y_axis, local_z_axis)

            print(f"point{i}: ")
            print("global position: ", point)
            print("local orientation (x): ", local_x_axis)
            print("local orientation (y): ", local_y_axis)
            print("local orientation (z): ", local_z_axis)
            print()

            # Plot local axes at this point
            ax.quiver(
                point[0], point[1], point[2],
                local_x_axis[0], local_x_axis[1], local_x_axis[2],
                color='blue', label="Local X-axis" if i == 0 else ""
            )
            ax.quiver(
                point[0], point[1], point[2],
                local_y_axis[0], local_y_axis[1], local_y_axis[2],
                color='green', label="Local Y-axis" if i == 0 else ""
            )
            ax.quiver(
                point[0], point[1], point[2],
                local_z_axis[0], local_z_axis[1], local_z_axis[2],
                color='purple', label="Local Z-axis" if i == 0 else ""
            )

    # Plot global center axes
    ax.quiver(0, 0, 0, 1, 0, 0, color='black', label="Global X-axis")
    ax.quiver(0, 0, 0, 0, 1, 0, color='black', linestyle='dashed', label="Global Y-axis")
    ax.quiver(0, 0, 0, 0, 0, 1, color='black', linestyle='dotted', label="Global Z-axis")

    # Labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Upper Hemisphere Path")
    ax.legend()
    plt.show()


# Generate path points with position and orientation
# radius = 5.0
# num_points = 15
# path_points = generate_upper_hemisphere_path_with_orientation(radius, num_points)

# Visualize the sphere and path
# visualize_sphere_with_path(radius, path_points)
