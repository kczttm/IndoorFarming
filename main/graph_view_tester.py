import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Sample data
x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)

# Create scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.xaxis._axinfo['grid'].update(adjustable='box')
ax.yaxis._axinfo['grid'].update(adjustable='box')
ax.zaxis._axinfo['grid'].update(adjustable='box')
ax.set_xticks(ax.get_xticks())
ax.set_yticks(ax.get_yticks())
ax.set_zticks(ax.get_zticks())

# Enable interactive mode
plt.ion()
plt.show()

# Interactively manipulate the plot to the desired perspective

# Once you're satisfied with the perspective, retrieve the azimuth and elevation angles
azim = ax.azim
elev = ax.elev

# Turn off interactive mode
plt.ioff()

# Now you can use the obtained azimuth and elevation angles in your plot
print("Azimuth angle:", azim)
print("Elevation angle:", elev)

# Later, when you want to use the same perspective again:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
ax.view_init(elev=elev, azim=azim)  # Apply the same perspective
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
plt.show()