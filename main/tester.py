import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

# Generate some data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a plot
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)

# Show the plot
plt.show()
