import matplotlib.pyplot as plt
import numpy as np

# Load the image
image = plt.imread('Test_Leaf_Obstacle_Frame_2.png')

# Display the image
plt.imshow(image)
plt.title('Click on the image to get the pixel location')
plt.axis('on')

# Function to handle mouse click event
def onclick(event):
    if event.button == 1:
        # Get the pixel coordinates
        x, y = int(event.xdata), int(event.ydata)
        print("Pixel location (x, y):", x, y)

# Connect the onclick function to the figure
plt.gcf().canvas.mpl_connect('button_press_event', onclick)

# Show the plot
plt.show()