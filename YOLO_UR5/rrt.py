import numpy as np
import matplotlib.pyplot as plt
import math
import UR5_commands as robot
import time

## ALL UNITS SHOULD BE METERS TO ALIGN WITH UR5 MOVEMENT COMMAND UNITS

# Class for Cylinder Obstacle 
class Cylinder:

    def __init__(self, center, radius, height, direction):
        self.center = center
        self.radius = radius
        self.height = height
        self.direction = direction

# Class for RRT tree
class RRT:

    # Class for Nodes (e.g. points)
    class Node:

        # Initialize Node fields
        def __init__(self, x, y, z): 
            self.x = x
            self.y = y
            self.z = z
            self.path_x = []
            self.path_y = []
            self.path_z = []
            self.parent = None

        # Override __str__ method for better representation
        def __str__(self):
            return f"Node(x={self.x}, y={self.y}, z={self.z})"

    # Class for Workspace 
    class Workspace:

        def __init__(self, volume):
            self.x_min = float(volume[0, 0])
            self.x_max = float(volume[0, 1])
            self.y_min = float(volume[1, 0])
            self.y_max = float(volume[1, 1])
            self.z_min = float(volume[2, 0])
            self.z_max = float(volume[2, 1])

    def __init__(self, start, goal, obstacle_list, rand_area, max_expansion=0.005, path_resolution=0.5, goal_sample_rate=5, max_iter=500, workspace_bounds=None, robot_radius=0.001):
        
        self.start = self.Node(start[0], start[1], start[2])
        self.end = self.Node(goal[0], goal[1], goal[2])
        self.min_r = rand_area[0]
        self.max_r = rand_area[1]

        if workspace_bounds is not None:
            self.workspace_bounds = self.Workspace(workspace_bounds)
        else:
            self.workspace_bounds = None

        self.max_expansion = max_expansion
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate # Number that represents how likely we want the random node generator to pick goal point
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list # Obstacles assumed to be spheres for now --> [x, y, z, radius]
        self.robot_radius = robot_radius
        self.node_list = []

    def planning(self, obstacle_list):

        self.node_list = [self.start] # Initialize list of nodes, starting with the start position

        # Iterate through maximum iterations or until path is found
        for iteration in range(self.max_iter):

            # print("\nCurrent Node List: ")
            # for node in self.node_list:
            #     print(np.array([node.x, node.y, node.z]))
            
            # print("\nEnd of Node List.")
                
            rand_node = self.get_random_node() # Generate a random node in the workspace
            nearest_ind = self.get_nearest_node_index(self.node_list, rand_node) # Find index of nearest node to random node in current list of nodes (tree)
            nearest_node = self.node_list[nearest_ind] # Declare nearest node found
            new_node = self.steer(nearest_node, rand_node, self.max_expansion) # Create new node in direction of randomly generated node from the nearest node

            # Modify to check if the new_node is within the play area in 3D
            if self.check_workspace_bounds(new_node, self.workspace_bounds) and self.no_collision_check(new_node, obstacle_list[0].center, obstacle_list[0].radius, obstacle_list[0].height, self.robot_radius):
                self.node_list.append(new_node)

            else:
                print("\nDid not meet adding to node list requirement.")

            # Modify to check if the distance to goal is less than or equal to the expand_dis in 3D
            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y, self.node_list[-1].z) <= self.max_expansion:
                
                final_node = self.steer(self.node_list[-1], self.end, self.max_expansion)
                
                # Modify to check collision in 3D for the final_node
                if self.no_collision_check(final_node, obstacle_list[0].center, obstacle_list[0].radius, obstacle_list[0].height, self.robot_radius):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None  # Cannot find path

    def steer(self, from_node, to_node, extend_length=float(0.003)):

        # Steer towards the target node while maintaining a maximum distance
        to_Node_Arr = np.array([to_node.x, to_node.y, to_node.z])
        from_Node_Arr = np.array([from_node.x, from_node.y, from_node.z])
        direction = to_Node_Arr - from_Node_Arr # Calculate vector between two nodes
        distance = np.linalg.norm(direction) # Calculate magnitude of vector

        # Limit the maximum distance traveled while steering
        if distance > extend_length:
            direction *= extend_length / distance

        new_node = self.Node(from_node.x, from_node.y, from_node.z) # Create node variable for starting node passed in
        new_node.x += direction[0] # Adjust x coordinate in direction of vector
        new_node.y += direction[1] # Adjust y coordinate in direction of vector
        new_node.z += direction[2] # Adjust z coordinate in direction of vector
        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_ind):

        path = [[self.end.x, self.end.y, self.end.z]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y, node.z])
            node = node.parent
        path.append([node.x, node.y, node.z])

        for point in path:
            point[0] = round(point[0], 2)
            point[1] = round(point[1], 2)
            point[2] = round(point[2], 2)

        path.reverse() # Change order so it starts with start point

        return path

    # Calculate distance to goal point
    def calc_dist_to_goal(self, x, y, z):

        dx = x - self.end.x
        dy = y - self.end.y
        dz = z - self.end.z
        return math.sqrt(dx**2 + dy**2 + dz**2)

    # Generates random node
    def get_random_node(self):
        
        rand = np.random # Instantiate numpy random module

        # If sampled number is higher than pre-defined sample rate, then generate random node
        if np.random.randint(0, 100) > self.goal_sample_rate: 

            x = rand.uniform(self.workspace_bounds.x_min, self.workspace_bounds.x_max)
            y = rand.uniform(self.workspace_bounds.y_min, self.workspace_bounds.y_max)
            z = rand.uniform(self.workspace_bounds.z_min, self.workspace_bounds.z_max)
            node = self.Node(x, y, z) 
            
        else:  # goal point sampling
            node = self.Node(self.end.x, self.end.y, self.end.z) # Else return the goal node as the random node

        return node

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        
        dist_list = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 + (node.z - rnd_node.z)**2 for node in node_list]
        nearest_ind = dist_list.index(min(dist_list))

        return nearest_ind

    @staticmethod
    def check_workspace_bounds(node, workspace_bounds):
        
        if workspace_bounds is None:
            return True
        
        if node.x < workspace_bounds.x_min or node.x > workspace_bounds.x_max or node.y < workspace_bounds.y_min or node.y > workspace_bounds.y_max or node.z < workspace_bounds.z_min or node.z > workspace_bounds.z_max:
            return False
        else:
            return True

    @staticmethod
    def no_collision_check(node, cylinder_center, cylinder_radius, cylinder_height, robot_radius):

        nodeCoords = np.array([node.x, node.y, node.z]) 

        cylinder_axis = np.array([0, 1, 0])
        # Vector from the cylinder center to the point
        vector_to_point = nodeCoords - cylinder_center

        # Project the vector onto the axis of the cylinder
        projection_onto_axis = np.dot(vector_to_point, cylinder_axis) / np.linalg.norm(cylinder_axis)

        # Calculate the perpendicular distance from the axis
        perpendicular_distance = np.linalg.norm(vector_to_point - projection_onto_axis * cylinder_axis)

        # Check if the point is within the cylinder
        if (projection_onto_axis <= (cylinder_height + robot_radius) and perpendicular_distance <= (cylinder_radius + robot_radius)):
            return False  # Collision detected
        else:
            return True  # No collision

    @staticmethod
    def calc_distance_and_angles(node_1, node_2):
            
        dx = node_2.x - node_1.x
        dy = node_2.y - node_1.y
        dz = node_2.z - node_1.z
        dist = math.sqrt(dx**2 + dy**2 + dz**2) # Euclidean distance in 3D space
        theta = math.atan2(dy, dx) # Azimuth angle (theta) in 3D space
        phi = math.atan2(math.sqrt(dx**2 + dy**2), dz) # Polar angle (phi) in 3D space

        return dist, theta, phi
    
    def plot_final_path(self, path, start, goal, workspace_bounds, obstacle_list):

        if path is not None:

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')        
            pathArr = np.array(path) # Convert list of tuples to numpy array
            ax.scatter(pathArr[:, 0], pathArr[:, 1], pathArr[:, 2], c='b', marker='.')
            ax.scatter(start[0], start[1], start[2], c='g', marker='o', label='Start') # Plot starting point
            ax.scatter(goal[0], goal[1], goal[2], c='r', marker='o', label='Goal') # Plot goal point

            ax.plot(pathArr[:, 0], pathArr[:, 1], pathArr[:, 2], c='b', linestyle='-', linewidth=1) # Connect the points in the scatter plot

            # Plot cylinder
            x_cylinder, y_cylinder_, z_cylinder = self.cylinder_along_y(obstacle_list[0].center[0], obstacle_list[0].center[2], obstacle_list[0].radius, obstacle_list[0].height) # Create plottable x, y, z values for cylinder
            ax.plot_surface(x_cylinder, y_cylinder_, z_cylinder, color='k', alpha=0.5) # Plot cylinder

            # Set limits 
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')

            # Set the same limits for x, y, and z axes
            ax.set_xlim([0, 0.3])
            ax.set_ylim([0, 0.3])
            ax.set_zlim([0, 0.3])

            ax.set_title("RRT 3D")
            ax.legend() # Create legend for plot
            plt.show() # Render plot

        else:

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')        
            ax.scatter(start[0], start[1], start[2], c='g', marker='o', label='Start') # Plot starting point
            ax.scatter(goal[0], goal[1], goal[2], c='r', marker='o', label='Goal') # Plot goal point
            x_cylinder, y_cylinder_, z_cylinder = self.cylinder_along_z(obstacle_list[0].center[0], obstacle_list[0].center[1], obstacle_list[0].radius, obstacle_list[0].height) # Create plottable x, y, z values for cylinder
            ax.plot_surface(x_cylinder, y_cylinder_, z_cylinder, color='k', alpha=1) # Plot cylinder

            # Set limits 
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')

            # Set the same limits for x, y, and z axes
            ax.set_xlim([0, 0.3])
            ax.set_ylim([0, 0.3])
            ax.set_zlim([0, 0.3])

            ax.set_title("RRT 3D")
            ax.legend() # Create legend for plot
            plt.show() # Render plot

    def cylinder_along_y(self, x_center, z_center, radius, height):

        # Generate cylindrical coordinates
        theta = np.linspace(0, 2 * np.pi, 100)
        y = np.linspace(-height / 2, height / 2, 10)
        theta_grid, y_grid = np.meshgrid(theta, y)
        x_grid = radius * np.cos(theta_grid) + x_center
        z_grid = radius * np.sin(theta_grid) + z_center

        return x_grid, y_grid, z_grid

    def rotation_matrix(self, axis, angle):
        # Compute rotation matrix for given axis and angle
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c
        x, y, z = axis / np.linalg.norm(axis)
        rotation_matrix = np.array([[t*x*x + c, t*x*y - z*s, t*x*z + y*s],
                                    [t*x*y + z*s, t*y*y + c, t*y*z - x*s],
                                    [t*x*z - y*s, t*y*z + x*s, t*z*z + c]])
        return rotation_matrix
    
    def cylinder_along_z(self, x_center, y_center, radius, height):

        z = np.linspace(0, height, 10)
        theta = np.linspace(0, 2 * np.pi, 40)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid) + x_center
        y_grid = radius * np.sin(theta_grid) + y_center

        return x_grid, y_grid, z_grid
    
    def calc_dist_to_start(self, node_x, node_y, node_z):

        dx = node_x - self.start.x
        dy = node_y - self.start.y
        dz = node_z - self.start.z

        return math.sqrt(dx**2 + dy**2 + dz**2)
        
# Main method        
def main(goal_pos):
       
    print("start " + __file__)

    # ====Search Path with RRT====

    # center = np.random.uniform(0, 20, size=3)  # Random center within the workspace bounds
    # radius = np.random.uniform(1, 5)  # Random radius within a range
    # height = np.random.uniform(5, 15)  # Random height within a range
    # direction = np.random.uniform(-1, 1, size=3)  # Random direction vector

    # print("\nCenter: ", center)
    # print("\nRadius: ", radius)
    # print("\nHeight: ", height)
    # print("\nDirection: ", direction)
    
    center = np.array([0, 0, 0.08])
    radius = 0.01
    height = 0.5
    direction = np.array([0, 1, 0])

    cylinder1 = Cylinder(center, radius, height, direction)
    obstacle_list = [cylinder1]  # [x, y, z, radius]

    # Set Initial parameters
    start_pos = [0, 0, 0]
    workspace_bounds = np.array([[0, 0.3], [0, 0.3], [0, 0.3]])
    rrt = RRT(start=start_pos, goal=goal_pos, rand_area=[-2, 15], obstacle_list=obstacle_list, workspace_bounds=workspace_bounds, robot_radius=0.025)
    path = rrt.planning(obstacle_list)

    if path is None:
        
        print("\nCannot find path")
        rrt.plot_final_path(None, start_pos, goal_pos, workspace_bounds, obstacle_list)
        exit()

    else:

        print("\nFound a path!")
        pose_list = []

        # Creating robot poses for each point along path
        for position in path:
            
            rpy = np.array([0, 0, 0]) # Roll, Pitch, Yaw vector
            pose = np.hstack((position, rpy)) # Convert to 1x6 row vector for move_robot function
            pose_list.append(pose)

        print("\nGenerated Pose List for Robot: ", pose_list)
        rrt.plot_final_path(path, start_pos, goal_pos, workspace_bounds, obstacle_list)

        return pose_list


# Checks whether script is being run as main program or if it's being imported as a module into another script
if __name__ == '__main__':
    goal_pos = [0.03, 0.025, 0.165]
    main(goal_pos)