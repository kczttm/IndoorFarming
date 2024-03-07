# ========== RRT IMPLEMENTATION USING OPEN3D OCTREE ========== #

import numpy as np
import plotly.express as px
import math
import open3d as o3d
import sys

## ALL UNITS SHOULD BE METERS TO ALIGN WITH UR5 MOVEMENT COMMAND UNITS

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

    def __init__(self, start, goal, obstacle_list, rand_area, max_expansion=0.003, path_resolution=0.5, goal_sample_rate=5, max_iter=10000, workspace_bounds=None, robot_radius=0.035):
        
        self.start = self.Node(start[0], start[1], start[2])
        self.end = self.Node(goal[0], goal[1], goal[2])
        self.min_r = rand_area[0]
        self.max_r = rand_area[1]

        if workspace_bounds is not None:
            self.workspace_bounds = self.Workspace(workspace_bounds)
        else:
            self.workspace_bounds = None

        self.max_expansion = max_expansion # Maximum distance the algorithm can move for each iteration
        self.path_resolution = path_resolution # Determines final path distance spacing between each point
        self.goal_sample_rate = goal_sample_rate # Number that represents how likely we want the random node generator to pick goal point
        self.max_iter = max_iter # Limit of iterations run for the algorithm
        self.obstacle_list = obstacle_list # Obstacles assumed to be spheres for now --> [x, y, z, radius]
        self.robot_radius = robot_radius
        self.node_list = []

    def planning(self, octree):

        self.node_list = [self.start] # Initialize list of nodes, starting with the start position

        iteration_count = 0

        # Iterate through maximum iterations or until path is found
        for _ in range(self.max_iter):

            # print("\nCurrent Node List: ")
            # for node in self.node_list:
            #     print(np.array([node.x, node.y, node.z]))
            
            # print("\nEnd of Node List.")
                
            rand_node = self.get_random_node() # Generate a random node in the workspace
            nearest_ind = self.get_nearest_node_index(self.node_list, rand_node) # Find index of nearest node to random node in current list of nodes (tree)
            nearest_node = self.node_list[nearest_ind] # Declare nearest node found
            new_node = self.steer(nearest_node, rand_node, self.max_expansion) # Create new node in direction of randomly generated node from the nearest node

            # Modify to check if the new_node is within the play area in 3D and has no collision with obstacle
            if self.check_workspace_bounds(new_node, self.workspace_bounds) and self.no_collision_check(new_node, octree, self.robot_radius):
                self.node_list.append(new_node)

            # Modify to check if the distance to goal is less than or equal to the expand_dis in 3D
            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y, self.node_list[-1].z) <= self.max_expansion:
                
                final_node = self.steer(self.node_list[-1], self.end, self.max_expansion)
                
                # Modify to check collision in 3D for the final_node
                if self.no_collision_check(final_node, octree, self.robot_radius):
                    return self.generate_final_course(len(self.node_list) - 1), iteration_count
                
            iteration_count += 1

        return None, iteration_count  # Cannot find path

    def steer(self, from_node, to_node, extend_length=float(0.007)):

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
    
    @staticmethod
    def check_workspace_bounds(node, workspace_bounds):
        
        if workspace_bounds is None:
            return True
        
        if node.x < workspace_bounds.x_min or node.x > workspace_bounds.x_max or node.y < workspace_bounds.y_min or node.y > workspace_bounds.y_max or node.z < workspace_bounds.z_min or node.z > workspace_bounds.z_max:
            return False
        else:
            return True

    @staticmethod
    def no_collision_check(node, octree, robot_radius):

        # Model Robot as sphere and discretize surface to check for collisions
        robot_sphere_points = []
        num_points_discretize = 50
        center = np.array([node.x, node.y, node.z])
        
        for phi in np.linspace(0, 2 * np.pi, num_points_discretize):
            for theta in np.linspace(0, 2 * np.pi, num_points_discretize):
                x = center[0] + robot_radius * np.sin(phi) * np.cos(theta)
                y = center[1] + robot_radius * np.sin(phi) * np.sin(theta)
                z = center[2] + robot_radius * np.cos(phi)
                robot_sphere_points.append([x, y, z])

        for point in robot_sphere_points:
            leaf_node_surface, leaf_node_info_surface = octree.locate_leaf_node(point)
            if leaf_node_surface is None:
                does_not_intersect = True
            else:
                does_not_intersect = False

        leaf_node_center, leaf_node_info_center = octree.locate_leaf_node(center)

        if leaf_node_center is None and does_not_intersect is True:
            # print("\nNo collision!")
            return True # no collision detected 
        else:
            print("\nCollision Detected.")
            return False # collision detected

        # center = np.array([node.x, node.y, node.z])
        # leaf_node_center, leaf_node_info_center = octree.locate_leaf_node(center)

        # if leaf_node_center is None:
        #     # print("\nNo collision!")
        #     return True # no collision detected 
        # else:
        #     print("\nCollision Detected.")
        #     return False # collision detected

    def generate_final_course(self, goal_ind):

        path = [[self.end.x, self.end.y, self.end.z]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y, node.z])
            node = node.parent
        path.append([node.x, node.y, node.z])

        for point in path:
            point[0] = round(point[0], 3)
            point[1] = round(point[1], 3)
            point[2] = round(point[2], 4)

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

    def calc_dist_to_start(self, node_x, node_y, node_z):

        dx = node_x - self.start.x
        dy = node_y - self.start.y
        dz = node_z - self.start.z

        return math.sqrt(dx**2 + dy**2 + dz**2)
    
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
            print()
        else:
            print()
            
# Main method        
def main(goal_pos, octree):
       
    # ====Search Path with RRT====
    print("\nStarting RRT Path Planning.")

    # Set Initial parameters
    start_pos = [0, 0, 0] # starting position in meters
    workspace_bounds = np.array([[-1, 1], [-1, 1], [-1, 1]])
    rrt = RRT(start=start_pos, goal=goal_pos, rand_area=[-1, 1], obstacle_list=octree, workspace_bounds=workspace_bounds, robot_radius=0.035)
    path, iteration_count = rrt.planning(octree)

    print("\nRan RRT algorithm without failures.")

    if path is None:
        
        print("\nCannot find path. Exiting.")
        rrt.plot_final_path(None, start_pos, goal_pos, workspace_bounds, octree)
        # sys.exit()

    else:

        print("\nFound a path!")
        pose_list = []

        # Creating robot poses for each point along path
        for position in path:
            
            rpy = np.array([0, 0, 0]) # Roll, Pitch, Yaw vector
            pose = np.hstack((position, rpy)) # Convert to 1x6 row vector for move_robot function
            pose_list.append(pose)

        # print("\nGenerated Pose List for Robot: ")
        # for pose in pose_list:
            # print(pose)
            # print()

        # print("Path: ", path)

        rrt.plot_final_path(path, start_pos, goal_pos, workspace_bounds, octree)        

        return rrt, path, pose_list, iteration_count # path is list of 1 x 3 arrays representing the points in space that compose the final path, pose_list is a list of 1 x 6 points to include roll, pitch, yaw for UR5

# Checks whether script is being run as main program or if it's being imported as a module into another script
if __name__ == '__main__':
    obstacle_3d_points = np.loadtxt('obstacle_3d_points.txt')
    pcd = o3d.geometry.PointCloud() # Init point cloud object
    points = o3d.utility.Vector3dVector(obstacle_3d_points) # Convert numpy array of points to Open3D Vector
    pcd.points = points
    octree_max_depth = 5
    octree = o3d.geometry.Octree(max_depth=octree_max_depth)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)

    goal_pos = np.array([-0.05, 0.05, 0.4])
    rrt, path, pose_list = main(goal_pos, octree)
    # main2(rrt, path, goal_pos)