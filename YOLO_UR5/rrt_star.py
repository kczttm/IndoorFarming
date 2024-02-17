import numpy as np
import matplotlib.pyplot as plt
import math
import UR5_commands as robot
import time

show_animation = True

# Class for Cylinder Obstacle 
class Cylinder:
    def __init__(self, center, radius, height, direction):
        self.center = center
        self.radius = radius
        self.height = height
        self.direction = direction

# Class for RRT tree
class RRT:
    class Node:
        def __init__(self, x, y, z): 
            self.x = x
            self.y = y
            self.z = z
            self.parent = None
            self.cost = 0 

    class Workspace:
        def __init__(self, volume):
            self.x_min, self.x_max = float(volume[0, 0]), float(volume[0, 1])
            self.y_min, self.y_max = float(volume[1, 0]), float(volume[1, 1])
            self.z_min, self.z_max = float(volume[2, 0]), float(volume[2, 1])

    def __init__(self, start, goal, obstacle_list, rand_area, max_expansion=0.005, path_resolution=0.5, goal_sample_rate=5, max_iter=1000, workspace_bounds=None, robot_radius=1):
        
        self.start = self.Node(start[0], start[1], start[2])
        self.end = self.Node(goal[0], goal[1], goal[2])
        self.min_r, self.max_r = rand_area[0], rand_area[1]
        self.workspace_bounds = self.Workspace(workspace_bounds) if workspace_bounds is not None else None
        self.max_expansion = max_expansion
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.robot_radius = robot_radius
        self.node_list = [self.start]

    def planning(self, obstacle_list):
        
        for iteration in range(self.max_iter):

            rand_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rand_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rand_node, self.max_expansion)

            if self.check_workspace_bounds(new_node, self.workspace_bounds) and self.no_collision_check(new_node, obstacle_list[0]):
                self.node_list.append(new_node)
                self.rewire(new_node)

            if self.calc_dist_to_goal(self.node_list[-1]) <= self.max_expansion:
                final_node = self.steer(self.node_list[-1], self.end, self.max_expansion)
                if self.no_collision_check(final_node, obstacle_list[0]):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None, None  # Cannot find path

    def steer(self, from_node, to_node, extend_length=float("inf")):
        
        vector = np.array([to_node.x - from_node.x, to_node.y - from_node.y, to_node.z - from_node.z])
        distance = np.linalg.norm(vector)

        if distance > extend_length:
            vector = extend_length * (vector / distance)

        new_node = self.Node(from_node.x + vector[0], from_node.y + vector[1], from_node.z + vector[2])
        new_node.parent = from_node
        self.set_cost(new_node)

        return new_node
    
    # Calculate euclidean distance between two nodes
    def calculate_dist(self, node_1, node_2):

        distance = np.linalg.norm(np.array([node_2.x - node_1.x, node_2.y - node_1.y, node_2.z - node_1.z]))

        return distance

    # Calculates and sets cost for new node 
    def set_cost(self, new_node):

        new_node_copy = self.Node(new_node.x, new_node.y, new_node.z)
        new_node_copy.parent = new_node.parent
        cost = 0
    
        while new_node_copy.parent is not None:
            dist = self.calculate_dist(new_node_copy, new_node_copy.parent)
            cost += dist
            new_node_copy = new_node_copy.parent

        new_node.cost = cost

    def find_near_nodes(self, new_node):
        
        num_nodes = len(self.node_list) + 1
        r = self.max_expansion * math.sqrt((math.log(num_nodes) / num_nodes))
        dist_list = [self.calc_dist_to_goal(node) for node in self.node_list]
        near_inds = [ind for ind, d in enumerate(dist_list) if d <= r]

        return near_inds

    def propagate_cost_to_leaves(self, parent_node):

        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = parent_node.cost + self.calculate_dist(node, parent_node)
                self.propagate_cost_to_leaves(node)


    def generate_final_course(self, goal_ind):
        
        path = [[self.end.x, self.end.y, self.end.z]]
        node = self.node_list[goal_ind]
        final_node = self.node_list[goal_ind]

        while node.parent is not None:
            path.append([node.x, node.y, node.z])
            node = node.parent

        path.append([node.x, node.y, node.z])

        for point in path:
            point[0] = round(point[0], 2)
            point[1] = round(point[1], 2)
            point[2] = round(point[2], 2)

        path.reverse()

        return path, final_node

    def calc_dist_to_goal(self, node):

        dx = node.x - self.end.x
        dy = node.y - self.end.y
        dz = node.z - self.end.z

        return math.sqrt(dx**2 + dy**2 + dz**2)

    def calc_dist_to_start(self, node_x, node_y, node_z):

        dx = node_x - self.start.x
        dy = node_y - self.start.y
        dz = node_z - self.start.z

        return math.sqrt(dx**2 + dy**2 + dz**2)

    def get_random_node(self):

        rand = np.random

        if np.random.randint(0, 100) > self.goal_sample_rate:

            x = rand.uniform(self.workspace_bounds.x_min, self.workspace_bounds.x_max)
            y = rand.uniform(self.workspace_bounds.y_min, self.workspace_bounds.y_max)
            z = rand.uniform(self.workspace_bounds.z_min, self.workspace_bounds.z_max)
            node = self.Node(x, y, z)

        else:
            node = self.Node(self.end.x, self.end.y, self.end.z)
        
        # print("\nRandom Node Generated: ", np.array([round(node.x, 2), round(node.y, 2), round(node.z, 2)]))

        return node

    def get_nearest_node_index(self, node_list, rnd_node):

        dist_list = [self.calc_dist_to_goal(node) for node in node_list]
        nearest_ind = dist_list.index(min(dist_list))

        return nearest_ind

    def check_workspace_bounds(self, node, workspace_bounds):

        if workspace_bounds is None:
            return True
        return (
            workspace_bounds.x_min <= node.x <= workspace_bounds.x_max and
            workspace_bounds.y_min <= node.y <= workspace_bounds.y_max and
            workspace_bounds.z_min <= node.z <= workspace_bounds.z_max
        )

    def no_collision_check(self, node, obstacle):

        node_coords = np.array([node.x, node.y, node.z])
        cylinder_axis = np.array([0, 0, 1])
        vector_to_point = node_coords - obstacle.center
        projection_onto_axis = np.dot(vector_to_point, cylinder_axis) / np.linalg.norm(cylinder_axis)
        perpendicular_distance = np.linalg.norm(vector_to_point - projection_onto_axis * cylinder_axis)
        if 0 <= projection_onto_axis <= obstacle.height and perpendicular_distance <= obstacle.radius:
            # print("\nCollision detected, skip current iteration.")
            return False
        else:
            # print("\nNo collision detected, add node to tree.")
            return True

    def rewire(self, new_node):

        near_inds = self.find_near_nodes(new_node)

        for near_ind in near_inds:
            near_node = self.node_list[near_ind]
            edge_cost = self.calc_dist_to_goal(near_node)
            new_cost = new_node.cost + edge_cost

            if new_cost < near_node.cost and self.no_collision_check(new_node, self.obstacle_list[0]):
                near_node.parent = new_node
                near_node.cost = new_cost
                self.propagate_cost_to_leaves(near_node)


    def plot_final_path(self, path, start, goal, workspace_bounds, obstacle_list):
        
        if path is not None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')        
            path_arr = np.array(path)
            ax.scatter(path_arr[:, 0], path_arr[:, 1], path_arr[:, 2], c='b', marker='.')
            ax.scatter(start[0], start[1], start[2], c='g', marker='o', label='Start')
            ax.scatter(goal[0], goal[1], goal[2], c='r', marker='o', label='Goal')
            ax.plot(path_arr[:, 0], path_arr[:, 1], path_arr[:, 2], c='b', linestyle='-', linewidth=1)

            x_cylinder, y_cylinder_, z_cylinder = self.cylinder_along_z(obstacle_list[0].center[0], obstacle_list[0].center[1], obstacle_list[0].radius, obstacle_list[0].height)
            ax.plot_surface(x_cylinder, y_cylinder_, z_cylinder, color='gray', alpha=1)

            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')

            ax.set_xlim([0, 20])
            ax.set_ylim([0, 20])
            ax.set_zlim([0, 20])

            ax.set_title("RRT-Star 3D")
            ax.legend()
            plt.show()

        else:

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')        
            ax.scatter(start[0], start[1], start[2], c='g', marker='o', label='Start')
            ax.scatter(goal[0], goal[1], goal[2], c='r', marker='o', label='Goal')
            x_cylinder, y_cylinder_, z_cylinder = self.cylinder_along_z(obstacle_list[0].center[0], obstacle_list[0].center[1], obstacle_list[0].radius, obstacle_list[0].height)
            ax.plot_surface(x_cylinder, y_cylinder_, z_cylinder, color='gray', alpha=1)

            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')

            ax.set_xlim([0, 20])
            ax.set_ylim([0, 20])
            ax.set_zlim([0, 20])

            ax.set_title("RRT-Star 3D")
            ax.legend()
            plt.show()

    def cylinder_along_z(self, x_center, y_center, radius, height):

        z = np.linspace(0, height, 50)
        theta = np.linspace(0, 2 * np.pi, 50)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid) + x_center
        y_grid = radius * np.sin(theta_grid) + y_center

        return x_grid, y_grid, z_grid

# Main method        
def main():

    print("start " + __file__)

    # center = np.random.uniform(0, 20, size=3)
    # radius = np.random.uniform(1, 5)
    # height = np.random.uniform(5, 15)
    # direction = np.random.uniform(-1, 1, size=3)

    center = np.array([10, 5, 10])
    radius = 2
    height = 8
    direction = np.array([0, 0, 1])

    cylinder1 = Cylinder(center, radius, height, direction)
    obstacle_list = [cylinder1]

    startPos = [0, 0, 0]
    goalPos = [20, 20, 0]
    workspace_bounds = np.array([[0, 20], [0, 20], [0, 20]])
    rrt = RRT(start=startPos, goal=goalPos, rand_area=[-2, 15], obstacle_list=obstacle_list, workspace_bounds=workspace_bounds, robot_radius=0.3)
    path, final_node = rrt.planning(obstacle_list)

    if path is None:

        print("\nCannot find path")
        print("Final Node List: ", rrt.node_list)
        rrt.plot_final_path(None, startPos, goalPos, workspace_bounds, obstacle_list)

    else:

        print("\nFound a path!")
        final_path_cost = final_node.cost
        print(f"\nFinal Path Cost: {final_path_cost}") # Print the final path cost
        rrt.plot_final_path(path, startPos, goalPos, workspace_bounds, obstacle_list)

        print("\nInitiating robot path execution.")

        for position in path:
            rpy = np.array([0, 0, 0])
            pose = np.hstack((position, rpy))
            robot.move_robot(pose)
            time.sleep(2)

if __name__ == '__main__':
    main()
