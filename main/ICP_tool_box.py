import numpy as np
np.set_printoptions(suppress=True)
import copy
import os, sys
import open3d as o3d

from vision_pkg.pointcloud_pub import PointCloudPublisher

curr_dir_abs = os.path.abspath(os.path.dirname(__file__))

def filter_pts_file(filename, x_min, x_max, y_min, y_max, z_min, z_max):

    coordinates = []
    colors=[]
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:  # Ensure it's a valid line with XYZ coordinates
                # Extract XYZ coordinates
                x, y, z = map(float, parts[:3])
                if (x_min <= x) and (x <= x_max) and (y_min <= y) and (y <= y_max) and (z_min <= z) and (z <= z_max):
                    coordinates.append([x, y, z])
                    r, g, b = map(int, parts[4:7])
                    colors.append([r, g, b])

    filtered_points = np.hstack((np.array(coordinates), np.array(colors)))

    return filtered_points


def filter_pts_data(points, x_min, x_max, y_min, y_max, z_min, z_max):
        filtered_points = points[(points[:, 0] >= x_min) & (points[:, 0] <= x_max) & 
                                (points[:, 1] >= y_min) & (points[:, 1] <= y_max) & 
                                (points[:, 2] >= z_min) & (points[:, 2] <= z_max)]
    
        return filtered_points


def read_pts_file(file_path):
    
    points = []
    colors = []

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip the first line (number of points)
            data = line.strip().split()
            x, y, z = map(float, data[:3])
            points.append((x, y, z))
            if len(data) >= 7:  # Check if color information is available
                r, g, b = map(int, data[4:7])
                colors.append((r, g, b))

    return points, colors


def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles to a rotation matrix.

    Args:
    - roll (float): Rotation around the x-axis (in radians).
    - pitch (float): Rotation around the y-axis (in radians).
    - yaw (float): Rotation around the z-axis (in radians).

    Returns:
    - rotation_matrix (numpy.ndarray): 3x3 rotation matrix.
    """
    # Calculate the trigonometric values
    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # Construct the rotation matrix
    rotation_matrix = np.array([
        [cos_yaw * cos_pitch, cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll, cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll],
        [sin_yaw * cos_pitch, sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll, sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll],
        [-sin_pitch, cos_pitch * sin_roll, cos_pitch * cos_roll]
    ])

    return rotation_matrix


def rotate_pcd_htm(point_cloud, htm_transformation):

    # Create augmented vector for points
    pcd_xyz = point_cloud[:, 0:3]
    pcd_xyz = np.insert(pcd_xyz, 3, 1, axis=1)
    pcd_xyz = pcd_xyz.T # Transpose matrix to make each point a column instead of row

    # Apply transformations
    transformed_pcd = np.matmul(htm_transformation, pcd_xyz).T
    transformed_pcd = transformed_pcd[:, 0:3] # Get rid of columnn of 1's
    transformed_pcd = np.concatenate((transformed_pcd, point_cloud[:, 3:6]), axis=1) # Add colors back to the point cloud

    return np.round(transformed_pcd, decimals=5)


def rotation_matrix_to_euler(rotation_matrix):
    """
    Convert a rotation matrix to Euler angles.

    Args:
    - rotation_matrix (numpy.ndarray): 3x3 rotation matrix.

    Returns:
    - roll (float): Rotation around the x-axis (in radians).
    - pitch (float): Rotation around the y-axis (in radians).
    - yaw (float): Rotation around the z-axis (in radians).
    """
    # Extract the Euler angles from the rotation matrix
    roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    pitch = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2))
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    return roll, pitch, yaw


def np_to_o3d_point_cloud(np_point_cloud, color = [0.0, 1.0, 0.0]):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_point_cloud[:, 0:3])
    pcd.paint_uniform_color(color)
    return pcd


def get_flower_template_pcd(visualize=True):
    isolated_flower_file_path = os.path.join(curr_dir_abs, "PolyCam_Point_Clouds", "Isolated_Fake_Flower.pts")
    x_min, x_max, y_min, y_max, z_min, z_max = -np.inf, np.inf, -0.05, 0.1, 0.05, 0.1 
    # change depending on what resolution is needed, units in meters
    fake_flower_template = np.round(filter_pts_file(isolated_flower_file_path, 
                                            x_min, x_max, y_min, y_max, z_min, z_max), decimals=4) 
    fake_flower_centroid = np.mean(fake_flower_template[:, 0:3], axis=0)
    # rotate and translate the template flower to the origin
    rpy_est = np.radians([-10, 150, 0])
    rot_est = euler_to_rotation_matrix(rpy_est[0], rpy_est[1], rpy_est[2])
    H_centering = np.eye(4)
    H_centering[0:3, 3] = -fake_flower_centroid
    H_aligning = np.eye(4)
    H_aligning[0:3, 0:3] = rot_est
    H_est = H_aligning @ H_centering    
    fake_flower_template_rotated = rotate_pcd_htm(fake_flower_template, H_est)
    x_min, x_max, y_min, y_max, z_min, z_max = -np.inf, np.inf, -np.inf, np.inf, -np.inf, 0.01 
    filtered_flower = filter_pts_data(fake_flower_template_rotated, 
                                      x_min, x_max, y_min, y_max, z_min, z_max)


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_flower[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(filtered_flower[:, 3:6])
    pcd.paint_uniform_color([1,0,0])

    if visualize:
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, coordinate_frame])
    
    return pcd


# Pre-processing before applying registration methods
def preprocess_point_cloud(pcd, voxel_size):

    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    # remove outliers from the point cloud
    # cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # Apply radius outlier removal
    # cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.05)

    return pcd_down, pcd_fpfh


# Fast Global Registration from https://vladlen.info/papers/fast-global-registration.pdf (Open3D implementation)
def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):

    distance_threshold = voxel_size * 4.0
    print(":: Apply fast global registration with distance threshold %.3f" % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(source=source_down, target=target_down, source_feature=source_fpfh, target_feature=target_fpfh, 
                                                                                   option=o3d.pipelines.registration.FastGlobalRegistrationOption(
                                                                                       maximum_correspondence_distance=distance_threshold))
    
    return result

# RANSAC Registration
def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    
    distance_threshold = voxel_size * 4.0
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(source=source_down, target=target_down, source_feature=source_fpfh, target_feature=target_fpfh, 
                                                                                      mutual_filter=True, max_correspondence_distance=distance_threshold, 
                                                                                      estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False), ransac_n=3, 
                                                                                      checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), 
                                                                                                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)], 
                                                                                      criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.9999))
    
    return result


def refine_registration(source, target, init_transformation, voxel_size):
    
    distance_threshold = voxel_size * 1.5
    # print(":: Point-to-point ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(source, target, distance_threshold, init_transformation,
                                                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000000, relative_rmse=1.0e-09))
    
    return result


def draw_registration_result(source, target, transformation):
    
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 1, 0])
    source_temp.transform(transformation)
    # Create a coordinate frame: x-axis -> red, y-axis -> green, z-axis -> blue
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([source_temp, target_temp, coordinate_frame])


def rotate_frame_on_ball(ball_center, roll, pitch, yaw):
    """
    Rotate a frame on a ball centered at the ball_center 
        with radius equal to the distance from the camera 
        to the flower.

    ball_center is a (3,) numpy array
    roll, pitch, yaw are in radians
    """

    # Create rotation matrix for each angle
    combined_rot = euler_to_rotation_matrix(roll, pitch, yaw)
    
    H_translation = np.eye(4)
    H_translation[0:3, 3] = ball_center  
    # note that here robot might not be perfectly pointing at the flower origin
    # and we are currently keeping it that way until further testing
    # result is not the most optimal, can choose 
    # 1) trust this pose estimate and point directly to the center of the flower
    # 2) do the yolo_pursuit again and use the previous pose as initial guess, then do ICP again with the new raft
    # this two can be combined too.
    H_rotation = np.eye(4)
    H_rotation[0:3, 0:3] = combined_rot

    # H_backward = np.linalg.inv(H_translation)
    # back up in z direction by ball radius
    radius = np.linalg.norm(ball_center)
    H_backward = np.eye(4)
    H_backward[2, 3] = -radius

    return H_translation @ H_rotation @ H_backward