from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'proj_farmhand'

data_files=[
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]

def package_files(data_files, directory_list):
    paths_dict = {}
    for directory in directory_list:

        for (path, directories, filenames) in os.walk(directory):

            for filename in filenames:

                file_path = os.path.join(path, filename)
                install_path = os.path.join('share', package_name, path)

                if install_path in paths_dict.keys():
                    paths_dict[install_path].append(file_path)

                else:
                    paths_dict[install_path] = [file_path]

    for key in paths_dict.keys():
        data_files.append((key, paths_dict[key]))

    return data_files

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files = package_files(data_files, ['launch/', 'config/']),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='C. Kong',
    maintainer_email='ckong35@gatech.edu',
    description='TODO: Package description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'test_RAFT_ICP = proj_farmhand.main_realtime_RAFT_ICP:main',
            'yolo_display = proj_farmhand.main_realtime_yolo:main',
            'yolo_pursuit_action_server = proj_farmhand.yolo_pursuit_action_server:main',
            'yolo_pursuit_action_client = proj_farmhand.yolo_pursuit_action_client:main',
            'main_pipeline = proj_farmhand.main_full_pipeline:main',
        ],
    },
)
