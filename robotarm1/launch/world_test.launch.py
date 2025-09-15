import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    package_name = "robotarm1"
    world_name = "chessboard.world"
    
    world_path = os.path.join(get_package_share_directory(package_name=package_name), 'worlds', world_name)

    # 1. Allow overriding the world file via a launch argument
    world_arg = DeclareLaunchArgument(
        'world',
        default_value= world_path,
        description='Full path to the Gazebo world file'
    )

    # 2. Launch gzserver (the physics engine) with --verbose and the world
    gzserver = ExecuteProcess(
        cmd=[
            'gzserver',
            '--verbose',
            '-s',
            'libgazebo_ros_init.so',
            '-s',
            'libgazebo_ros_factory.so',
            LaunchConfiguration('world')
        ],
        output='screen'
    ) # If you open the world first you need to have pluggin, declared as: -s libgazebo_ros_init.so -s libgazebo_ros_factory.so

    # 3. Launch gzclient (the GUI)
    gzclient = ExecuteProcess(
        cmd=['gzclient'],
        output='screen'
    )

    return LaunchDescription([
        world_arg,
        gzserver,
        gzclient,
    ])