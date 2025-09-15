import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess

from launch.substitutions import Command, FindExecutable
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import xacro
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch.actions import IncludeLaunchDescription, RegisterEventHandler

from launch_xml.launch_description_sources import XMLLaunchDescriptionSource

def generate_launch_description():
    gazebo = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('gazebo_ros'), 'launch'), '/gazebo.launch.py']),
             )

    robotarm_path = os.path.join(
        get_package_share_directory('robotarm1'))
    xacro_file = os.path.join(robotarm_path,
                              'urdf',
                              'robot_thesis.xacro')

    doc = xacro.parse(open(xacro_file))
    xacro.process_doc(doc)
    params = {'robot_description': doc.toxml()}

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params]
    )

    spawn_entity = Node(package='gazebo_ros', executable='spawn_entity.py',
                        arguments=['-topic', 'robot_description',
                                   '-entity', 'manipulator'],
                        output='screen')

    load_joint_state_broadcaster = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'joint_state_broadcaster'],
        output='screen'
    )

    load_joint_trajectory_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'joint_trajectory_controller'],
        output='screen'
    )
    ik_launch_file = IncludeLaunchDescription(
        XMLLaunchDescriptionSource(
            os.path.join(('node_bringup'),'launch/ik_app.launch.xml')))
    kinematics_node = Node(
        package='ik_pkg',
        executable='kinematics_node',
        name='kinematics_node',
        output='screen',
        # parameters=[{'robot_description': Command(['xacro ', xacro_file])}],
        # remappings=[
        #     ('/compute_ik', '/computation_service'),
        #     ('/control_motor', '/control_motor')
        # ]
    )
    user_node = Node(
        package='ik_pkg',
        executable='input_node',
        name='input_node',
        output='screen',
        # parameters=[{'robot_description': Command(['xacro ', xacro_file])}],
        # remappings=[
        #     ('/compute_ik', '/computation_service'),
        #     ('/control_motor', '/control_motor')
        # ]
    )

    return LaunchDescription([
        # RegisterEventHandler(
        #     event_handler=OnProcessExit(
        #         target_action=spawn_entity,
        #         on_exit=[load_joint_state_broadcaster],
        #     )
        # ),
        # RegisterEventHandler(
        #     event_handler=OnProcessExit(
        #         target_action=load_joint_state_broadcaster,
        #         on_exit=[load_joint_trajectory_controller],
        #     )
        # ),
        gazebo,
        node_robot_state_publisher,
        spawn_entity,
    ])
