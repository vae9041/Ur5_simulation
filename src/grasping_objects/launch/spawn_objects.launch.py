from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from launch.substitutions import Command, FindExecutable
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_path = get_package_share_directory('grasping_objects')
    urdf_path = os.path.join(pkg_path, 'urdf', 'tennis_ball.urdf')

    return LaunchDescription([
        ExecuteProcess(
            cmd=[
                FindExecutable(name='gazebo'),
                '--verbose',
                '-s', 'libgazebo_ros_factory.so'
            ],
            output='screen'
        ),
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-file', urdf_path,
                       '-entity', 'tennis_ball',
                       '-x', '0.3', '-y', '0.2', '-z', '0.9'],
            output='screen'
        )
    ])
