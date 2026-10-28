import os
from xml.dom import Node as DomNode

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def _remove_comments_and_blank_text(node):
    for child in list(node.childNodes):
        if child.nodeType == DomNode.COMMENT_NODE:
            node.removeChild(child)
        elif child.nodeType == DomNode.TEXT_NODE and not child.data.strip():
            node.removeChild(child)
        else:
            _remove_comments_and_blank_text(child)


def _robot_description(context):
    xacro_file = os.path.join(
        get_package_share_directory("thorvald_model"),
        "urdf",
        "thorvald_model.xacro",
    )
    mappings = {
        "robot_name": LaunchConfiguration("robot_name").perform(context),
        "tf_prefix": LaunchConfiguration("tf_prefix").perform(context),
        "robot_config": LaunchConfiguration("robot_model").perform(context),
        "model_extras": LaunchConfiguration("model_extras").perform(context),
    }
    doc = xacro.process_file(xacro_file, mappings=mappings)
    _remove_comments_and_blank_text(doc)
    return doc.documentElement.toxml()


def _launch_setup(context, *args, **kwargs):
    robot_description = _robot_description(context)

    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[{
            "robot_description": robot_description,
            "robot_tf_prefix": LaunchConfiguration("tf_prefix"),
        }],
        output="screen")

    return [rsp_node]


def generate_launch_description():
    # launch arguments
    robot_name_arg = DeclareLaunchArgument("robot_name", default_value="", description="Robot Name")
    tf_prefix_arg = DeclareLaunchArgument("tf_prefix",   default_value="", description="Prefix for TF frames")

    robot_model_arg = DeclareLaunchArgument("robot_model",
        default_value=PathJoinSubstitution([FindPackageShare("thorvald_model"), "config", "thorvald_ii_4wd4ws_std", "thorvald_ii_4wd4ws_std.yaml"]),
        description="Thorvald robot configuration file")
    model_extras_arg = DeclareLaunchArgument("model_extras",
        default_value=PathJoinSubstitution([FindPackageShare("thorvald_model"), "urdf", "sensor_modules", "sensor_frames_example.xacro"]),
        description="Thorvald attachements file")

    return LaunchDescription([
        robot_name_arg,
        tf_prefix_arg,
        robot_model_arg,
        model_extras_arg,
        OpaqueFunction(function=_launch_setup),])
