import os
from glob import glob

from setuptools import setup

package_name = "imrt_virtual_joy"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*launch.[pxy][yma]*"))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="NMBU Robotics",
    maintainer_email="noreply@nmbu.no",
    description="ROS2 package providing a virtual gamepad that publishes sensor_msgs/Joy messages.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "gamepad_talker = imrt_virtual_joy.gamepad_talker:main",
        ],
    },
)
