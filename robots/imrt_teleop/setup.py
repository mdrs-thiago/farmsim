from setuptools import setup

package_name = "imrt_teleop"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="NMBU Robotics",
    maintainer_email="noreply@nmbu.no",
    description="ROS2 teleoperation node converting Joy to Twist commands.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "imrt_teleop = imrt_teleop.imrt_teleop:main",
        ],
    },
)
