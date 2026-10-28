from setuptools import setup

package_name = 'thorvald_base'

setup(
    name=package_name,
    version='0.2.2',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TODO',
    maintainer_email='todo@todo.com',
    description='Thorvald II base platform driver and kinematics (ROS2 Humble)',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'twist_to_joints = thorvald_base.twist_to_joints:main',
        ],
    },
)
