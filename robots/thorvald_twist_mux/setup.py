from setuptools import setup
import os
from glob import glob

package_name = 'thorvald_twist_mux'

setup(
    name=package_name,
    version='2.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Gbrtenorio',
    maintainer_email='gbr.tenorio@gmail.com',
    description='Priority-based velocity multiplexer for Thorvald II',
    license='BSD',
    entry_points={
        'console_scripts': [
            'twist_mux_node = thorvald_twist_mux.twist_mux_node:main',
        ],
    },
)
