from setuptools import setup
import os
from glob import glob

package_name = 'thorvald_gui'

setup(
    name=package_name,
    version='2.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'plugin.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Gbrtenorio',
    maintainer_email='gbr.tenorio@gmail.com',
    description='RQT monitoring panel for Thorvald II',
    license='BSD',
    entry_points={
        'console_scripts': [
            'thorvald_gui = thorvald_gui.main:main',
        ],
    },
)
