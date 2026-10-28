import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'farmsim_farm'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test', 'tools']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='thiane',
    maintainer_email='thiago.medeiros.carvalho@gmail.com',
    description='Surveyed farm fields on gz-sim: worlds, rows, paths.',
    license='BSD',
    entry_points={
        'console_scripts': [
            'capture_topdown = farmsim_farm.capture_topdown:main',
            'mark_rows = farmsim_farm.mark_rows:main',
            'segment_rows = farmsim_farm.segment_rows:main',
            'annotate_plots = farmsim_farm.annotate_plots:main',
            'spawn_row_markers = farmsim_farm.spawn_row_markers:main',
            'polygon_to_centerline = farmsim_farm.polygon_to_centerline:main',
            'plot_lattice_to_rows = farmsim_farm.plot_lattice_to_rows:main',
        ],
    },
)
