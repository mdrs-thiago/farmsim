from setuptools import find_packages, setup

package_name = 'farmsim_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='thiane',
    maintainer_email='thiago.medeiros.carvalho@gmail.com',
    description='Row perception models, label rendering and the perception node.',
    license='BSD',
    entry_points={
        'console_scripts': [
            'perception_node = farmsim_perception.node:main',
            'record_camera_poses = farmsim_perception.labeling.record_camera_poses:main',
            'export_recording = farmsim_perception.labeling.export_recording:main',
            'label_row_center = farmsim_perception.labeling.label_row_center:main',
            'live_label_teleop = farmsim_perception.labeling.live_label_teleop:main',
            'live_segmentation_view = farmsim_perception.labeling.live_segmentation_view:main',
        ],
    },
)
