from setuptools import find_packages, setup

package_name = 'farmsim_control'

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
    description='Row-following controllers and the path-following node.',
    license='BSD',
    entry_points={'console_scripts': ['controller_node = farmsim_control.node:main']},
)
