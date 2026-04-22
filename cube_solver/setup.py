from glob import glob

from setuptools import find_packages, setup

package_name = 'cube_solver'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='junsoo',
    maintainer_email='junsoo@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'cube_perception_node = cube_solver.cube_perception_node:main',
            'cube_master_node = cube_solver.cube_master_node:main',
            'gripper_control_node = cube_solver.gripper_control_node:main',
            'drl_block_runner = cube_solver.drl_block_runner:main',
            'perception_sequence_runner = cube_solver.perception_sequence_runner:main',
            'perception_timed_capture_runner = cube_solver.perception_timed_capture_runner:main',
            'perception_workflow_coordinator = cube_solver.perception_workflow_coordinator:main',
        ],
    },
)
