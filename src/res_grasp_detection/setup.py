import glob
import os
from setuptools import find_packages, setup

package_name = 'res_grasp_detection'
model_files = sorted(glob.glob(os.path.join(package_name, 'Model', '*.pth')))
pruned_model_files = sorted(glob.glob(os.path.join(package_name, 'Pruned_Model', '*.pth')))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'Model'), model_files),
        (os.path.join('share', package_name, 'Pruned_Model'), pruned_model_files),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'lower_model_node = res_grasp_detection.lower_model_node:main',
            'pruned_model_node = res_grasp_detection.pruned_model_node:main',
        ],
    },
)
