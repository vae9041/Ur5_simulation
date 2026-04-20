from setuptools import setup, find_packages

package_name = 'grasp_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=[package_name]),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/Model', ['grasp_detection/Model/best_model.pth']), 
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='vae9041@g.rit.edu',
    description='Faster R-CNN based grasp detection node.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'faster_rcnn_node = grasp_detection.faster_rcnn_node:main',
            'pick_executor = grasp_detection.pick_executor:main',
            'sulabh_grasp_detection = grasp_detection.sulabh_grasp_detection:main',
        ],
    },
)

