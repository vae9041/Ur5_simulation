# UR5 RGB-D Robotic Grasping with Faster R-CNN

This repository contains the ROS 2 workspace used for our conference-paper experiments on RGB-D object localization and robotic grasping with a UR5e manipulator. The system integrates Gazebo simulation, MoveIt 2 motion planning, TF-based coordinate transformation, and multiple Faster R-CNN perception nodes based on ResNet-50, ResNet-18, and a structured-pruned ResNet-18 backbone.

## System Overview

The pipeline consists of the following stages:

1. A Gazebo workspace is launched with the UR5e arm, Robotiq gripper, RGB-D camera, and MoveIt 2 configuration.
2. A Faster R-CNN node detects the target object from the RGB stream.
3. The aligned depth image is used to estimate the target position in the camera frame.
4. A TF-based conversion node transforms the detected position into the robot base frame.
5. A pick-and-place node uses MoveIt 2 to plan and execute the grasping motion.
6. `rqt` is used to visualize annotated detections during experiments.

## Software Stack

- ROS 2
- Gazebo
- MoveIt 2
- `ros2_control`
- TF2
- PyTorch
- Torchvision Faster R-CNN
- OpenCV / `cv_bridge`
- Intel RealSense RGB-D sensing model in simulation

## Main Packages

- `ur5e_sim`
  - Gazebo worlds, robot description, launch files, and simulation setup
- `ur5_camera_gripper_moveit_config`
  - MoveIt 2 configuration for the UR5e arm and Robotiq gripper
- `grasp_detection`
  - Faster R-CNN perception node with a ResNet-50 FPN backbone
- `res_grasp_detection`
  - Faster R-CNN perception nodes with a ResNet-18 backbone and a structured-pruned ResNet-18 variant
- `frame_transform`
  - TF-based conversion from camera-frame detections to `base_link`
- `ur5_ik_master_class`
  - Pick-and-place execution logic for simulated grasping
- `IFRA_LinkAttacher`
  - Gazebo link attach/detach support used during simulated grasp execution

## Repository Layout

```text
vince_ros2_ws/
├── src/
│   ├── grasp_detection/
│   ├── res_grasp_detection/
│   ├── frame_transform/
│   ├── ur5e_sim/
│   ├── ur5_camera_gripper_moveit_config/
│   ├── ur5_ik_master_class/
│   └── IFRA_LinkAttacher/
├── build/
├── install/
└── log/
```

## Model Checkpoints

Model checkpoint files are intentionally excluded from the repository to keep the project lightweight for GitHub.

Expected local checkpoint locations are:

- `src/grasp_detection/grasp_detection/Model/best_model.pth`
- `src/res_grasp_detection/res_grasp_detection/Model/best_model.pth`
- `src/res_grasp_detection/res_grasp_detection/Pruned_Model/structured_pruned.pth`

If you store checkpoints elsewhere, update the corresponding node parameters or package paths before running the nodes.

## Build

From the workspace root:

```bash
colcon build
source install/setup.bash
```

## Running the Simulation Pipeline

### 1. Launch the Gazebo world and MoveIt 2 setup

```bash
ros2 launch ur5e_sim spawn_ur5_camera_gripper_lab_world_moveit.launch.py
```

### 2. Run one of the perception nodes

ResNet-50 Faster R-CNN:

```bash
ros2 run grasp_detection faster_rcnn_node
```

ResNet-18 Faster R-CNN:

```bash
ros2 run res_grasp_detection lower_model_node
```

Structured-pruned ResNet-18 Faster R-CNN:

```bash
ros2 run res_grasp_detection pruned_model_node
```

### 3. Run the frame transformation node

```bash
ros2 run frame_transform object_position_camera_base_link
```

### 4. Visualize detections

```bash
rqt
```

Typical visualization topic:

- `/grasp_detection/image_annotated`

### 5. Execute the simulated pick task

```bash
ros2 run ur5_ik lab_sim_attachlink
```

## Notes

- The ResNet-50 node is implemented in `grasp_detection/faster_rcnn_node.py`.
- The ResNet-18 and pruned ResNet-18 nodes are implemented in `res_grasp_detection/`.
- The perception nodes publish annotated images and object positions in the camera frame.
- The transformation node converts the detected target into the robot base frame for planning.
- The pick node uses MoveIt 2 together with Gazebo link attachment for simulated grasp completion.

## Demo Video

A montage of the grasping execution is available here:

[Watch the grasping execution montage](https://drive.google.com/file/d/1xQ_WxqXcIqmu-JwPXTSn9yqCD7iW_jOo/view?usp=sharing)

## Intended Use

This repository is prepared for academic dissemination and accompanies a conference-paper workflow demonstrating end-to-end robotic grasping with RGB-D perception, coordinate transformation, and motion execution in simulation.
