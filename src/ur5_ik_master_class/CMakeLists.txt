cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0074 NEW)
project(ur5_ik)


if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(gazebo_msgs REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(PCL REQUIRED)

find_package(linkattacher_msgs REQUIRED)  
find_package(ros2_linkattacher REQUIRED)
find_package(frame_transform REQUIRED)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(ur5_ik_test src/ur5_ik_test.cpp)
ament_target_dependencies(ur5_ik_test rclcpp moveit_ros_planning_interface gazebo_msgs sensor_msgs pcl_conversions pcl_ros PCL geometry_msgs)

add_executable(pick_place src/pick_place.cpp)
ament_target_dependencies(pick_place rclcpp moveit_ros_planning_interface gazebo_msgs linkattacher_msgs ros2_linkattacher sensor_msgs pcl_conversions pcl_ros PCL geometry_msgs)

add_executable(pick_place_service src/pick_place_service.cpp)
ament_target_dependencies(pick_place_service rclcpp moveit_ros_planning_interface frame_transform geometry_msgs tf2_geometry_msgs gazebo_msgs linkattacher_msgs ros2_linkattacher sensor_msgs pcl_conversions pcl_ros PCL)

add_executable(pick_place_camera src/pick_place_camera.cpp)
ament_target_dependencies(pick_place_camera rclcpp moveit_ros_planning_interface gazebo_msgs linkattacher_msgs ros2_linkattacher sensor_msgs pcl_conversions pcl_ros PCL geometry_msgs)

add_executable(inverseKinematics src/inverseKinematics.cpp)
ament_target_dependencies(inverseKinematics rclcpp moveit_ros_planning_interface gazebo_msgs linkattacher_msgs ros2_linkattacher sensor_msgs pcl_conversions pcl_ros PCL geometry_msgs)

add_executable(real_pnp src/real_pnp.cpp)
ament_target_dependencies(real_pnp rclcpp moveit_ros_planning_interface geometry_msgs tf2_geometry_msgs)

add_executable(lab_sim src/lab_sim.cpp)
ament_target_dependencies(lab_sim rclcpp moveit_ros_planning_interface geometry_msgs tf2_geometry_msgs frame_transform)

add_executable(lab_sim_attachlink src/lab_sim_attachlink.cpp)
ament_target_dependencies(lab_sim_attachlink rclcpp moveit_ros_planning_interface geometry_msgs tf2_geometry_msgs frame_transform linkattacher_msgs ros2_linkattacher)

install(TARGETS 
  ur5_ik_test pick_place pick_place_service pick_place_camera inverseKinematics real_pnp lab_sim lab_sim_attachlink
  DESTINATION lib/${PROJECT_NAME}/
)

ament_package()
