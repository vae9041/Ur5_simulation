cmake_minimum_required(VERSION 3.8)
project(grasping_objects)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(xacro REQUIRED)
find_package(gazebo_ros REQUIRED)

# Install Python files (if you have any)
install(
  PROGRAMS
  # scripts/my_script.py  # <-- uncomment and replace if you add scripts
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch directory
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install urdf or xacro files if needed
install(DIRECTORY
  urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
