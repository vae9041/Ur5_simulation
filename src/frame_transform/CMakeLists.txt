cmake_minimum_required(VERSION 3.8)
project(frame_transform)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate service interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/FrameTransform.srv"
  DEPENDENCIES geometry_msgs
)

# Install Python script directly (no ament_python_install_package)
install(PROGRAMS
  frame_transform/object_position_camera_base_link.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME object_position_camera_base_link
)

install(PROGRAMS
  frame_transform/joint_states_filter.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME joint_states_filter
)

ament_package()
