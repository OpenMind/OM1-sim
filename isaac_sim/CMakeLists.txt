cmake_minimum_required(VERSION 3.8)
project(isaac_sim)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  lowstate_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME lowstate_node
)

ament_package()
