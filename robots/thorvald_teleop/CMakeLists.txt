cmake_minimum_required(VERSION 3.8)
project(thorvald_teleop)

# link_ament_deps() was removed in recent ROS 2 releases; link the
# packages' exported targets instead (plain signature, like the rest of this file).
macro(link_ament_deps target)
  foreach(dep ${ARGN})
    if(TARGET ${dep}::${dep})
      target_link_libraries(${target} ${dep}::${dep})
    else()
      target_link_libraries(${target} ${${dep}_TARGETS})
    endif()
  endforeach()
endmacro()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

add_executable(teleop_node src/teleop_node.cpp)
link_ament_deps(teleop_node
  rclcpp geometry_msgs sensor_msgs std_msgs std_srvs)

install(TARGETS teleop_node
  RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME})

ament_package()
