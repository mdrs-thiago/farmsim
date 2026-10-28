cmake_minimum_required(VERSION 3.8)
project(thorvald_io_serial)

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
find_package(thorvald_msgs REQUIRED)

add_executable(thorvald_io_arduino src/thorvald_io_arduino.cpp)
target_include_directories(thorvald_io_arduino PRIVATE include)
link_ament_deps(thorvald_io_arduino rclcpp thorvald_msgs)

install(TARGETS thorvald_io_arduino
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

ament_package()
