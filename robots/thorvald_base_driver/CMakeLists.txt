cmake_minimum_required(VERSION 3.8)
project(thorvald_base_driver)

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
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(thorvald_msgs REQUIRED)
find_package(thorvald_can_devices REQUIRED)

set(plugin_deps
  rclcpp
  geometry_msgs
  nav_msgs
  thorvald_msgs
  pluginlib
  tf2_geometry_msgs
)

set(driver_deps
  rclcpp
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  std_srvs
  tf2_ros
  tf2_geometry_msgs
  pluginlib
  thorvald_msgs
  thorvald_can_devices
)

# Kinematics plugin library (header-only impl; this .cpp just exports the class)
add_library(pltf_clc_std SHARED src/pltf_clc_std_plugin.cpp)
target_include_directories(pltf_clc_std PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
link_ament_deps(pltf_clc_std ${plugin_deps})
pluginlib_export_plugin_description_file(thorvald_base_driver plugins.xml)

# CAN platform controller library
add_library(can_ctrl_pltf SHARED src/can_ctrl_pltf.cpp)
target_include_directories(can_ctrl_pltf PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
link_ament_deps(can_ctrl_pltf ${driver_deps})

# Base driver executable
add_executable(base_driver src/base_driver.cpp)
target_include_directories(base_driver PRIVATE include)
link_ament_deps(base_driver ${driver_deps})
target_link_libraries(base_driver can_ctrl_pltf)

install(TARGETS pltf_clc_std can_ctrl_pltf base_driver
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/ DESTINATION include)
install(FILES plugins.xml DESTINATION share/${PROJECT_NAME})

ament_package()
