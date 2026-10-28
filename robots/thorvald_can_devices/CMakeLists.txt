cmake_minimum_required(VERSION 3.8)
project(thorvald_can_devices)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

include_directories(include)

add_library(thorvald_can_devices_roboteq SHARED
  src/RoboteqFbl2360.cpp
  src/RoboteqFbl2360Diff.cpp
)
target_include_directories(thorvald_can_devices_roboteq PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_library(thorvald_can_devices_kc SHARED
  src/kc_battery.cpp
)
target_include_directories(thorvald_can_devices_kc PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

install(TARGETS thorvald_can_devices_roboteq thorvald_can_devices_kc
  EXPORT export_thorvald_can_devices
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

ament_export_targets(export_thorvald_can_devices HAS_LIBRARY_TARGET)
ament_export_include_directories(include)

ament_package()
