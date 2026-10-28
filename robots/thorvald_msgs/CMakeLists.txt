cmake_minimum_required(VERSION 3.8)
project(thorvald_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BaseState.msg"
  "msg/BatteryAlarms.msg"
  "msg/BatteryArray.msg"
  "msg/BatteryData.msg"
  "msg/CANFrame.msg"
  "msg/ControllerArray.msg"
  "msg/ControllerConfig.msg"
  "msg/ControllerData.msg"
  "msg/ControllerInfo.msg"
  "msg/ControllerState.msg"
  "msg/DriveParam.msg"
  "msg/EnclosureState.msg"
  "msg/MotorState.msg"
  "msg/ThorvaldIO.msg"
  "srv/CanID.srv"
  "srv/DigitalWrite.srv"
  "srv/DriveParams.srv"
  "srv/MotorControllerSetup.srv"
  "srv/SwitchBool.srv"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
