#ifndef CANCTRLPLTFCOB3_INCLUDEDEF_H
#define CANCTRLPLTFCOB3_INCLUDEDEF_H

#include <thorvald_can_devices/CanDriveItf.h>
#include <thorvald_can_devices/RoboteqFbl2360.h>
#include <thorvald_can_devices/RoboteqFbl2360Diff.h>
#include <thorvald_can_devices/kc_battery.h>
#include <thorvald_can_devices/CanItf.h>
#include <thorvald_msgs/msg/base_state.hpp>
#include <thorvald_msgs/msg/controller_array.hpp>
#include <thorvald_msgs/msg/battery_array.hpp>
#include <thorvald_msgs/msg/can_frame.hpp>
#include <map>
#include <mutex>
#include <string>
#include <vector>

class CanCtrlPltf
{
public:
  CanCtrlPltf();
  ~CanCtrlPltf();

  enum CanAdapters
  {
    CAN_ADAPTER_NONE,
    CAN_ADAPTER_PCAN_USB,
  };

  bool initPltf(
    int can_adapter_type,
    std::vector<std::map<std::string, double>> motor_drives,
    std::vector<std::map<std::string, double>> battery_map);

  bool setParams(std::vector<std::map<std::string, double>> motor_drives);

  void setDriveParams(int node_id, std::map<std::string, double> params);

  int sendCommandsToAllDrives(thorvald_msgs::msg::BaseState commands);

  int sendDeviceCommand(thorvald_msgs::msg::CANFrame can_frame);

  int simulateAllDrives(thorvald_msgs::msg::BaseState commands);

  int setCommand(int drive_index, int node_id, double speed, double position, double pos_max_vel);

  bool setupMotorController(int can_id, int setup_id, int setup_value);

  bool setCurrentPosAsZeroAll();
  bool setCurrentPosAsZero(int can_id);

  bool homeSteeringAll();
  bool homeSteering(int can_id);

  int evalCanBuffer(
    std::vector<thorvald_msgs::msg::CANFrame> & can_msgs_base,
    std::vector<thorvald_msgs::msg::CANFrame> & can_msgs_device);

  int getMotorControllerCanId(int index);
  int getMotorControllerIndex(int can_id);

  void getBaseState(thorvald_msgs::msg::BaseState & base_state);
  void getControllerArray(thorvald_msgs::msg::ControllerArray & controller_array_out);
  void getAllBatteryVars(thorvald_msgs::msg::BatteryArray & batt_array_in);

  bool getControllerSetupMap(int can_id, std::vector<std::string> & setup_map);

  void requestNextVarsAllBatteries(int params_to_update);

protected:
  void initMotorDrives(std::vector<std::map<std::string, double>> drive_map);
  void initBatteries(std::vector<std::map<std::string, double>> battery_map);
  void sendNetStartCanOpen();

  CanMsg can_msg_rec_;
  std::mutex mutex_;

  CanItf * can_ctrl_;

  int number_of_drives_;
  int number_of_motor_controllers_;

  std::vector<CanDriveItf *> motor_controllers_;
  std::vector<KCBattery *> batteries_;
};

#endif  // CANCTRLPLTFCOB3_INCLUDEDEF_H
