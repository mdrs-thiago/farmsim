#include <math.h>
#include <unistd.h>
#include <sstream>
#include <iostream>

#include <thorvald_can_devices/CANPeakSysUSB.h>
#include <thorvald_base_driver/can_ctrl_pltf.h>

CanCtrlPltf::CanCtrlPltf()
{
  can_ctrl_ = nullptr;
}

CanCtrlPltf::~CanCtrlPltf()
{
  if (can_ctrl_ != nullptr) {
    delete can_ctrl_;
  }

  CanDriveItf * prev_drive = nullptr;
  for (unsigned int i = 0; i < motor_controllers_.size(); i++) {
    if (motor_controllers_[i] != nullptr && motor_controllers_[i] != prev_drive) {
      delete motor_controllers_[i];
      prev_drive = motor_controllers_[i];
    }
  }
}

int CanCtrlPltf::evalCanBuffer(
  std::vector<thorvald_msgs::msg::CANFrame> & can_msgs_base,
  std::vector<thorvald_msgs::msg::CANFrame> & can_msgs_device)
{
  bool b_ret;
  mutex_.lock();

  can_msgs_base.clear();
  can_msgs_device.clear();

  while (can_ctrl_->receiveMsg(&can_msg_rec_) == true) {
    b_ret = false;

    for (unsigned int i = 0; i < batteries_.size(); i++) {
      b_ret |= batteries_[i]->evalReceivedMsg(can_msg_rec_);
    }

    for (unsigned int i = 0; i < motor_controllers_.size(); i++) {
      b_ret |= motor_controllers_[i]->evalReceivedMsg(can_msg_rec_);
    }

    int length = can_msg_rec_.m_iLen;
    thorvald_msgs::msg::CANFrame t_frame;
    t_frame.id = can_msg_rec_.m_iID;
    t_frame.length = length;
    for (int i = 0; i < length; i++) {
      t_frame.data.push_back(can_msg_rec_.getAt(i));
    }

    if (b_ret) {
      can_msgs_base.push_back(t_frame);
    } else {
      can_msgs_device.push_back(t_frame);
    }
  }

  mutex_.unlock();
  return 0;
}

void CanCtrlPltf::initMotorDrives(std::vector<std::map<std::string, double>> motor_drives)
{
  number_of_drives_ = static_cast<int>(motor_drives.size());
  number_of_motor_controllers_ = 0;
  motor_controllers_.resize(number_of_drives_);

  std::cout << "Assuming the following base hardware:" << std::endl;
  for (int i = 0; i < number_of_drives_; i++) {
    if (static_cast<int>(motor_drives[i]["type"]) == 0) {
      std::cout << "  Roboteq FBL2360, prop/steer, node ID: " << motor_drives[i]["node"] << std::endl;
      motor_controllers_[i] = new ::RoboteqFbl2360();
      auto * drv = static_cast<::RoboteqFbl2360 *>(motor_controllers_[i]);
      drv->setCanOpenParam(static_cast<int>(motor_drives[i]["node"]));
      drv->setCanItf(can_ctrl_);
      drv->setControllerIndex(i);
      drv->setDriveParams(0, motor_drives[i]);
      number_of_motor_controllers_++;
    } else if (static_cast<int>(motor_drives[i]["type"]) == 1) {
      if (i + 1 < number_of_drives_ &&
          motor_drives[i + 1]["node"] == motor_drives[i]["node"] &&
          motor_drives[i + 1]["type"] == motor_drives[i]["type"])
      {
        std::cout << "  Roboteq FBL2360, prop/prop, node ID: " << motor_drives[i]["node"] << std::endl;
        motor_controllers_[i] = new ::RoboteqFbl2360Diff();
        auto * drv = static_cast<::RoboteqFbl2360Diff *>(motor_controllers_[i]);
        drv->setCanOpenParam(static_cast<int>(motor_drives[i]["node"]));
        drv->setCanItf(can_ctrl_);
        drv->setChannelOneId(i);
        drv->setDriveParams(i, motor_drives[i]);

        motor_controllers_[i + 1] = motor_controllers_[i];
        static_cast<::RoboteqFbl2360 *>(motor_controllers_[i + 1])->setDriveParams(i + 1, motor_drives[i + 1]);
        number_of_motor_controllers_++;
        i++;
      } else {
        std::cerr << "Cannot create RoboteqFbl2360Diff: drives must share node ID and type." << std::endl;
      }
    }
  }
}

void CanCtrlPltf::initBatteries(std::vector<std::map<std::string, double>> batteries)
{
  std::cout << "Assuming the following batteries:" << std::endl;
  for (unsigned int i = 0; i < batteries.size(); i++) {
    int type = static_cast<int>(batteries[i]["type"]);
    if (type == 0) {
      std::cout << "  Empty battery enclosure" << std::endl;
    } else if (type == 1) {
      std::cout << "  KC Electronics, node ID: " << batteries[i]["node"] << std::endl;
      ::KCBattery * t_battery = new ::KCBattery();
      t_battery->setCanParam(0x321);
      t_battery->setCanItf(can_ctrl_);
      batteries_.push_back(t_battery);
    } else {
      std::cerr << "  UNKNOWN BATTERY TYPE: " << type << std::endl;
    }
  }
}

bool CanCtrlPltf::initPltf(
  int can_adapter_type,
  std::vector<std::map<std::string, double>> drive_map,
  std::vector<std::map<std::string, double>> battery_map)
{
  bool connected_to_can = true;

  if (can_adapter_type == CAN_ADAPTER_NONE) {
    std::cout << "Uses NO CAN adapter" << std::endl;
    connected_to_can = false;
  } else if (can_adapter_type == CAN_ADAPTER_PCAN_USB) {
    bool adapter_found = false;
    for (int i = 2; i < 5; i++) {
      std::string dev = "/dev/pcan3" + std::to_string(i);
      std::cout << "Looking for adapter at: " << dev << std::endl;
      can_ctrl_ = new CANPeakSysUSB(dev.c_str(), CAN_BAUD_500K);
      adapter_found = can_ctrl_->init_ret();
      if (adapter_found) { break; }
      delete can_ctrl_;
      can_ctrl_ = nullptr;
    }
    if (adapter_found) {
      std::cout << "Uses CAN-Peak-USB" << std::endl;
    } else {
      std::cerr << "Did not find CAN-Peak-USB" << std::endl;
      return false;
    }
  } else {
    std::cerr << "Unknown CAN adapter specified" << std::endl;
    return false;
  }

  initMotorDrives(drive_map);
  initBatteries(battery_map);

  if (connected_to_can) {
    std::cout << "StartCanOpen" << std::endl;
    sendNetStartCanOpen();
  }

  return true;
}

bool CanCtrlPltf::setParams(std::vector<std::map<std::string, double>> motor_drives)
{
  for (int i = 0; i < number_of_drives_; i++) {
    motor_controllers_[i]->setDriveParams(i, motor_drives[i]);
  }
  return true;
}

void CanCtrlPltf::setDriveParams(int node_id, std::map<std::string, double> params)
{
  motor_controllers_[node_id - 1]->setDriveParams(-1, params);
}

void CanCtrlPltf::sendNetStartCanOpen()
{
  CanMsg msg;
  msg.m_iID  = 0;
  msg.m_iLen = 2;
  msg.set(1, 0, 0, 0, 0, 0, 0, 0);
  can_ctrl_->transmitMsg(msg, false);
  usleep(100000);
}

int CanCtrlPltf::getMotorControllerIndex(int can_id)
{
  for (int i = 0; i < number_of_drives_; i++) {
    if (getMotorControllerCanId(i) == can_id) { return i; }
  }
  return -1;
}

bool CanCtrlPltf::setupMotorController(int can_id, int setup_id, int setup_value)
{
  int index = getMotorControllerIndex(can_id);
  if (index >= 0) {
    motor_controllers_[index]->setupController(setup_id, setup_value);
    return true;
  }
  return false;
}

int CanCtrlPltf::sendCommandsToAllDrives(thorvald_msgs::msg::BaseState commands)
{
  mutex_.lock();
  for (int i = 0; i < number_of_drives_; i++) {
    motor_controllers_[i]->sendDriveCommand(
      i, commands.prop_speed[i], commands.steer_pos[i], commands.steer_max_speed[i]);
  }
  mutex_.unlock();
  return 0;
}

int CanCtrlPltf::sendDeviceCommand(thorvald_msgs::msg::CANFrame can_frame)
{
  CanMsg msg;
  msg.m_iID  = can_frame.id;
  msg.m_iLen = can_frame.length;

  if (can_frame.length != static_cast<int>(can_frame.data.size())) {
    std::cerr << "[CanCtrlPltf] CAN device frame data/length mismatch: "
              << can_frame.length << " != " << can_frame.data.size() << std::endl;
  } else {
    for (int i = 0; i < can_frame.length; i++) {
      msg.setAt(can_frame.data[i], i);
    }
    can_ctrl_->transmitMsg(msg);
  }
  return 0;
}

int CanCtrlPltf::simulateAllDrives(thorvald_msgs::msg::BaseState commands)
{
  for (int i = 0; i < number_of_drives_; i++) {
    motor_controllers_[i]->simulateDrive(i, commands.prop_speed[i], commands.steer_pos[i]);
  }
  return 0;
}

int CanCtrlPltf::setCommand(
  int drive_index, int node_id, double speed, double position, double position_max_speed)
{
  mutex_.lock();
  motor_controllers_[node_id - 1]->sendDriveCommand(drive_index, speed, position, position_max_speed);
  mutex_.unlock();
  return 0;
}

bool CanCtrlPltf::setCurrentPosAsZeroAll()
{
  mutex_.lock();
  bool success = true;
  for (int i = 0; i < number_of_drives_; i++) {
    success &= motor_controllers_[i]->setCurrentPosAsZero();
  }
  mutex_.unlock();
  return success;
}

bool CanCtrlPltf::setCurrentPosAsZero(int can_id)
{
  int index = getMotorControllerIndex(can_id);
  if (index >= 0) {
    mutex_.lock();
    motor_controllers_[index]->setCurrentPosAsZero();
    mutex_.unlock();
    return true;
  }
  return false;
}

bool CanCtrlPltf::homeSteeringAll()
{
  mutex_.lock();
  bool success = true;
  for (int i = 0; i < number_of_drives_; i++) {
    success &= motor_controllers_[i]->homeSteering();
  }
  mutex_.unlock();
  return success;
}

bool CanCtrlPltf::homeSteering(int can_id)
{
  int index = getMotorControllerIndex(can_id);
  if (index >= 0) {
    mutex_.lock();
    motor_controllers_[index]->homeSteering();
    mutex_.unlock();
    return true;
  }
  return false;
}

void CanCtrlPltf::getAllBatteryVars(thorvald_msgs::msg::BatteryArray & batt_array_in)
{
  mutex_.lock();

  batt_array_in.battery_data.resize(batteries_.size());
  for (unsigned int i = 0; i < batteries_.size(); i++) {
    thorvald_msgs::msg::BatteryData batt;
    batt.can_id           = batteries_[i]->getCanId();
    batt.state_of_charge  = batteries_[i]->getStateOfCharge();
    batt.state_of_health  = batteries_[i]->getStateOfHealth();
    batt.battery_voltage  = batteries_[i]->getVoltage();
    batt.battery_current  = batteries_[i]->getCurrent();
    batt.charge_fet_on    = batteries_[i]->getChargeFet();
    batt.discharge_fet_on = batteries_[i]->getDischargeFet();
    batt.cycle_count      = batteries_[i]->getCycleCount();
    batt.battery_capacity = batteries_[i]->getCapacity();
    batt.temperatures     = batteries_[i]->getTemps();

    thorvald_msgs::msg::BatteryAlarms alarms;
    alarms.over_voltage_protection    = batteries_[i]->getAOverVolt();
    alarms.under_voltage_protection   = batteries_[i]->getAUnderVolt();
    alarms.over_temperature_protection  = batteries_[i]->getAOverTemp();
    alarms.under_temperature_protection = batteries_[i]->getAUnderTemp();
    alarms.over_charge_current         = batteries_[i]->getAOverCharge();
    alarms.over_discharge_current      = batteries_[i]->getAOverDischarge();
    alarms.can_communication_fail      = batteries_[i]->getACanFail();
    alarms.cell_voltage_sensing_fault  = batteries_[i]->getACellVoltSensFault();
    alarms.bms_fail                    = batteries_[i]->getABmsFail();
    alarms.short_circuit_discharge     = batteries_[i]->getAShortCircuitDischarge();
    alarms.temperature_sensing_fault   = batteries_[i]->getATempSensFault();
    alarms.over_voltage_fault          = batteries_[i]->getAOverVoltFault();
    alarms.under_voltage_fault         = batteries_[i]->getAUnderVoltFault();
    alarms.pack_voltage_sensing_fault  = batteries_[i]->getAPackVoltSensFault();
    batt.battery_alarms = alarms;

    batt_array_in.battery_data[i] = batt;
  }

  mutex_.unlock();
}

void CanCtrlPltf::requestNextVarsAllBatteries(int /*params_to_update*/)
{
  // Battery polling not implemented in this driver version
}

int CanCtrlPltf::getMotorControllerCanId(int index)
{
  mutex_.lock();
  int can_id = motor_controllers_[index]->getCanId();
  mutex_.unlock();
  return can_id;
}

void CanCtrlPltf::getControllerArray(thorvald_msgs::msg::ControllerArray & controller_array_out)
{
  mutex_.lock();

  controller_array_out.controller_data.clear();
  int i = 0;
  CanDriveItf * prev_controller = nullptr;
  while (i < number_of_drives_) {
    if (motor_controllers_[i] != prev_controller) {
      thorvald_msgs::msg::ControllerData cd;
      cd.controller_config.name           = motor_controllers_[i]->getName();
      cd.controller_config.can_id         = motor_controllers_[i]->getCanId();
      cd.controller_config.drive_direction = motor_controllers_[i]->getDriveDirection();
      cd.controller_config.home_speed     = motor_controllers_[i]->getHomeSpeed();
      cd.controller_config.home_count     = motor_controllers_[i]->getHomeCount();

      cd.controller_state.controller_mode = motor_controllers_[i]->getControllerMode();
      cd.controller_state.battery_volts   = motor_controllers_[i]->getBatteryVolts();
      cd.controller_state.battery_amps    = motor_controllers_[i]->getBatteryAmps();
      cd.controller_state.max_temp        = motor_controllers_[i]->getMaxTemp();
      for (auto b : motor_controllers_[i]->getStatusFlags()) {
        cd.controller_state.status_flags.push_back(b != 0);
      }

      int ctrl_index = i;
      while (motor_controllers_[i]->hasIndex(ctrl_index)) {
        thorvald_msgs::msg::MotorState ms;
        ms.speed         = motor_controllers_[i]->getSpeed(ctrl_index);
        ms.position      = motor_controllers_[i]->getPosition(ctrl_index);
        ms.motor_amps    = motor_controllers_[i]->getMotorAmps(ctrl_index);
        ms.relative_power = motor_controllers_[i]->getRelativePower(ctrl_index);
        cd.motor_state.push_back(ms);
        ctrl_index++;
      }
      controller_array_out.controller_data.push_back(cd);
    }
    prev_controller = motor_controllers_[i];
    i++;
  }

  mutex_.unlock();
}

void CanCtrlPltf::getBaseState(thorvald_msgs::msg::BaseState & base_state)
{
  mutex_.lock();
  base_state.prop_speed.resize(number_of_drives_);
  base_state.prop_pos.resize(number_of_drives_);
  base_state.steer_pos.resize(number_of_drives_);
  base_state.channel.resize(number_of_drives_);

  for (int i = 0; i < number_of_drives_; i++) {
    base_state.prop_speed[i] = motor_controllers_[i]->getPropulsionSpeed(i);
    base_state.prop_pos[i]   = motor_controllers_[i]->getPropulsionPosition(i);
    base_state.steer_pos[i]  = motor_controllers_[i]->getSteeringPosition(i);
  }
  mutex_.unlock();
}

bool CanCtrlPltf::getControllerSetupMap(int can_id, std::vector<std::string> & setup_map)
{
  int index = getMotorControllerIndex(can_id);
  if (index >= 0) {
    mutex_.lock();
    setup_map = motor_controllers_[index]->getControllerSetupMap();
    mutex_.unlock();
    return true;
  }
  return false;
}
