#include <assert.h>
#include <thorvald_can_devices/RoboteqFbl2360Diff.h>
#include <unistd.h>
#include <cmath>
#include <map>
#include <iostream>
#include <chrono>

static inline double elapsed_sec(
  std::chrono::steady_clock::time_point start,
  std::chrono::steady_clock::time_point end)
{
  return std::chrono::duration<double>(end - start).count();
}

RoboteqFbl2360Diff::RoboteqFbl2360Diff()
{
  can_ctrl_ = NULL;
  node_id_ = -1;

  drive_direction_ = 0;
  home_speed_ = 0;
  home_count_ = 0;

  controller_mode_ = MODE_PRE_INITIATION;
  battery_volts_ = 0;
  battery_amps_ = 0;
  max_temp_ = 0;
  status_flags_.resize(NUMBER_OF_STATUS_FLAGS, 0);
  status_flags_old_ = 0;

  m1_vel_ = 0; m1_pos_ = 0; m1_motor_amps_ = 0; m1_relative_power_ = 0;
  m1_sim_command_received_ = false; m1_latest_sim_command_ = 0;
  m1_command_received_ = false; m1_latest_command_ = 0;

  m2_vel_ = 0; m2_pos_ = 0; m2_motor_amps_ = 0; m2_relative_power_ = 0;
  m2_sim_command_received_ = false; m2_latest_sim_command_ = 0;
  m2_command_received_ = false; m2_latest_command_ = 0;

  command_seq_ = 0;
  last_heartbeat_time_ = time(NULL);
  heartbeat_timeout_time_ = 2;
  drive_inverted_ = false;

  previous_simulation_timestep_ = std::chrono::steady_clock::now();
}

void RoboteqFbl2360Diff::updateWheelPos(double speed_1, double speed_2, double dt)
{
  m1_pos_ += (speed_1 / m1_params_.radius_wheel) * dt;
  m2_pos_ += (speed_2 / m2_params_.radius_wheel) * dt;
}

void RoboteqFbl2360Diff::setDriveParams(int channel, std::map<std::string, double> param_map)
{
  if (channel == m1_id_) {
    name_ = param_map["name"];
    m1_params_.radius_wheel = param_map["r_wheel"];
    m1_params_.gear_ratio = param_map["prp_gr_rt"];
    m1_params_.command_scaling_factor = 1000. / param_map["prp_max_rpm"];
    m1_params_.sim_a = param_map["prp_sim_a"];
  } else if (channel == m2_id_) {
    m2_params_.radius_wheel = param_map["r_wheel"];
    m2_params_.gear_ratio = param_map["prp_gr_rt"];
    m2_params_.command_scaling_factor = 1000. / param_map["prp_max_rpm"];
    m2_params_.sim_a = param_map["prp_sim_a"];
  }
}

const std::string RoboteqFbl2360Diff::CONTROLLER_SETUP_MAP[] = {
  "print setup map", "drive direction (permanent)", "home count (permanent)",
  "home speed (permanent)", "max propulsion speed", "max propulsion acceleration",
  "max steering speed", "max steering acceleration",
  "tpdo1 cycle time", "tpdo2 cycle time", "tpdo3 cycle time", "tpdo4 cycle time"
};

void RoboteqFbl2360Diff::setCanOpenParam(int node_id)
{
  node_id_ = node_id;
  can_headers_.iTxPDO1 = 0x180 + node_id_;
  can_headers_.iTxPDO2 = 0x280 + node_id_;
  can_headers_.iTxPDO3 = 0x380 + node_id_;
  can_headers_.iTxPDO4 = 0x480 + node_id_;
  can_headers_.iRxPDO1 = 0x200 + node_id_;
  can_headers_.iRxPDO2 = 0x300 + node_id_;
  can_headers_.iRxPDO3 = 0x400 + node_id_;
  can_headers_.iRxPDO4 = 0x500 + node_id_;
  can_headers_.heartbeat = 0x700 + node_id_;
  can_headers_.iTxSDO = 0x580 + node_id_;
  can_headers_.iRxSDO = 0x600 + node_id_;
}

void RoboteqFbl2360Diff::simulateDrive(
  int channel, double speed_command, double position_command, bool invert_as_needed)
{
  double dir = 0;
  if (position_command == 0) dir = 1;
  else if (fabs(position_command) > M_PI - 0.01) dir = -1;

  if (channel == m1_id_) {
    m1_latest_sim_command_ = speed_command * dir;
    m1_sim_command_received_ = true;
  } else if (channel == m2_id_) {
    m2_latest_sim_command_ = speed_command * dir;
    m2_sim_command_received_ = true;
  }

  if (m1_sim_command_received_ && m2_sim_command_received_) {
    auto now = std::chrono::steady_clock::now();
    double dt = elapsed_sec(previous_simulation_timestep_, now);
    if (dt > 0.5) dt = 0;

    double error_prop_1 = m1_latest_sim_command_ - m1_vel_;
    if (error_prop_1 != 0) {
      if (fabs(error_prop_1) < m1_params_.sim_a * dt)
        m1_vel_ = m1_latest_sim_command_;
      else
        m1_vel_ += m1_params_.sim_a * dt * (error_prop_1 / fabs(error_prop_1));
    }

    double error_prop_2 = m2_latest_sim_command_ - m2_vel_;
    if (error_prop_2 != 0) {
      if (fabs(error_prop_2) < m2_params_.sim_a * dt)
        m2_vel_ = m2_latest_sim_command_;
      else
        m2_vel_ += m2_params_.sim_a * dt * (error_prop_2 / fabs(error_prop_2));
    }

    updateWheelPos(m1_vel_, m2_vel_, dt);
    m1_sim_command_received_ = false;
    m2_sim_command_received_ = false;
    previous_simulation_timestep_ = now;
  }
}

bool RoboteqFbl2360Diff::evalReceivedMsg(CanMsg & msg)
{
  bool b_ret = false;
  last_can_msg_ = msg;

  int temp_data1, temp_data2, temp_data3, temp_data4;

  if (msg.m_iID == can_headers_.heartbeat)
    last_heartbeat_time_ = time(NULL);

  if (msg.m_iID == can_headers_.iTxPDO1) {
    auto now = std::chrono::steady_clock::now();
    double dt = elapsed_sec(previous_simulation_timestep_, now);
    if (dt > 0.5) dt = 0;

    temp_data1 = (msg.getAt(1) << 8) | msg.getAt(0);
    m1_vel_ = (double)((short)temp_data1) / m1_params_.gear_ratio * M_PI / 30. * m1_params_.radius_wheel;

    temp_data2 = (msg.getAt(3) << 8) | msg.getAt(2);
    (void)temp_data2;

    temp_data3 = (msg.getAt(7) << 24) | (msg.getAt(6) << 16) | (msg.getAt(5) << 8) | msg.getAt(4);
    m2_vel_ = (double)((short)temp_data3) / m2_params_.gear_ratio * M_PI / 30. * m2_params_.radius_wheel;

    updateWheelPos(m1_vel_, m2_vel_, dt);
    previous_simulation_timestep_ = now;
    b_ret = true;
  } else if (msg.m_iID == can_headers_.iTxPDO2) {
    temp_data1 = (msg.getAt(1) << 8) | msg.getAt(0);
    temp_data2 = (msg.getAt(3) << 8) | msg.getAt(2);
    temp_data3 = (msg.getAt(5) << 8) | msg.getAt(4);
    temp_data4 = (msg.getAt(7) << 8) | msg.getAt(6);
    m1_motor_amps_ = (short)temp_data1 / 10.;
    m2_motor_amps_ = (short)temp_data2 / 10.;
    m1_relative_power_ = (short)temp_data3;
    m2_relative_power_ = (short)temp_data4;
    b_ret = true;
  } else if (msg.m_iID == can_headers_.iTxPDO3) {
    temp_data1 = msg.getAt(0);
    temp_data2 = (msg.getAt(3) << 8) | msg.getAt(2);
    temp_data3 = (msg.getAt(5) << 8) | msg.getAt(4);
    temp_data4 = (msg.getAt(7) << 8) | msg.getAt(6);
    char flags_char = (signed char)temp_data1;
    max_temp_ = (short)temp_data2;
    battery_amps_ = (short)temp_data3 / 10.;
    battery_volts_ = (short)temp_data4 / 10.;
    for (int i = 0; i < NUMBER_OF_STATUS_FLAGS; i++) {
      status_flags_[i] = flags_char & static_cast<char>(0b10000000);
      flags_char = flags_char << 1;
    }
    b_ret = true;
  } else if (msg.m_iID == can_headers_.iTxPDO4) {
    temp_data1 = msg.getAt(0);
    temp_data2 = msg.getAt(1);
    temp_data3 = (msg.getAt(3) << 8) | msg.getAt(2);
    temp_data4 = (msg.getAt(7) << 24) | (msg.getAt(6) << 16) | (msg.getAt(5) << 8) | msg.getAt(4);
    controller_mode_ = (signed char)temp_data1;
    drive_direction_ = (signed char)temp_data2;
    home_speed_ = (short)temp_data3;
    home_count_ = temp_data4;
    b_ret = true;
  }

  if (msg.m_iID == can_headers_.iTxSDO) {
    std::cout << "SDO received, ID: " << msg.m_iID << std::endl;
    b_ret = true;
  }

  return b_ret;
}

void RoboteqFbl2360Diff::sendHeartbeat()
{
  CanMsg msg;
  msg.m_iID  = 0x700;
  msg.m_iLen = 5;
  msg.set(0x00, 0, 0, 0, 0, 0, 0, 0);
  can_ctrl_->transmitMsg(msg);
}

bool RoboteqFbl2360Diff::setCurrentPosAsZero()
{
  int pos_counts = m2_pos_ * (m2_params_.encoder_ppr * 4. * m2_params_.gear_ratio) / (2. * M_PI);
  setupController(2, home_count_ - pos_counts);
  return true;
}

void RoboteqFbl2360Diff::setupController(int setup_id, int setup_value)
{
  sendPDOdownload(can_headers_.iRxPDO2, setup_id, setup_value);
}

void RoboteqFbl2360Diff::sendDriveCommand(
  int channel, double speed, double position, double position_max_speed, bool invert_as_needed)
{
  double dir = 0;
  if (position == 0) dir = 1;
  else if (fabs(position) > M_PI - 0.01) dir = -1;

  if (channel == m1_id_) { m1_latest_command_ = speed * dir; m1_command_received_ = true; }
  else if (channel == m2_id_) { m2_latest_command_ = speed * dir; m2_command_received_ = true; }

  if (m1_command_received_ && m2_command_received_) {
    int speed_command_1 = (m1_latest_command_ / m1_params_.radius_wheel) *
                          m1_params_.gear_ratio * (30. / M_PI) * m1_params_.command_scaling_factor;
    int speed_command_2 = (m2_latest_command_ / m2_params_.radius_wheel) *
                          m2_params_.gear_ratio * (30. / M_PI) * m2_params_.command_scaling_factor;
    sendPDOdownload(can_headers_.iRxPDO1, command_seq_, speed_command_1);
    sendPDOdownload(can_headers_.iRxPDO4, command_seq_, speed_command_2);
    command_seq_++;
    m1_command_received_ = false;
    m2_command_received_ = false;
  }
  (void)position_max_speed; (void)invert_as_needed;
}

void RoboteqFbl2360Diff::driveMode() { sendModeCommand(MODE_DRIVE); }
bool RoboteqFbl2360Diff::homeSteering() { sendModeCommand(MODE_HOME); return true; }
void RoboteqFbl2360Diff::decouple() { sendModeCommand(MODE_DECOUPLED); }
void RoboteqFbl2360Diff::sendModeCommand(int mode)
{
  std::cout << "Sending mode: " << mode << std::endl;
  sendPDOdownload(can_headers_.iRxPDO3, mode, mode);
}

void RoboteqFbl2360Diff::sendAccelerationSDO(int channel, int value)
{
  if (channel == 1) sendSDODownload(0x2006, 0x01, value);
  else if (channel == 2) sendSDODownload(0x2006, 0x02, value);
}

void RoboteqFbl2360Diff::sendDecelerationSDO(int channel, int value)
{
  if (channel == 1) sendSDODownload(0x2007, 0x01, value);
  else if (channel == 2) sendSDODownload(0x2007, 0x02, value);
}

void RoboteqFbl2360Diff::sendPDOdownload(int pdoNum, int iData1, int iData2)
{
  char cInt1[4], cInt2[4];
  CanMsg CMsgTr;
  CMsgTr.m_iType = 0;
  CMsgTr.m_iID = pdoNum;
  CMsgTr.m_iLen = 8;
  cInt1[0] = iData1; cInt1[1] = iData1 >> 8; cInt1[2] = iData1 >> 16; cInt1[3] = iData1 >> 24;
  cInt2[0] = iData2; cInt2[1] = iData2 >> 8; cInt2[2] = iData2 >> 16; cInt2[3] = iData2 >> 24;
  CMsgTr.set(cInt1[0], cInt1[1], cInt1[2], cInt1[3], cInt2[0], cInt2[1], cInt2[2], cInt2[3]);
  can_ctrl_->transmitMsg(CMsgTr, true);
}

void RoboteqFbl2360Diff::sendPDOdownload(int pdoNum, int iData1, int iData2, int iData3, int iData4)
{
  char cInt1[2], cInt2[2], cInt3[2], cInt4[2];
  CanMsg CMsgTr;
  CMsgTr.m_iType = 0;
  CMsgTr.m_iID = pdoNum;
  CMsgTr.m_iLen = 8;
  cInt1[0] = iData1; cInt1[1] = iData1 >> 8;
  cInt2[0] = iData2; cInt2[1] = iData2 >> 8;
  cInt3[0] = iData3; cInt3[1] = iData3 >> 8;
  cInt4[0] = iData4; cInt4[1] = iData4 >> 8;
  CMsgTr.set(cInt1[0], cInt1[1], cInt2[0], cInt2[1], cInt3[0], cInt3[1], cInt4[0], cInt4[1]);
  can_ctrl_->transmitMsg(CMsgTr);
}

void RoboteqFbl2360Diff::sendSDOAbort(int iObjIndex, int iObjSubIndex, unsigned int iErrorCode)
{
  CanMsg CMsgTr;
  const int ciAbortTransferReq = 0x04 << 5;
  CMsgTr.m_iLen = 8;
  CMsgTr.m_iID = can_headers_.iRxSDO;
  unsigned char cMsg[8];
  cMsg[0] = ciAbortTransferReq;
  cMsg[1] = iObjIndex;       cMsg[2] = iObjIndex >> 8;
  cMsg[3] = iObjSubIndex;
  cMsg[4] = iErrorCode;      cMsg[5] = iErrorCode >> 8;
  cMsg[6] = iErrorCode >> 16; cMsg[7] = iErrorCode >> 24;
  CMsgTr.set(cMsg[0], cMsg[1], cMsg[2], cMsg[3], cMsg[4], cMsg[5], cMsg[6], cMsg[7]);
  can_ctrl_->transmitMsg(CMsgTr);
}

void RoboteqFbl2360Diff::setChannelOneId(int id) { m1_id_ = id; m2_id_ = id + 1; }

void RoboteqFbl2360Diff::sendSDODownload(int iObjIndex, int iObjSubIndex, int iData)
{
  CanMsg CMsgTr;
  const int ciInitDownloadReq = 0x20;
  const int ciExpedited = 0x02;
  const int ciDataSizeInd = 0x01;
  CMsgTr.m_iLen = 8;
  CMsgTr.m_iID = can_headers_.iRxSDO;
  unsigned char cMsg[8];
  cMsg[0] = ciInitDownloadReq | ciExpedited | ciDataSizeInd;
  cMsg[1] = iObjIndex; cMsg[2] = iObjIndex >> 8;
  cMsg[3] = iObjSubIndex;
  cMsg[4] = iData; cMsg[5] = iData >> 8; cMsg[6] = iData >> 16; cMsg[7] = iData >> 24;
  CMsgTr.set(cMsg[0], cMsg[1], cMsg[2], cMsg[3], cMsg[4], cMsg[5], cMsg[6], cMsg[7]);
  can_ctrl_->transmitMsg(CMsgTr);
}

void RoboteqFbl2360Diff::sendSDOUpload(int iObjIndex, int iObjSubIndex)
{
  CanMsg CMsgTr;
  CMsgTr.m_iLen = 8;
  CMsgTr.m_iID = can_headers_.iRxSDO;
  unsigned char cMsg[8] = {0x40, (unsigned char)iObjIndex, (unsigned char)(iObjIndex >> 8),
                            (unsigned char)iObjSubIndex, 0, 0, 0, 0};
  CMsgTr.set(cMsg[0], cMsg[1], cMsg[2], cMsg[3], cMsg[4], cMsg[5], cMsg[6], cMsg[7]);
  can_ctrl_->transmitMsg(CMsgTr);
}

std::string RoboteqFbl2360Diff::getName() { return name_; }
int RoboteqFbl2360Diff::getCanId() { return node_id_; }
int RoboteqFbl2360Diff::getDriveDirection() { return drive_direction_; }
int RoboteqFbl2360Diff::getHomeSpeed() { return home_speed_; }
int RoboteqFbl2360Diff::getHomeCount() { return home_count_; }

int RoboteqFbl2360Diff::getControllerMode()
{
  if (controller_mode_ != MODE_PRE_INITIATION) {
    time_t now_time = time(NULL);
    if (difftime(now_time, last_heartbeat_time_) > heartbeat_timeout_time_)
      controller_mode_ = MODE_LOST_CONNECTION;
  }
  return controller_mode_;
}

double RoboteqFbl2360Diff::getBatteryVolts() { return battery_volts_; }
double RoboteqFbl2360Diff::getBatteryAmps() { return battery_amps_; }
int RoboteqFbl2360Diff::getMaxTemp() { return max_temp_; }
std::vector<unsigned char> RoboteqFbl2360Diff::getStatusFlags() { return status_flags_; }

bool RoboteqFbl2360Diff::hasIndex(int index)
{
  return index == m1_id_ || index == m2_id_;
}

double RoboteqFbl2360Diff::getSpeed(int channel)
{
  if (channel == m1_id_) return m1_vel_;
  if (channel == m2_id_) return m2_vel_;
  return 0;
}

double RoboteqFbl2360Diff::getPosition(int channel)
{
  if (channel == m1_id_) return m1_pos_;
  if (channel == m2_id_) return m2_pos_;
  return 0;
}

double RoboteqFbl2360Diff::getMotorAmps(int channel)
{
  if (channel == m1_id_) return m1_motor_amps_;
  if (channel == m2_id_) return m2_motor_amps_;
  return 0;
}

int RoboteqFbl2360Diff::getRelativePower(int channel)
{
  if (channel == m1_id_) return m1_relative_power_;
  if (channel == m2_id_) return m2_relative_power_;
  return 0;
}

double RoboteqFbl2360Diff::getPropulsionSpeed(int channel)
{
  if (channel == m1_id_) return m1_vel_;
  if (channel == m2_id_) return m2_vel_;
  return 0;
}

double RoboteqFbl2360Diff::getPropulsionPosition(int channel) { return getPosition(channel); }
double RoboteqFbl2360Diff::getSteeringPosition(int channel) { return 0.; }
double RoboteqFbl2360Diff::getSteeringMotorAmps() { return m2_motor_amps_; }
int RoboteqFbl2360Diff::getSteeringRelativePower() { return m2_relative_power_; }

std::vector<std::string> RoboteqFbl2360Diff::getControllerSetupMap()
{
  return std::vector<std::string>(
    CONTROLLER_SETUP_MAP,
    CONTROLLER_SETUP_MAP + sizeof(CONTROLLER_SETUP_MAP) / sizeof(CONTROLLER_SETUP_MAP[0]));
}
