#ifndef ROBOTEQFBL2360_INCLUDEDEF_H
#define ROBOTEQFBL2360_INCLUDEDEF_H

#include <thorvald_can_devices/CanDriveItf.h>
#include <time.h>
#include <map>
#include <ctime>
#include <chrono>

class RoboteqFbl2360 : public CanDriveItf
{
public:
  enum StatusFlagTypes
  {
    IND_FEEDBACK_SPEED_ERROR,
    IND_FEEDBACK_POS_ERROR,
    IND_ALIGNMENT_ERROR,
    IND_POWER_STAGE_OFF,
    IND_STALL_DETECTED,
    IND_HOMING_REACHED_MAX_COUNT,
    IND_ILLEGAL_COMMAND,
    IND_CONTROLLER_OK,
    NUMBER_OF_STATUS_FLAGS
  };

  struct CanHeaders
  {
    int iTxPDO1, iTxPDO2, iTxPDO3, iTxPDO4;
    int iRxPDO1, iRxPDO2, iRxPDO3, iRxPDO4;
    int heartbeat;
    int iTxSDO, iRxSDO;
  };

  struct DriveParam
  {
    double radius_wheel;
    double command_scaling_factor;
    double position_limit;
    double gear_ratio;
    double encoder_ppr;
    double sim_a;
    double sim_v;
  };

  void setCanItf(CanItf * pCanItf) { can_ctrl_ = pCanItf; }
  void setDriveParams(int channel, std::map<std::string, double> param_map);

  RoboteqFbl2360();

  void setCanOpenParam(int node_id);
  void setupController(int setup_id, int setup_value);
  bool setCurrentPosAsZero();
  void driveMode();
  bool homeSteering();
  void decouple();
  void sendDriveCommand(
    int channel, double speed, double position,
    double position_max_speed, bool invert_as_needed = true);
  void sendDiffDriveCommand(int speed_1, int speed_2);
  void sendAccelerationSDO(int channel, int value);
  void sendDecelerationSDO(int channel, int value);
  void sendHeartbeat();
  bool evalReceivedMsg(CanMsg & msg);
  void setControllerIndex(int index);
  void simulateDrive(
    int channel, double speed_command, double position_command,
    bool invert_as_needed = true);

  std::string getName();
  int getCanId();
  int getDriveDirection();
  int getHomeSpeed();
  int getHomeCount();
  int getControllerMode();
  double getBatteryVolts();
  double getBatteryAmps();
  int getMaxTemp();
  std::vector<unsigned char> getStatusFlags();
  bool hasIndex(int index);
  double getSpeed(int channel);
  double getPosition(int channel);
  double getMotorAmps(int channel);
  int getRelativePower(int channel);
  double getPropulsionSpeed(int channel);
  double getPropulsionPosition(int channel);
  double getSteeringPosition(int channel);
  double getSteeringMotorAmps();
  int getSteeringRelativePower();
  std::vector<std::string> getControllerSetupMap();

protected:
  const static std::string CONTROLLER_SETUP_MAP[];

  CanHeaders can_headers_;
  DriveParam m1_params_, m2_params_;

  CanItf * can_ctrl_;
  CanMsg last_can_msg_;
  int controller_index_;

  std::string name_;
  int node_id_, drive_direction_, home_speed_, home_count_;

  int controller_mode_;
  double battery_volts_, battery_amps_;
  int max_temp_;
  std::vector<unsigned char> status_flags_;
  char status_flags_old_;

  double m1_vel_, m1_pos_, m1_motor_amps_;
  int m1_relative_power_;

  int m2_vel_;
  double m2_pos_, m2_motor_amps_;
  int m2_relative_power_;

  time_t last_heartbeat_time_;
  double heartbeat_timeout_time_;
  bool drive_inverted_;

  std::map<std::string, double> drive_parameters_;
  std::chrono::steady_clock::time_point previous_simulation_timestep_;
  void updateWheelPos(double speed, double dt);

  int command_seq_;

  void sendModeCommand(int mode);
  void sendPDOdownload(int pdoNum, int iData1, int iData2);
  void sendPDOdownload(int pdoNum, int iData1, int iData2, int iData3, int iData4);
  void sendSDOAbort(int iObjIndex, int iObjSubIndex, unsigned int iErrorCode);
  void sendSDODownload(int iObjIndex, int iObjSub, int iData);
  void sendSDOUpload(int iObjIndex, int iObjSubIndex);
};

#endif  // ROBOTEQFBL2360_INCLUDEDEF_H
