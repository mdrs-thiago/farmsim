#ifndef CANDRIVEITF_INCLUDEDEF_H
#define CANDRIVEITF_INCLUDEDEF_H

#include <thorvald_can_devices/CanItf.h>
#include <map>
#include <string>
#include <vector>

class CanDriveItf
{
public:
  enum ModuleMode
  {
    MODE_DRIVE = 0,
    MODE_HOME = 1,
    MODE_DECOUPLED = 2,
    MODE_PRE_INITIATION = -99,
    MODE_LOST_CONNECTION = -98
  };

  virtual void setCanItf(CanItf * pCanItf) = 0;
  virtual void setDriveParams(int channel, std::map<std::string, double> param_map) = 0;
  virtual void setCanOpenParam(int node_id) = 0;
  virtual void setupController(int setup_id, int setup_value) = 0;
  virtual bool setCurrentPosAsZero() = 0;
  virtual void sendDriveCommand(
    int channel, double speed, double position,
    double position_max_speed, bool invert_as_needed = true) = 0;
  virtual bool evalReceivedMsg(CanMsg & msg) = 0;
  virtual void simulateDrive(
    int channel, double speed_command, double position_command,
    bool invert_as_needed = true) = 0;

  virtual std::string getName() = 0;
  virtual int getCanId() = 0;
  virtual int getDriveDirection() = 0;
  virtual int getHomeSpeed() = 0;
  virtual int getHomeCount() = 0;
  virtual int getControllerMode() = 0;
  virtual double getBatteryVolts() = 0;
  virtual double getBatteryAmps() = 0;
  virtual int getMaxTemp() = 0;
  virtual std::vector<unsigned char> getStatusFlags() = 0;
  virtual bool hasIndex(int index) = 0;
  virtual double getSpeed(int channel) = 0;
  virtual double getPosition(int channel) = 0;
  virtual double getMotorAmps(int channel) = 0;
  virtual int getRelativePower(int channel) = 0;
  virtual double getPropulsionSpeed(int channel) = 0;
  virtual double getPropulsionPosition(int channel) = 0;
  virtual double getSteeringPosition(int channel) = 0;
  virtual std::vector<std::string> getControllerSetupMap() = 0;
  virtual bool homeSteering() = 0;
};

#endif  // CANDRIVEITF_INCLUDEDEF_H
