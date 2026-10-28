#ifndef KCBATTERY_INCLUDEDEF_H
#define KCBATTERY_INCLUDEDEF_H

#include <thorvald_can_devices/CanItf.h>
#include <time.h>
#include <ctime>
#include <stdint.h>
#include <vector>
#include <map>
#include <string>

class KCBattery
{
public:
  struct CanHeaders
  {
    int tpdo_1, tpdo_2, tpdo_3;
  };

  KCBattery();

  void setCanItf(CanItf * pCanItf) { can_ctrl_ = pCanItf; }
  void setCanParam(int node_id);
  bool hasPulse();
  bool evalReceivedMsg(CanMsg & msg);

  int getCanId();
  int getStateOfCharge();
  int getStateOfHealth();
  double getVoltage();
  double getCurrent();
  bool getChargeFet();
  bool getDischargeFet();
  short getCycleCount();
  short getCapacity();
  std::vector<double> getTemps();
  bool getAOverVolt();
  bool getAUnderVolt();
  bool getAOverTemp();
  bool getAUnderTemp();
  bool getAOverCharge();
  bool getAOverDischarge();
  bool getACanFail();
  bool getACellVoltSensFault();
  bool getABmsFail();
  bool getAShortCircuitDischarge();
  bool getATempSensFault();
  bool getAOverVoltFault();
  bool getAUnderVoltFault();
  bool getAPackVoltSensFault();

protected:
  struct BatteryVars
  {
    short soc, soh, voltage, current;
    bool charge_fet, discharge_fet;
    short cycle_count, capacity;
    std::vector<double> temps;
    bool a_over_volt, a_under_volt, a_over_temp, a_under_temp;
    bool a_over_charge, a_over_discharge, a_can_fail, a_cell_volt_sens_fault;
    bool a_bms_fail, a_short_circ_discharge, a_temp_sens_fault;
    bool a_over_volt_fault, a_under_volt_fault, a_pack_volt_sens_fault;
  };

  CanHeaders can_headers_;
  BatteryVars battery_vars_;

  CanItf * can_ctrl_;
  CanMsg last_can_msg_;

  int node_id_;
  int device_mode_;
  int state_of_charge_;
  int next_var_to_update_;

  time_t last_heartbeat_time_;
  double heartbeat_timeout_time_;

  std::map<std::string, double> drive_parameters_;

  bool hasCommunicated();
};

#endif  // KCBATTERY_INCLUDEDEF_H
