#include <assert.h>
#include <thorvald_can_devices/kc_battery.h>
#include <unistd.h>
#include <cmath>

KCBattery::KCBattery()
{
  can_ctrl_ = NULL;
  node_id_ = -1;
  state_of_charge_ = 0;
  next_var_to_update_ = 0;
  last_heartbeat_time_ = time(NULL);
  heartbeat_timeout_time_ = 2;

  battery_vars_.soc = -99;
  battery_vars_.soh = 0;
  battery_vars_.voltage = 0;
  battery_vars_.current = 0;
  battery_vars_.charge_fet = false;
  battery_vars_.discharge_fet = false;
  battery_vars_.cycle_count = 0;
  battery_vars_.capacity = 0;
  battery_vars_.temps.resize(7, 0.0);
  battery_vars_.a_over_volt = false;
  battery_vars_.a_under_volt = false;
  battery_vars_.a_over_temp = false;
  battery_vars_.a_under_temp = false;
  battery_vars_.a_over_charge = false;
  battery_vars_.a_over_discharge = false;
  battery_vars_.a_can_fail = false;
  battery_vars_.a_cell_volt_sens_fault = false;
  battery_vars_.a_bms_fail = false;
  battery_vars_.a_short_circ_discharge = false;
  battery_vars_.a_temp_sens_fault = false;
  battery_vars_.a_over_volt_fault = false;
  battery_vars_.a_under_volt_fault = false;
  battery_vars_.a_pack_volt_sens_fault = false;
}

void KCBattery::setCanParam(int node_id)
{
  node_id_ = node_id;
  can_headers_.tpdo_1 = 0x3c1;
  can_headers_.tpdo_2 = 0x3d1;
  can_headers_.tpdo_3 = 0x3e1;
}

bool KCBattery::hasPulse()
{
  time_t now_time = time(NULL);
  return difftime(now_time, last_heartbeat_time_) <= heartbeat_timeout_time_;
}

bool KCBattery::hasCommunicated()
{
  return battery_vars_.soc != -99;
}

bool KCBattery::evalReceivedMsg(CanMsg & msg)
{
  bool b_ret = false;
  last_can_msg_ = msg;

  if (msg.m_iID == can_headers_.tpdo_1) {
    char t_byte_1 = msg.getAt(1);
    battery_vars_.charge_fet = t_byte_1 & 1;
    battery_vars_.discharge_fet = t_byte_1 & (1 << 1);
    battery_vars_.voltage = (msg.getAt(3) << 8) | msg.getAt(2);

    short t_byte_4_5 = (msg.getAt(5) << 8) | msg.getAt(4);
    battery_vars_.a_over_volt          = t_byte_4_5 & 1;
    battery_vars_.a_under_volt         = t_byte_4_5 & (1 << 1);
    battery_vars_.a_over_temp          = t_byte_4_5 & (1 << 2);
    battery_vars_.a_under_temp         = t_byte_4_5 & (1 << 3);
    battery_vars_.a_over_charge        = t_byte_4_5 & (1 << 4);
    battery_vars_.a_over_discharge     = t_byte_4_5 & (1 << 5);
    battery_vars_.a_can_fail           = t_byte_4_5 & (1 << 7);
    battery_vars_.a_cell_volt_sens_fault = t_byte_4_5 & (1 << 9);
    battery_vars_.a_bms_fail           = t_byte_4_5 & (1 << 10);
    battery_vars_.a_short_circ_discharge = t_byte_4_5 & (1 << 11);
    battery_vars_.a_temp_sens_fault    = t_byte_4_5 & (1 << 12);
    battery_vars_.a_over_volt_fault    = t_byte_4_5 & (1 << 13);
    battery_vars_.a_under_volt_fault   = t_byte_4_5 & (1 << 14);
    battery_vars_.a_pack_volt_sens_fault = t_byte_4_5 & (1 << 15);

    battery_vars_.temps[0] = msg.getAt(6);
    battery_vars_.temps[1] = msg.getAt(7);
  } else if (msg.m_iID == can_headers_.tpdo_2) {
    battery_vars_.temps[2] = msg.getAt(0);
    battery_vars_.temps[3] = msg.getAt(1);
    battery_vars_.temps[4] = msg.getAt(2);
    battery_vars_.temps[5] = msg.getAt(3);
    battery_vars_.current  = (msg.getAt(5) << 8) | msg.getAt(4);
    battery_vars_.temps[6] = msg.getAt(6);
  } else if (msg.m_iID == can_headers_.tpdo_3) {
    battery_vars_.soc        = msg.getAt(0);
    battery_vars_.soh        = msg.getAt(1);
    battery_vars_.cycle_count = (msg.getAt(3) << 8) | msg.getAt(2);
    battery_vars_.capacity   = (msg.getAt(5) << 8) | msg.getAt(4);
    last_heartbeat_time_     = time(NULL);
    b_ret = true;
  }

  return b_ret;
}

int KCBattery::getCanId() { return node_id_; }
int KCBattery::getStateOfCharge() { return battery_vars_.soc; }
int KCBattery::getStateOfHealth() { return battery_vars_.soh; }
double KCBattery::getVoltage() { return battery_vars_.voltage / 100.; }
double KCBattery::getCurrent() { return battery_vars_.current / 100.; }
bool KCBattery::getChargeFet() { return battery_vars_.charge_fet; }
bool KCBattery::getDischargeFet() { return battery_vars_.discharge_fet; }
short KCBattery::getCycleCount() { return battery_vars_.cycle_count; }
short KCBattery::getCapacity() { return battery_vars_.capacity; }
std::vector<double> KCBattery::getTemps() { return battery_vars_.temps; }
bool KCBattery::getAOverVolt() { return battery_vars_.a_over_volt; }
bool KCBattery::getAUnderVolt() { return battery_vars_.a_under_volt; }
bool KCBattery::getAOverTemp() { return battery_vars_.a_over_temp; }
bool KCBattery::getAUnderTemp() { return battery_vars_.a_under_temp; }
bool KCBattery::getAOverCharge() { return battery_vars_.a_over_charge; }
bool KCBattery::getAOverDischarge() { return battery_vars_.a_over_discharge; }
bool KCBattery::getACanFail() { return battery_vars_.a_can_fail; }
bool KCBattery::getACellVoltSensFault() { return battery_vars_.a_cell_volt_sens_fault; }
bool KCBattery::getABmsFail() { return battery_vars_.a_bms_fail; }
bool KCBattery::getAShortCircuitDischarge() { return battery_vars_.a_short_circ_discharge; }
bool KCBattery::getATempSensFault() { return battery_vars_.a_temp_sens_fault; }
bool KCBattery::getAOverVoltFault() { return battery_vars_.a_over_volt_fault; }
bool KCBattery::getAUnderVoltFault() { return battery_vars_.a_under_volt_fault; }
bool KCBattery::getAPackVoltSensFault() { return battery_vars_.a_pack_volt_sens_fault; }
