#ifndef CANPEAKSYSUSB_H
#define CANPEAKSYSUSB_H

// Stub: real hardware requires libpcan (PEAK-System PCAN-USB driver).
// Replace this file with a libpcan-backed implementation when building for real hardware.

#include <thorvald_can_devices/CanItf.h>
#include <iostream>

static const int CAN_BAUD_500K = 500000;

class CANPeakSysUSB : public CanItf
{
public:
  CANPeakSysUSB(const char * /*device_file*/, int /*baud_rate*/)
  {
    std::cerr << "[CANPeakSysUSB] libpcan not linked — real CAN hardware unsupported.\n";
  }

  bool transmitMsg(CanMsg & /*msg*/, bool /*bBlocking*/ = true) override { return false; }
  bool receiveMsg(CanMsg * /*msg*/) override { return false; }
  bool init_ret() override { return false; }
};

#endif  // CANPEAKSYSUSB_H
