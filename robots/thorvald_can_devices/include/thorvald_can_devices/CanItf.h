#ifndef CANITF_INCLUDEDEF_H
#define CANITF_INCLUDEDEF_H

// Minimal CAN interface vendored from cob_generic_can (ROS1).
// Only the subset used by thorvald_can_devices is reproduced here.

#include <cstring>

struct CanMsg
{
  int m_iID{0};
  int m_iLen{0};
  int m_iType{0};

private:
  unsigned char m_ucBuf[8]{};

public:
  void set(unsigned char b0, unsigned char b1, unsigned char b2, unsigned char b3,
           unsigned char b4, unsigned char b5, unsigned char b6, unsigned char b7)
  {
    m_ucBuf[0] = b0; m_ucBuf[1] = b1; m_ucBuf[2] = b2; m_ucBuf[3] = b3;
    m_ucBuf[4] = b4; m_ucBuf[5] = b5; m_ucBuf[6] = b6; m_ucBuf[7] = b7;
  }

  unsigned char getAt(int index) const { return m_ucBuf[index]; }
  void setAt(unsigned char val, int index) { m_ucBuf[index] = val; }
};

class CanItf
{
public:
  virtual ~CanItf() = default;
  virtual bool transmitMsg(CanMsg & msg, bool bBlocking = true) = 0;
  virtual bool receiveMsg(CanMsg * msg) = 0;
  virtual bool init_ret() { return false; }
};

#endif  // CANITF_INCLUDEDEF_H
