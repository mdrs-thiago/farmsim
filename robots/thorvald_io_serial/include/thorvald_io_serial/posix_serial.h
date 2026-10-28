#ifndef THORVALD_IO_SERIAL_POSIX_SERIAL_H_
#define THORVALD_IO_SERIAL_POSIX_SERIAL_H_

// Minimal POSIX serial port helper replacing the wjwwood/serial library.
// Supports: open with baud rate, write string, readline with delimiter timeout.

#include <string>
#include <stdexcept>
#include <cstring>
#include <cerrno>

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <sys/select.h>

namespace posix_serial
{

static speed_t baudToSpeed(int baud)
{
  switch (baud) {
    case 9600:   return B9600;
    case 19200:  return B19200;
    case 38400:  return B38400;
    case 57600:  return B57600;
    case 115200: return B115200;
    default:     return B9600;
  }
}

class Serial
{
public:
  Serial(const std::string & port, int baud, int timeout_ms)
  : fd_(-1), timeout_ms_(timeout_ms)
  {
    fd_ = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) { return; }

    struct termios tty;
    if (tcgetattr(fd_, &tty) != 0) { ::close(fd_); fd_ = -1; return; }

    speed_t spd = baudToSpeed(baud);
    cfsetispeed(&tty, spd);
    cfsetospeed(&tty, spd);
    cfmakeraw(&tty);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 0;

    if (tcsetattr(fd_, TCSANOW, &tty) != 0) { ::close(fd_); fd_ = -1; }
  }

  ~Serial()
  {
    if (fd_ >= 0) { ::close(fd_); }
  }

  bool isOpen() const { return fd_ >= 0; }

  void write(const std::string & data)
  {
    if (fd_ < 0) { return; }
    ssize_t wr = ::write(fd_, data.c_str(), data.size());
    (void)wr;
  }

  // Read until one of the characters in terminator, up to max_size bytes.
  std::string readline(size_t max_size, const std::string & terminator)
  {
    std::string result;
    if (fd_ < 0) { return result; }

    auto deadline_ms = timeout_ms_;
    while (result.size() < max_size) {
      fd_set rds;
      FD_ZERO(&rds);
      FD_SET(fd_, &rds);
      struct timeval tv;
      tv.tv_sec  = deadline_ms / 1000;
      tv.tv_usec = (deadline_ms % 1000) * 1000;
      int ret = select(fd_ + 1, &rds, nullptr, nullptr, &tv);
      if (ret <= 0) { break; }

      char c;
      ssize_t n = ::read(fd_, &c, 1);
      if (n <= 0) { break; }
      result += c;
      if (terminator.find(c) != std::string::npos) { break; }
    }
    return result;
  }

private:
  int fd_;
  int timeout_ms_;
};

}  // namespace posix_serial

#endif  // THORVALD_IO_SERIAL_POSIX_SERIAL_H_
