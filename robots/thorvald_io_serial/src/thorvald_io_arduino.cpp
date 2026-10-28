#include <rclcpp/rclcpp.hpp>
#include <thorvald_io_serial/posix_serial.h>
#include <thorvald_msgs/msg/thorvald_io.hpp>

#include <sstream>
#include <string>

class LightRigSerial : public rclcpp::Node
{
public:
  LightRigSerial(const std::string & port, int baud, int serial_timeout)
  : Node("thorvald_io_arduino"),
    serial_(port, baud, serial_timeout)
  {
    status_pub_ = create_publisher<thorvald_msgs::msg::ThorvaldIO>("serial_io", 1);

    if (serial_.isOpen()) {
      RCLCPP_INFO(get_logger(), "Serial port %s open at %d baud.", port.c_str(), baud);
    } else {
      RCLCPP_WARN(get_logger(), "Serial port %s could not be opened.", port.c_str());
    }
  }

  void requestStatus()
  {
    thorvald_msgs::msg::ThorvaldIO status_msg;
    status_msg.header.stamp = this->now();

    // Read digital end-stops
    status_msg.digitals.resize(2);
    std::string end_stops_str = sendSerialCommand("?e:,");
    auto colon  = end_stops_str.find(':');
    auto comma  = end_stops_str.find(',');
    if (colon != std::string::npos && comma != std::string::npos && comma > colon) {
      std::istringstream ss(end_stops_str.substr(colon + 1, comma - colon - 1));
      int d1 = 0, d2 = 0;
      ss >> d1 >> d2;
      status_msg.digitals[0] = d1;
      status_msg.digitals[1] = d2;
    }

    // Read distance sensors
    std::string dists_str = sendSerialCommand("?c:,");
    auto tilde = dists_str.find('~');
    if (tilde != std::string::npos) {
      dists_str = dists_str.substr(tilde + 1);
    }
    dists_str += " ,";

    int n = 0;
    bool finished = false;
    while (!finished && n < 10) {
      if (dists_str.empty() || dists_str[0] == ',') {
        finished = true;
      } else {
        size_t pos = dists_str.find(' ');
        if (pos == std::string::npos) { break; }
        int dist = std::stoi(dists_str.substr(0, pos));
        status_msg.ranges.push_back(static_cast<double>(dist));
        dists_str.erase(0, pos + 1);
        n++;
      }
    }

    status_pub_->publish(status_msg);
  }

private:
  posix_serial::Serial serial_;
  rclcpp::Publisher<thorvald_msgs::msg::ThorvaldIO>::SharedPtr status_pub_;

  std::string sendSerialCommand(const std::string & cmd)
  {
    serial_.write(cmd);
    std::string reply = serial_.readline(1000, ",\r\n");
    if (reply.size() >= 2) {
      reply = reply.substr(0, reply.size() - 2);
    }
    return reply;
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  if (argc < 3) {
    RCLCPP_ERROR(rclcpp::get_logger("thorvald_io_arduino"),
      "Usage: thorvald_io_arduino <serial_port> <baudrate>");
    return 1;
  }

  std::string port(argv[1]);
  int baud = std::stoi(argv[2]);

  auto node = std::make_shared<LightRigSerial>(port, baud, 1000);

  rclcpp::Rate rate(20);
  rclcpp::sleep_for(std::chrono::seconds(3));

  while (rclcpp::ok()) {
    rclcpp::spin_some(node);
    node->requestStatus();
    rate.sleep();
  }

  rclcpp::shutdown();
  return 0;
}
