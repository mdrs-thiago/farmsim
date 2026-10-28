#ifndef THORVALD_BASE_DRIVER_PLTF_CLC_ITF_H_
#define THORVALD_BASE_DRIVER_PLTF_CLC_ITF_H_

#include <geometry_msgs/msg/twist.hpp>
#include <thorvald_msgs/msg/base_state.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/time.hpp>
#include <vector>
#include <map>
#include <string>

namespace pltf_clc_itf
{
  class PltfClcItf
  {
  public:
    virtual void initialize(std::vector<std::map<std::string, double>> motor_drives) = 0;

    virtual bool setParams(std::vector<std::map<std::string, double>> motor_drives) = 0;

    virtual bool calculateCommand(
      geometry_msgs::msg::Twist twist_in,
      thorvald_msgs::msg::BaseState & joint_states_out) = 0;

    // stamp is passed from the node clock so the plugin does not need rclcpp::Node
    virtual bool calculateOdometry(
      thorvald_msgs::msg::BaseState joint_states_in,
      nav_msgs::msg::Odometry & odom_out,
      rclcpp::Time stamp) = 0;

    virtual bool setZeroSpeed(thorvald_msgs::msg::BaseState & joint_states_out) = 0;

    virtual void zeroOdometryPose() = 0;

  protected:
    PltfClcItf() {}
  };
}  // namespace pltf_clc_itf

#endif  // THORVALD_BASE_DRIVER_PLTF_CLC_ITF_H_
