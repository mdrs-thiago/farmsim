#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <cmath>
#include <map>
#include <string>
#include <vector>

using std::placeholders::_1;

class TeleopRobot : public rclcpp::Node
{
public:
  TeleopRobot()
  : Node("teleop_node",
         rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true))
  {
    loadParameters();

    joy_sub_   = create_subscription<sensor_msgs::msg::Joy>(
      "joy", 1, std::bind(&TeleopRobot::joyCallback, this, _1));
    twist_pub_ = create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 1);
    lock_pub_  = create_publisher<std_msgs::msg::Bool>("joy_priority", 1);
    home_client_ = create_client<std_srvs::srv::Trigger>("home_steering");

    // Build service clients for trigger_map
    {
      auto listed = list_parameters({"trigger_map"}, 10);
      auto params = get_parameters(listed.names);
      for (const auto & p : params) {
        std::string btn = p.get_name().substr(std::string("trigger_map.").size());
        std::string svc = p.as_string();
        if (!svc.empty()) {
          RCLCPP_INFO(get_logger(), "%s → service %s", btn.c_str(), svc.c_str());
          custom_trigger_map_[btn] = create_client<std_srvs::srv::Trigger>(svc);
        }
      }
    }

    previous_drive_mode_    = MODE_FORWARD;
    previous_non_turn_mode_ = MODE_FORWARD;
    previous_buttons_.resize(number_of_buttons_, false);
    teleop_lock_on_          = true;
    turning_buttons_initiated_ = false;

    ang_max_      = M_PI / 2.0;
    ang_min_      = 0.0;
    deadzone_     = 0.2;
    turning_limit_ = 0.8;
    a_ = (ang_max_ - ang_min_) / (1.0 - deadzone_);
    b_ = ang_min_ - deadzone_ * a_;
  }

private:
  enum DriveModes { MODE_FORWARD, MODE_LEFT, MODE_TURNING, MODE_OMNI };

  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr lock_pub_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr home_client_;
  std::map<std::string, rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr> custom_trigger_map_;

  std::map<std::string, int> axis_map_;
  std::map<std::string, int> button_map_;
  std::string button_turning_safety_;
  std::vector<std::string> home_buttons_;
  std::vector<std::string> omni_buttons_;
  std::vector<std::string> kv_default_buttons_;
  std::vector<std::string> mode_forward_buttons_;
  std::vector<std::string> mode_left_buttons_;

  std::vector<bool> previous_buttons_;
  std::vector<int>  buttons_;
  std::vector<float> axes_;
  int previous_drive_mode_;
  int previous_non_turn_mode_;
  int number_of_buttons_;
  bool teleop_lock_on_;
  bool turning_buttons_initiated_;

  double kv_, kv_max_, kv_min_, dkv_, kv_default_;
  double ang_min_, ang_max_, deadzone_, turning_limit_;
  double a_, b_;
  double turn_calc_w_, turn_calc_l_;

  void loadParameters()
  {
    turn_calc_w_ = get_parameter_or("turn_calc_w", 1.0);
    turn_calc_l_ = get_parameter_or("turn_calc_l", 1.0);

    // axis_map — dot-separated keys under "axis_map.*"
    {
      auto listed = list_parameters({"axis_map"}, 10);
      auto params = get_parameters(listed.names);
      for (const auto & p : params) {
        std::string key = p.get_name().substr(std::string("axis_map.").size());
        axis_map_[key] = static_cast<int>(p.as_int());
      }
    }
    if (!axis_map_.count("axis_vx") || !axis_map_.count("axis_wz")) {
      RCLCPP_WARN(get_logger(), "axis_map missing axis_vx or axis_wz");
    }

    // button_map
    {
      auto listed = list_parameters({"button_map"}, 10);
      auto params = get_parameters(listed.names);
      for (const auto & p : params) {
        std::string key = p.get_name().substr(std::string("button_map.").size());
        button_map_[key] = static_cast<int>(p.as_int());
      }
    }
    number_of_buttons_ = 0;
    for (const auto & [k, v] : button_map_) {
      if (v > number_of_buttons_) { number_of_buttons_ = v; }
    }
    number_of_buttons_++;  // size = max_index + 1

    button_turning_safety_ = get_parameter_or<std::string>("button_turning_safety", "");
    home_buttons_          = get_parameter_or("home_buttons",       std::vector<std::string>{});
    omni_buttons_          = get_parameter_or("omni_buttons",       std::vector<std::string>{});
    kv_default_buttons_    = get_parameter_or("kv_default_buttons", std::vector<std::string>{});
    mode_forward_buttons_  = get_parameter_or("mode_forward_buttons", std::vector<std::string>{});
    mode_left_buttons_     = get_parameter_or("mode_left_buttons",    std::vector<std::string>{});

    kv_min_     = get_parameter_or("gains.kv_min",       0.1);
    kv_max_     = get_parameter_or("gains.kv_max",       1.5);
    dkv_        = get_parameter_or("gains.kv_increment", 0.1);
    kv_default_ = get_parameter_or("gains.kv_default",   0.75);
    kv_ = kv_default_;

    RCLCPP_INFO(get_logger(), "Teleop: %zu axes, %zu buttons mapped",
      axis_map_.size(), button_map_.size());
  }

  template<typename T>
  T get_parameter_or(const std::string & name, T default_val)
  {
    try {
      if (has_parameter(name)) {
        return get_parameter(name).get_value<T>();
      }
    } catch (...) {}
    return default_val;
  }

  bool hasAxis(const std::string & name) const
  {
    return axis_map_.count(name) > 0;
  }

  bool hasButton(const std::string & name) const
  {
    return button_map_.count(name) > 0;
  }

  int safeButton(const std::string & name) const
  {
    if (!hasButton(name)) { return 0; }
    int idx = button_map_.at(name);
    if (idx < 0 || idx >= static_cast<int>(buttons_.size())) { return 0; }
    return buttons_[idx];
  }

  bool safePrevButton(const std::string & name) const
  {
    if (!hasButton(name)) { return false; }
    int idx = button_map_.at(name);
    if (idx < 0 || idx >= static_cast<int>(previous_buttons_.size())) { return false; }
    return previous_buttons_[idx];
  }

  bool evaluateButtonPress(const std::vector<std::string> & combo)
  {
    if (combo.empty()) { return false; }
    bool all_now = true, all_prev = true;
    for (const auto & b : combo) {
      all_now  &= safeButton(b) != 0;
      all_prev &= safePrevButton(b);
    }
    return all_now && !all_prev;
  }

  bool evaluateButtonPress(const std::string & b)
  {
    return safeButton(b) && !safePrevButton(b);
  }

  std::vector<int> buttonifyAxes(const std::vector<float> & axes_in)
  {
    std::vector<int> out;
    for (float v : axes_in) { out.push_back(v > 0.9f ? 1 : 0); }
    return out;
  }

  void joyCallback(const sensor_msgs::msg::Joy::SharedPtr joy)
  {
    buttons_.clear();
    axes_ = joy->axes;
    for (int b : joy->buttons) { buttons_.push_back(b); }
    auto int_axes = buttonifyAxes(axes_);
    buttons_.insert(buttons_.end(), int_axes.begin(), int_axes.end());

    // Resize previous_buttons_ if needed
    if (static_cast<int>(previous_buttons_.size()) < number_of_buttons_) {
      previous_buttons_.resize(number_of_buttons_, false);
    }

    double vx = 0, vy = 0, wz = 0;
    int mode = previous_drive_mode_;

    // Axis helpers
    auto axisVal = [&](const std::string & k) -> double {
      if (!hasAxis(k)) { return 0.0; }
      int idx = axis_map_.at(k);
      if (idx < 0 || idx >= static_cast<int>(axes_.size())) { return 0.0; }
      return static_cast<double>(axes_[idx]);
    };

    double axis_v_primary   = axisVal("axis_vx");
    double axis_v_secondary = axisVal("axis_vy");
    double axis_wz_raw      = axisVal("axis_wz");
    double axis_turn_left   = axisVal("axis_turn_left");
    double axis_turn_right  = axisVal("axis_turn_right");

    // Initialise turning-trigger axes (first time both at 1)
    if (!turning_buttons_initiated_ && axis_turn_left == 1.0 && axis_turn_right == 1.0) {
      turning_buttons_initiated_ = true;
    }

    // Teleop lock
    if (safeButton("button_teleop_lock")) {
      if (safeButton("button_function") && !safePrevButton("button_function") && teleop_lock_on_) {
        teleop_lock_on_ = false;
      } else if (!safePrevButton("button_teleop_lock")) {
        teleop_lock_on_ = true;
      }
    }
    if (teleop_lock_on_) {
      std_msgs::msg::Bool lk;
      lk.data = true;
      lock_pub_->publish(lk);
    }

    // Custom trigger services (button_function held + individual button)
    if (safeButton("button_function")) {
      for (auto & [btn, client] : custom_trigger_map_) {
        if (hasButton(btn) && safeButton(btn) && !safePrevButton(btn)) {
          auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
          client->async_send_request(req);
        }
      }
    } else if (evaluateButtonPress(home_buttons_)) {
      auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
      home_client_->async_send_request(req);
    }

    // Drive mode selection
    if ((axis_turn_left < turning_limit_ || axis_turn_right < turning_limit_) &&
        safeButton(button_turning_safety_) && turning_buttons_initiated_)
    {
      mode = MODE_TURNING;
    } else if (fabs(axis_v_primary) > deadzone_ && previous_drive_mode_ == MODE_TURNING) {
      mode = previous_non_turn_mode_;
    } else if (evaluateButtonPress(mode_forward_buttons_)) {
      mode = MODE_FORWARD; previous_non_turn_mode_ = MODE_FORWARD;
    } else if (evaluateButtonPress(mode_left_buttons_)) {
      mode = MODE_LEFT; previous_non_turn_mode_ = MODE_LEFT;
    } else if (evaluateButtonPress(omni_buttons_)) {
      mode = MODE_OMNI; previous_non_turn_mode_ = MODE_FORWARD;
    }

    // Gain
    if (evaluateButtonPress(kv_default_buttons_)) {
      kv_ = kv_default_;
    } else if (evaluateButtonPress("button_more_gain")) {
      kv_ = std::min(kv_ + dkv_, kv_max_);
    } else if (evaluateButtonPress("button_less_gain")) {
      kv_ = std::max(kv_ - dkv_, kv_min_);
    }

    // Commands
    if (mode == MODE_TURNING) {
      wz = (0.5 - 0.5 * axis_turn_left) - (0.5 - 0.5 * axis_turn_right);
      if (!safeButton(button_turning_safety_)) { wz = 0.0; }
      if (wz == 0.0) { wz = 0.00001; }
    } else {
      if (axis_v_primary > -deadzone_ && axis_v_primary < deadzone_) { axis_v_primary = 0; }
      axis_v_primary   *= kv_;
      axis_v_secondary *= kv_;

      if (axis_wz_raw > -deadzone_ && axis_wz_raw < deadzone_) {
        wz = 0;
      } else if (axis_wz_raw > 0) {
        double ang = a_ * axis_wz_raw + b_;
        wz = axis_v_primary / (turn_calc_l_ / tan(ang) + turn_calc_w_);
      } else {
        double ang = a_ * axis_wz_raw - b_;
        wz = axis_v_primary / (turn_calc_l_ / tan(ang) - turn_calc_w_);
      }

      if (mode == MODE_FORWARD) {
        vx = axis_v_primary; vy = 0;
      } else if (mode == MODE_LEFT) {
        vy = axis_v_primary != 0 ? axis_v_primary : 0.000001; vx = 0;
      } else if (mode == MODE_OMNI) {
        vx = axis_v_primary;
        vy = axis_v_secondary;
        if (fabs(vx) > 0 || fabs(vy) > 0 || safeButton("button_6")) {
          wz = axis_wz_raw * 1.5 * kv_ * (1 - 2 * (fabs(atan2(vy, vx)) > M_PI / 2));
        } else {
          wz = 0;
        }
      }
    }

    if (teleop_lock_on_) {
      geometry_msgs::msg::Twist t;
      t.linear.x = vx; t.linear.y = vy;
      t.angular.z = wz;
      twist_pub_->publish(t);
    }

    previous_drive_mode_ = mode;
    for (int i = 0; i < number_of_buttons_ && i < static_cast<int>(buttons_.size()); i++) {
      previous_buttons_[i] = buttons_[i] != 0;
    }
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TeleopRobot>());
  rclcpp::shutdown();
  return 0;
}
