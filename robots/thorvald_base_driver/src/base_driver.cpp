#include <rclcpp/rclcpp.hpp>
#include <pluginlib/class_loader.hpp>
#include <thorvald_base_driver/pltf_clc_itf.h>
#include <thorvald_base_driver/can_ctrl_pltf.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <thorvald_msgs/msg/base_state.hpp>
#include <thorvald_msgs/msg/battery_array.hpp>
#include <thorvald_msgs/msg/can_frame.hpp>
#include <thorvald_msgs/msg/controller_array.hpp>
#include <thorvald_msgs/srv/can_id.hpp>
#include <thorvald_msgs/srv/drive_params.hpp>
#include <thorvald_msgs/srv/motor_controller_setup.hpp>

#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using std::placeholders::_1;
using std::placeholders::_2;

class BaseDriver : public rclcpp::Node
{
public:
  BaseDriver()
  : Node("base_driver"),
    pltf_clc_loader_("thorvald_base_driver", "pltf_clc_itf::PltfClcItf")
  {
    can_ctrl_pltf_ = new CanCtrlPltf();

    // Subscribers
    twist_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      "cmd_vel", 10, std::bind(&BaseDriver::twistCallback, this, _1));
    can_device_t_sub_ = create_subscription<thorvald_msgs::msg::CANFrame>(
      "can_frames_device_t", 100, std::bind(&BaseDriver::canDeviceTCallback, this, _1));

    // Publishers
    joint_command_pub_         = create_publisher<thorvald_msgs::msg::BaseState>("joint_commands", 1);
    joint_state_pub_           = create_publisher<sensor_msgs::msg::JointState>("joint_states", 1);
    base_state_pub_            = create_publisher<thorvald_msgs::msg::BaseState>("base_state", 1);
    odom_pub_                  = create_publisher<nav_msgs::msg::Odometry>("odometry/base_raw", 1);
    motor_controller_data_pub_ = create_publisher<thorvald_msgs::msg::ControllerArray>("motor_controller_data", 1);
    battery_pub_               = create_publisher<thorvald_msgs::msg::BatteryArray>("battery_data", 1);
    can_base_pub_              = create_publisher<thorvald_msgs::msg::CANFrame>("can_frames_base_r", 10);
    can_device_pub_            = create_publisher<thorvald_msgs::msg::CANFrame>("can_frames_device_r", 10);

    // Services
    server_setup_controller_ = create_service<thorvald_msgs::srv::MotorControllerSetup>(
      "setup_motor_controller",
      std::bind(&BaseDriver::srvCallbackSetupController, this, _1, _2));
    server_home_steering_ = create_service<std_srvs::srv::Trigger>(
      "home_steering",
      std::bind(&BaseDriver::srvCallbackHomeSteering, this, _1, _2));
    server_current_pos_zero_ = create_service<thorvald_msgs::srv::CanID>(
      "set_home_count_current_pos_zero",
      std::bind(&BaseDriver::srvCallbackCurrentPosZero, this, _1, _2));
    server_reset_odom_ = create_service<std_srvs::srv::Trigger>(
      "reset_base_odom",
      std::bind(&BaseDriver::srvCallbackResetOdom, this, _1, _2));
    server_set_drive_params_ = create_service<thorvald_msgs::srv::DriveParams>(
      "set_drive_params",
      std::bind(&BaseDriver::srvCallbackSetDriveParams, this, _1, _2));
    server_safety_stop_ = create_service<std_srvs::srv::SetBool>(
      "safety_stop",
      std::bind(&BaseDriver::srvCallbackSafetyStop, this, _1, _2));
    server_mute_device_commands_ = create_service<std_srvs::srv::SetBool>(
      "mute_extra_can_device_commands",
      std::bind(&BaseDriver::srvCallbackMuteDeviceCommands, this, _1, _2));

    // TF broadcaster
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // Parameters
    declare_parameter("odom_frame_id", std::string("odom"));
    declare_parameter("enable_odom_tf", true);
    frame_id_    = get_parameter("odom_frame_id").as_string();
    broadcast_tf_ = get_parameter("enable_odom_tf").as_bool();

    // Load motor drive parameters
    // Each drive has a "type" sentinel: if -1.0, no more drives exist
    static const std::vector<std::string> drive_keys = {
      "node", "x", "y", "leg_mesh",
      "r_wheel", "prp_gr_rt", "prp_enc_ppr", "prp_max_rpm", "prp_sim_a",
      "str_gr_rt", "str_enc_ppr", "str_lim", "str_sim_v"
    };
    int drive_idx = 0;
    while (true) {
      std::string pfx = "motor_drives.drive" + std::to_string(drive_idx);
      declare_parameter(pfx + ".type", -1.0);
      double type_val = get_parameter(pfx + ".type").as_double();
      if (type_val < 0) { break; }

      std::map<std::string, double> drive;
      drive["type"] = type_val;
      for (const auto & k : drive_keys) {
        declare_parameter(pfx + "." + k, 0.0);
        drive[k] = get_parameter(pfx + "." + k).as_double();
      }

      std::string steer_name = "steering" + std::to_string(drive_idx);
      std::string wheel_name = "wheel" + std::to_string(drive_idx);
      joint_names_.push_back(steer_name);
      joint_names_.push_back(wheel_name);
      motor_drives_.push_back(drive);
      drive_idx++;
    }

    // Load battery parameters
    static const std::vector<std::string> bat_keys = {"node", "x", "y", "z", "yaw", "bat_mesh"};
    int bat_idx = 0;
    while (true) {
      std::string pfx = "batteries.bat" + std::to_string(bat_idx);
      declare_parameter(pfx + ".type", -1.0);
      double btype = get_parameter(pfx + ".type").as_double();
      if (btype < 0) { break; }

      std::map<std::string, double> bat;
      bat["type"] = btype;
      for (const auto & k : bat_keys) {
        declare_parameter(pfx + "." + k, 0.0);
        bat[k] = get_parameter(pfx + "." + k).as_double();
      }
      batteries_.push_back(bat);
      bat_idx++;
    }

    RCLCPP_INFO(get_logger(), "Loaded %zu drive(s) and %zu battery module(s).",
      motor_drives_.size(), batteries_.size());

    emergency_stop_         = false;
    latest_base_command_time_ = this->now();
    command_timeout_time_   = 0.5;
    prev_bat_time_          = this->now();
    bat_duration_           = rclcpp::Duration::from_seconds(0.000001);
    mute_device_commands_   = false;
    can_device_t_frame_count_ = 0;

    // Load kinematics plugin
    declare_parameter("base_calculator", std::string("pltf_clc_std::PltfClcStd"));
    std::string pltf_type = get_parameter("base_calculator").as_string();
    RCLCPP_INFO(get_logger(), "Using base calculator plugin: %s", pltf_type.c_str());
    loadClcPlugin(pltf_type);
  }

  ~BaseDriver()
  {
    if (can_ctrl_pltf_ != nullptr) {
      delete can_ctrl_pltf_;
    }
  }

  bool loadClcPlugin(const std::string & type)
  {
    try {
      pltf_clc_type_ = pltf_clc_loader_.createSharedInstance(type);
      pltf_clc_type_->initialize(motor_drives_);

      geometry_msgs::msg::Twist zero_twist;
      pltf_clc_type_->calculateCommand(zero_twist, latest_base_command_);
    } catch (const pluginlib::PluginlibException & ex) {
      RCLCPP_ERROR(get_logger(), "Plugin load failed: %s", ex.what());
      return false;
    }
    return true;
  }

  bool initPltf(int can_adapter_type)
  {
    return can_ctrl_pltf_->initPltf(can_adapter_type, motor_drives_, batteries_);
  }

  void evalCanBuffer()
  {
    std::vector<thorvald_msgs::msg::CANFrame> can_msgs_base;
    std::vector<thorvald_msgs::msg::CANFrame> can_msgs_device;
    can_ctrl_pltf_->evalCanBuffer(can_msgs_base, can_msgs_device);

    for (const auto & f : can_msgs_base)   { can_base_pub_->publish(f); }
    for (const auto & f : can_msgs_device) { can_device_pub_->publish(f); }
  }

  void publishJointCommands()
  {
    joint_command_pub_->publish(latest_base_command_);
  }

  void handleFeedback()
  {
    // Motor feedback
    thorvald_msgs::msg::BaseState base_state;
    thorvald_msgs::msg::ControllerArray controller_array;

    can_ctrl_pltf_->getBaseState(base_state);
    base_state_pub_->publish(base_state);
    can_ctrl_pltf_->getControllerArray(controller_array);

    nav_msgs::msg::Odometry odom_msg;
    if (pltf_clc_type_->calculateOdometry(base_state, odom_msg, this->now())) {
      if (broadcast_tf_) {
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp    = this->now();
        t.header.frame_id = frame_id_;
        t.child_frame_id  = "base_link";
        t.transform.translation.x = odom_msg.pose.pose.position.x;
        t.transform.translation.y = odom_msg.pose.pose.position.y;
        t.transform.translation.z = odom_msg.pose.pose.position.z;
        t.transform.rotation      = odom_msg.pose.pose.orientation;
        tf_broadcaster_->sendTransform(t);
      }
      odom_msg.header.frame_id = frame_id_;
      odom_pub_->publish(odom_msg);
    } else {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Odometry returned NaN");
    }

    // Joint states
    sensor_msgs::msg::JointState js;
    js.header.stamp = this->now();
    js.name = joint_names_;
    for (unsigned int i = 0; i < base_state.steer_pos.size(); i++) {
      js.position.push_back(base_state.steer_pos[i]);
      js.position.push_back(base_state.prop_pos[i]);
    }
    joint_state_pub_->publish(js);

    // Controller data
    controller_array.header.stamp = this->now();
    motor_controller_data_pub_->publish(controller_array);

    // Battery feedback
    thorvald_msgs::msg::BatteryArray batt_array;
    can_ctrl_pltf_->getAllBatteryVars(batt_array);
    battery_pub_->publish(batt_array);
  }

  void sendDriveCommands()
  {
    rclcpp::Time now = this->now();
    if ((now - latest_base_command_time_) > bat_duration_ ||   // reuse duration type
        (now > latest_base_command_time_ + rclcpp::Duration::from_seconds(command_timeout_time_)) ||
        emergency_stop_)
    {
      // Only zero speed on timeout/estop
      if ((now > latest_base_command_time_ + rclcpp::Duration::from_seconds(command_timeout_time_)) ||
          emergency_stop_)
      {
        pltf_clc_type_->setZeroSpeed(latest_base_command_);
      }
    }
    if ((this->now() > latest_base_command_time_ + rclcpp::Duration::from_seconds(command_timeout_time_)) ||
        emergency_stop_)
    {
      pltf_clc_type_->setZeroSpeed(latest_base_command_);
    }
    can_ctrl_pltf_->sendCommandsToAllDrives(latest_base_command_);
  }

  void sendDeviceCommands()
  {
    if (can_device_t_frame_count_ < 100) {
      if (!mute_device_commands_) {
        for (const auto & f : can_device_t_frames_) {
          can_ctrl_pltf_->sendDeviceCommand(f);
        }
      }
    } else {
      RCLCPP_WARN(get_logger(), "Too many CAN device msgs in queue");
    }
    can_device_t_frame_count_ = 0;
    can_device_t_frames_.clear();
  }

  void requestBatteryParameters()
  {
    if (this->now() > prev_bat_time_ + bat_duration_) {
      can_ctrl_pltf_->requestNextVarsAllBatteries(1);
      prev_bat_time_ = this->now();
    }
  }

  void sendSimCommands()
  {
    if (this->now() > latest_base_command_time_ + rclcpp::Duration::from_seconds(command_timeout_time_)) {
      pltf_clc_type_->setZeroSpeed(latest_base_command_);
    }
    can_ctrl_pltf_->simulateAllDrives(latest_base_command_);
  }

private:
  // Subscribers
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_sub_;
  rclcpp::Subscription<thorvald_msgs::msg::CANFrame>::SharedPtr can_device_t_sub_;

  // Publishers
  rclcpp::Publisher<thorvald_msgs::msg::BaseState>::SharedPtr joint_command_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::Publisher<thorvald_msgs::msg::BaseState>::SharedPtr base_state_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<thorvald_msgs::msg::ControllerArray>::SharedPtr motor_controller_data_pub_;
  rclcpp::Publisher<thorvald_msgs::msg::BatteryArray>::SharedPtr battery_pub_;
  rclcpp::Publisher<thorvald_msgs::msg::CANFrame>::SharedPtr can_base_pub_;
  rclcpp::Publisher<thorvald_msgs::msg::CANFrame>::SharedPtr can_device_pub_;

  // Services
  rclcpp::Service<thorvald_msgs::srv::MotorControllerSetup>::SharedPtr server_setup_controller_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr server_home_steering_;
  rclcpp::Service<thorvald_msgs::srv::CanID>::SharedPtr server_current_pos_zero_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr server_reset_odom_;
  rclcpp::Service<thorvald_msgs::srv::DriveParams>::SharedPtr server_set_drive_params_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr server_safety_stop_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr server_mute_device_commands_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  CanCtrlPltf * can_ctrl_pltf_;

  bool mute_device_commands_;
  int can_device_t_frame_count_;
  std::vector<thorvald_msgs::msg::CANFrame> can_device_t_frames_;

  std::vector<std::string> joint_names_;

  thorvald_msgs::msg::BaseState latest_base_command_;
  rclcpp::Time latest_base_command_time_;
  double command_timeout_time_;

  pluginlib::ClassLoader<pltf_clc_itf::PltfClcItf> pltf_clc_loader_;
  std::shared_ptr<pltf_clc_itf::PltfClcItf> pltf_clc_type_;

  std::string frame_id_;
  bool broadcast_tf_;
  std::vector<std::map<std::string, double>> motor_drives_;
  std::vector<std::map<std::string, double>> batteries_;

  rclcpp::Time prev_bat_time_;
  rclcpp::Duration bat_duration_{0, 0};

  bool emergency_stop_;

  // Service callbacks
  void srvCallbackSetupController(
    const std::shared_ptr<thorvald_msgs::srv::MotorControllerSetup::Request> req,
    std::shared_ptr<thorvald_msgs::srv::MotorControllerSetup::Response> res)
  {
    std::vector<std::string> setup_map;
    can_ctrl_pltf_->getControllerSetupMap(req->can_id, setup_map);

    std::ostringstream map_ss;
    map_ss << "\n\nCONTROLLER SETUP MAP\nSETUP ID    SETUP VALUE\n";
    for (int i = 0; i < static_cast<int>(setup_map.size()); i++) {
      map_ss << std::right << std::setw(4) << std::setfill(' ') << i
             << "        " << setup_map[i] << "\n";
    }

    if (req->setup_id < 0 || req->setup_id >= static_cast<int>(setup_map.size())) {
      res->message = "Invalid setup ID. Try setup_id=0 for the setup map.";
      res->status = -2;
    } else if (req->setup_id == 0) {
      res->message = map_ss.str();
      res->status = 0;
    } else {
      RCLCPP_INFO(get_logger(), "Setup PDO → CAN ID %d: id=%d value=%d",
        req->can_id, req->setup_id, req->setup_value);
      can_ctrl_pltf_->setupMotorController(req->can_id, req->setup_id, req->setup_value);
      std::ostringstream msg;
      msg << "Sent to CAN ID=" << req->can_id << ", " << setup_map[req->setup_id]
          << "=" << req->setup_value;
      res->message = msg.str();
      res->status = 1;
    }
  }

  void srvCallbackHomeSteering(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    RCLCPP_INFO(get_logger(), "Homing steering");
    if (can_ctrl_pltf_->homeSteeringAll()) {
      res->success = true;
      res->message = "Homing commands sent";
    } else {
      res->success = false;
      res->message = "Homing failed";
    }
  }

  void srvCallbackCurrentPosZero(
    const std::shared_ptr<thorvald_msgs::srv::CanID::Request> req,
    std::shared_ptr<thorvald_msgs::srv::CanID::Response> res)
  {
    if (req->can_id == -1) {
      can_ctrl_pltf_->setCurrentPosAsZeroAll();
      res->success = true;
      res->message = "Setting ALL home counts so that current positions are zero";
    } else if (req->can_id > 0) {
      can_ctrl_pltf_->setCurrentPosAsZero(req->can_id);
      res->success = true;
      res->message = "Setting home count of selected id so that current position is zero";
    } else {
      res->success = false;
      res->message = "can_id must be positive or -1 (sets home count for all drives)";
    }
  }

  void srvCallbackResetOdom(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    RCLCPP_INFO(get_logger(), "Resetting base odometry to zero");
    pltf_clc_type_->zeroOdometryPose();
    res->success = true;
    res->message = "Resetting base odometry. x=0, y=0, yaw=0";
  }

  void srvCallbackSetDriveParams(
    const std::shared_ptr<thorvald_msgs::srv::DriveParams::Request> req,
    std::shared_ptr<thorvald_msgs::srv::DriveParams::Response> res)
  {
    int num_drives = static_cast<int>(motor_drives_.size());
    res->message = "Parameters have been set";
    res->status = 0;

    for (unsigned int i = 0; i < req->params.size(); i++) {
      int idx = req->params[i].index;
      if (idx >= 0 && idx < num_drives && motor_drives_[idx].count(req->params[i].type)) {
        motor_drives_[idx][req->params[i].type] = req->params[i].value;
        res->status++;
      } else {
        res->message = "One or more params could not be set";
      }
    }

    pltf_clc_type_->setParams(motor_drives_);
    can_ctrl_pltf_->setParams(motor_drives_);
  }

  void srvCallbackSafetyStop(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> req,
    std::shared_ptr<std_srvs::srv::SetBool::Response> res)
  {
    emergency_stop_ = req->data;
    RCLCPP_INFO(get_logger(), "Safety stop %s", emergency_stop_ ? "ENABLED" : "disabled");
    res->success = true;
    res->message = emergency_stop_ ? "Safety stop Enabled" : "Safety stop Disabled";
  }

  void srvCallbackMuteDeviceCommands(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> req,
    std::shared_ptr<std_srvs::srv::SetBool::Response> res)
  {
    mute_device_commands_ = req->data;
    RCLCPP_INFO(get_logger(), "%s CAN commands to non-drive/battery devices",
      mute_device_commands_ ? "Muting" : "Unmuting");
    can_device_t_frame_count_ = 0;
    can_device_t_frames_.clear();
    res->success = true;
    res->message = mute_device_commands_
      ? "Muting CAN commands to non drive/battery devices"
      : "Unmuting CAN commands to non drive/battery devices";
  }

  // Topic callbacks
  void twistCallback(const geometry_msgs::msg::Twist::SharedPtr twist_in)
  {
    latest_base_command_time_ = this->now();
    pltf_clc_type_->calculateCommand(*twist_in, latest_base_command_);
  }

  void canDeviceTCallback(const thorvald_msgs::msg::CANFrame::SharedPtr can_frame)
  {
    can_device_t_frame_count_++;
    can_device_t_frames_.push_back(*can_frame);
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<BaseDriver>();

  // Determine mode
  node->declare_parameter("simple_sim", true);
  bool simple_sim = node->get_parameter("simple_sim").as_bool();
  RCLCPP_INFO(node->get_logger(), "simple_sim = %s", simple_sim ? "true" : "false");

  int can_adapter_type = simple_sim
    ? CanCtrlPltf::CAN_ADAPTER_NONE
    : CanCtrlPltf::CAN_ADAPTER_PCAN_USB;

  if (!node->initPltf(can_adapter_type)) {
    RCLCPP_ERROR(node->get_logger(),
      "Failed to init robot base. Check CAN adapter type, connection and drivers.");
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::Rate rate(20);

  if (simple_sim) {
    RCLCPP_INFO(node->get_logger(), "Simulating feedback");
    while (rclcpp::ok()) {
      node->publishJointCommands();
      node->sendSimCommands();
      node->handleFeedback();
      rclcpp::spin_some(node);
      rate.sleep();
    }
  } else {
    RCLCPP_INFO(node->get_logger(), "Assuming feedback from real robot");
    while (rclcpp::ok()) {
      node->publishJointCommands();
      node->sendDriveCommands();
      node->sendDeviceCommands();
      node->evalCanBuffer();
      node->handleFeedback();
      rclcpp::spin_some(node);
      rate.sleep();
    }
  }

  rclcpp::shutdown();
  return 0;
}
