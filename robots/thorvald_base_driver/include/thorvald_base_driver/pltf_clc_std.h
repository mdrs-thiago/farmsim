#ifndef THORVALD_BASE_DRIVER_PLTF_CLC_STD_H_
#define THORVALD_BASE_DRIVER_PLTF_CLC_STD_H_

#include <thorvald_base_driver/pltf_clc_itf.h>
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <array>
#include <iostream>

namespace pltf_clc_std
{

class PltfClcStd : public pltf_clc_itf::PltfClcItf
{
public:
  PltfClcStd() {}

  void initialize(std::vector<std::map<std::string, double>> motor_drives) override
  {
    robot_x_ = 0;
    robot_y_ = 0;
    robot_yaw_ = 0;
    previous_odom_timestep_ = std::chrono::steady_clock::now();
    setParams(motor_drives);
  }

  bool setParams(std::vector<std::map<std::string, double>> motor_drives) override
  {
    motor_drives_ = motor_drives;
    number_of_drives_ = static_cast<int>(motor_drives_.size());
    turning_angles_.clear();
    for (int i = 0; i < number_of_drives_; i++) {
      turning_angles_.push_back(-normalizeAngleHalfPi(atan(motor_drives[i]["x"] / motor_drives[i]["y"])));
    }
    return true;
  }

  bool calculateCommand(
    geometry_msgs::msg::Twist twist_in,
    thorvald_msgs::msg::BaseState & joint_states_out) override
  {
    double vx = twist_in.linear.x;
    double vy = twist_in.linear.y;
    double wz = twist_in.angular.z;

    std::vector<double> speed(number_of_drives_);
    std::vector<double> steering(number_of_drives_);

    if (wz != 0) {
      double turn_rad_d = sqrt(vx * vx + vy * vy) / wz;
      double turn_rad_ang = atan2(vy, vx);
      double turn_rad_x = -turn_rad_d * sin(turn_rad_ang);
      double turn_rad_y =  turn_rad_d * cos(turn_rad_ang);

      for (int i = 0; i < number_of_drives_; i++) {
        double drive_x = motor_drives_[i]["x"];
        double drive_y = motor_drives_[i]["y"];
        double ds = normalizeAngle(-atan2((turn_rad_x - drive_x), (turn_rad_y - drive_y)) + M_PI * (wz < 0));
        speed[i]   = sqrt(pow(turn_rad_x - drive_x, 2) + pow(turn_rad_y - drive_y, 2)) * fabs(wz);
        steering[i] = ds;
      }
    } else {
      for (int i = 0; i < number_of_drives_; i++) {
        steering[i] = atan2(vy, vx);
        speed[i]    = sqrt(vx * vx + vy * vy);
      }
    }

    joint_states_out.prop_speed.resize(number_of_drives_);
    joint_states_out.steer_pos.resize(number_of_drives_);
    joint_states_out.steer_max_speed.resize(number_of_drives_);
    joint_states_out.channel.resize(number_of_drives_);
    for (int i = 0; i < number_of_drives_; i++) {
      joint_states_out.steer_pos[i]  = steering[i];
      joint_states_out.prop_speed[i] = speed[i];
    }
    return true;
  }

  bool setZeroSpeed(thorvald_msgs::msg::BaseState & joint_states_out) override
  {
    for (auto & s : joint_states_out.prop_speed) {
      s = 0;
    }
    return true;
  }

  bool calculateOdometry(
    thorvald_msgs::msg::BaseState joint_states_in,
    nav_msgs::msg::Odometry & odom_out,
    rclcpp::Time stamp) override
  {
    joint_states_in.prop_speed.resize(number_of_drives_);
    joint_states_in.prop_pos.resize(number_of_drives_);
    joint_states_in.steer_pos.resize(number_of_drives_);

    auto now_tp = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now_tp - previous_odom_timestep_).count();
    if (dt > 0.5) {
      dt = 0;
    }

    if (number_of_drives_ > 1) {
      double rot_y = 0;
      double rot_x = 0;
      int counted = 0;
      int inf_count = 0;
      int iterations = number_of_drives_ * 10;

      for (int i = 0; i < iterations; i++) {
        int rand_drive_1 = 0, rand_drive_2 = 0;
        while (rand_drive_1 == rand_drive_2) {
          rand_drive_1 = rand() % number_of_drives_;
          rand_drive_2 = rand() % number_of_drives_;
        }

        double x_1 = motor_drives_[rand_drive_1]["x"];
        double y_1 = motor_drives_[rand_drive_1]["y"];
        double a_1 = -joint_states_in.steer_pos[rand_drive_1];
        double x_2 = motor_drives_[rand_drive_2]["x"];
        double y_2 = motor_drives_[rand_drive_2]["y"];
        double a_2 = -joint_states_in.steer_pos[rand_drive_2];

        double d12_x = x_2 - x_1;
        double d12_y = y_2 - y_1;
        double v1_unitv_x = sin(a_1), v1_unitv_y = cos(a_1);
        double v2_unitv_x = sin(a_2), v2_unitv_y = cos(a_2);
        double d12_len = sqrt(d12_x * d12_x + d12_y * d12_y);
        double d12_dot_v1 = (d12_x * v1_unitv_x + d12_y * v1_unitv_y) / d12_len;
        double d12_dot_v2 = (d12_x * v2_unitv_x + d12_y * v2_unitv_y) / d12_len;

        if (joint_states_in.prop_speed[rand_drive_1] == 0 &&
            joint_states_in.prop_speed[rand_drive_2] == 0)
        {
          inf_count++;
        } else if (fabs(d12_dot_v1) > 0.9 && fabs(d12_dot_v2) > 0.9) {
          double vx1 = joint_states_in.prop_speed[rand_drive_1] * cos(joint_states_in.steer_pos[rand_drive_1]);
          double vy1 = joint_states_in.prop_speed[rand_drive_1] * sin(joint_states_in.steer_pos[rand_drive_1]);
          double vx2 = joint_states_in.prop_speed[rand_drive_2] * cos(joint_states_in.steer_pos[rand_drive_2]);
          double vy2 = joint_states_in.prop_speed[rand_drive_2] * sin(joint_states_in.steer_pos[rand_drive_2]);
          if (vx1 == vx2) {
            inf_count++;
          } else {
            if (x_2 == x_1) {
              x_2 += 0.000001 * (1 - 2 * (1 && rand() % 2));
            }
            double a1 = (y_2 - y_1) / (x_2 - x_1);
            double b1 = y_2 - a1 * x_2;
            double vx1t = x_1 + vx1, vy1t = y_1 + vy1;
            double vx2t = x_2 + vx2, vy2t = y_2 + vy2;
            if (vx2t == vx1t) {
              inf_count++;
            } else {
              double a2 = (vy2t - vy1t) / (vx2t - vx1t);
              double b2 = vy2t - a2 * vx2t;
              if (a1 == a2) {
                inf_count++;
              } else {
                double trx = (b2 - b1) / (a1 - a2);
                double try_ = a1 * trx + b1;
                rot_x += trx;
                rot_y += try_;
                counted++;
              }
            }
          }
        } else if (fabs(d12_dot_v1) > 0.9 && fabs(d12_dot_v2) < 0.1) {
          rot_y += y_1; rot_x += x_1; counted++;
        } else if (fabs(d12_dot_v1) < 0.1 && fabs(d12_dot_v2) > 0.9) {
          rot_y += y_2; rot_x += x_2; counted++;
        } else if (tan(a_1) == tan(a_2)) {
          inf_count++;
        } else {
          double try_ = (x_1 - x_2 - y_1 * tan(a_1) + y_2 * tan(a_2)) / (tan(a_2) - tan(a_1));
          double trx = x_1 - tan(a_1) * (y_1 - try_);
          if (isinf(try_) || isinf(trx)) {
            inf_count++;
          } else if (!isnan(try_) && !isnan(trx)) {
            rot_y += try_; rot_x += trx; counted++;
          }
        }
      }

      if (counted > 0) {
        rot_y /= counted;
        rot_x /= counted;
      }

      bool translating = (static_cast<double>(inf_count) / static_cast<double>(iterations)) > 0.5;

      double vx = 0, vy = 0, wz = 0;

      if (translating) {
        for (int i = 0; i < number_of_drives_; i++) {
          vx += joint_states_in.prop_speed[i] * cos(joint_states_in.steer_pos[i]) / number_of_drives_;
          vy += joint_states_in.prop_speed[i] * sin(joint_states_in.steer_pos[i]) / number_of_drives_;
        }
      } else {
        int wz_used = 0;
        for (int i = 0; i < number_of_drives_; i++) {
          double tx = motor_drives_[i]["x"];
          double ty = motor_drives_[i]["y"];
          double rad = sqrt(pow(tx - rot_x, 2) + pow(ty - rot_y, 2));
          double tvx = joint_states_in.prop_speed[i] * cos(joint_states_in.steer_pos[i]);
          double tvy = joint_states_in.prop_speed[i] * sin(joint_states_in.steer_pos[i]);
          double twz = ((tx - rot_x) * tvy - (ty - rot_y) * tvx) / (rad * rad);
          if (twz != 0) { wz_used++; }
          wz += twz;
        }
        if (wz_used > 0) { wz /= wz_used; }

        double rad_ang = atan2(rot_x, rot_y);
        if (isinf(rad_ang) || isnan(rad_ang)) {
          rad_ang = atan2(rot_x + 0.0000001, rot_y);
        }
        double rad_d = sqrt(rot_y * rot_y + rot_x * rot_x);
        vx =  rad_d * wz * cos(rad_ang);
        vy = -rad_d * wz * sin(rad_ang);
      }

      if (!isnan(vx) && !isnan(vy) && !isnan(wz)) {
        robot_x_   += (vx * cos(robot_yaw_) - vy * sin(robot_yaw_)) * dt;
        robot_y_   += (vx * sin(robot_yaw_) + vy * cos(robot_yaw_)) * dt;
        robot_yaw_ += wz * dt;
      } else {
        std::cerr << "[pltf_clc_std] odom NaN (translating=" << translating << ")\n";
        return false;
      }

      odom_out.header.frame_id = "odom";
      odom_out.child_frame_id  = "base_link";
      odom_out.header.stamp    = stamp;

      odom_out.pose.pose.position.x = robot_x_;
      odom_out.pose.pose.position.y = robot_y_;
      odom_out.pose.pose.position.z = 0;

      tf2::Quaternion q;
      q.setRPY(0, 0, robot_yaw_);
      odom_out.pose.pose.orientation = tf2::toMsg(q);

      std::array<double, 36> pose_cov = {
        0.001, 0, 0, 0, 0, 0,
        0, 0.001, 0, 0, 0, 0,
        0, 0, 0.001, 0, 0, 0,
        0, 0, 0, 0.001, 0, 0,
        0, 0, 0, 0, 0.001, 0,
        0, 0, 0, 0, 0, 0.03};
      for (int k = 0; k < 36; k++) { odom_out.pose.covariance[k] = pose_cov[k]; }

      odom_out.twist.twist.linear.x  = vx;
      odom_out.twist.twist.linear.y  = vy;
      odom_out.twist.twist.angular.z = wz;

      std::array<double, 36> twist_cov = {
        0.001, 0, 0, 0, 0, 0,
        0, 0.001, 0, 0, 0, 0,
        0, 0, 0.001, 0, 0, 0,
        0, 0, 0, 0.001, 0, 0,
        0, 0, 0, 0, 0.001, 0,
        0, 0, 0, 0, 0, 0.03};
      for (int k = 0; k < 36; k++) { odom_out.twist.covariance[k] = twist_cov[k]; }

      previous_odom_timestep_ = now_tp;
      return true;
    }

    return false;
  }

  void zeroOdometryPose() override
  {
    robot_x_ = 0;
    robot_y_ = 0;
    robot_yaw_ = 0;
  }

private:
  int number_of_drives_;
  double robot_x_, robot_y_, robot_yaw_;
  std::chrono::steady_clock::time_point previous_odom_timestep_;
  std::vector<double> turning_angles_;
  std::vector<std::map<std::string, double>> motor_drives_;

  static double normalizeAngle(double a)
  {
    return atan2(sin(a), cos(a));
  }

  static double normalizeAngleHalfPi(double angle)
  {
    while (fabs(angle) > M_PI / 2) {
      angle += M_PI * (-1 * angle / fabs(angle));
    }
    return angle;
  }
};

}  // namespace pltf_clc_std

#endif  // THORVALD_BASE_DRIVER_PLTF_CLC_STD_H_
