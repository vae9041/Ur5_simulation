#include <memory>
#include <vector>
#include <cmath>

#include <rclcpp/rclcpp.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/msg/collision_object.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <shape_msgs/msg/solid_primitive.hpp>


const double tau = 2.0 * M_PI;

class PickAndPlace
{
public:
  explicit PickAndPlace(const rclcpp::Node::SharedPtr& node)
  : move_group_(node, "ur5_manipulator"),
    gripper_(node, "robotiq_gripper"),
    planning_scene_interface_(),
    logger_(rclcpp::get_logger("PickAndPlace")),
    node_(node)
  {
    move_group_.setPoseReferenceFrame("base_link");

    // Optional: default motion parameters (can be overridden per action)
    move_group_.setMaxVelocityScalingFactor(1.0);
    move_group_.setMaxAccelerationScalingFactor(1.0);
    move_group_.setPlanningTime(10.0);
    move_group_.allowReplanning(true);
    move_group_.setGoalTolerance(0.03);
  }

  void close_gripper()
  {
    gripper_.setJointValueTarget("robotiq_85_left_knuckle_joint", 0.5);
    gripper_.move();
  }

  void open_gripper()
  {
    gripper_.setJointValueTarget("robotiq_85_left_knuckle_joint", 0.0);
    gripper_.move();
  }

  void pick()
  {
    // ----------------------------
    // 1) Define pick and pre-pick
    // ----------------------------
    geometry_msgs::msg::Pose pick_pose = makePickPose();
    geometry_msgs::msg::Pose pre_pick_pose = pick_pose;

    // Pre-pick is 0.20 m above pick along +Z in the reference frame
    pre_pick_pose.position.z += 0.20;

    // ----------------------------
    // 2) Plan & execute to pre-pick
    // ----------------------------
    if (!planAndExecutePose(pre_pick_pose, "tool0", "pre-pick"))
    {
      RCLCPP_ERROR(logger_, "Failed to reach pre-pick pose. Aborting pick.");
      return;
    }

    // ----------------------------
    // 3) Cartesian linear move: pre-pick -> pick (straight down)
    // ----------------------------
    if (!executeCartesianDown(pre_pick_pose, pick_pose))
    {
      RCLCPP_ERROR(logger_, "Cartesian approach failed. Aborting pick.");
      return;
    }

    // ----------------------------
    // 4) Close gripper
    // ----------------------------
    close_gripper();
    rclcpp::sleep_for(std::chrono::seconds(1));
  }

  void place()
  {
    // Keep your original "place" behavior as a normal pose target.
    geometry_msgs::msg::Pose place_pose = makePlacePose();

    if (!planAndExecutePose(place_pose, "tool0", "place"))
    {
      RCLCPP_ERROR(logger_, "Motion planning for place failed!");
      return;
    }

    RCLCPP_INFO(logger_, "Place motion execution completed.");
  }

  void addCollisionObjects()
  {
    collision_objects_.resize(3);

    // Table 1
    collision_objects_[0].id = "table1";
    collision_objects_[0].header.frame_id = "world";
    collision_objects_[0].primitives.resize(1);
    collision_objects_[0].primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
    collision_objects_[0].primitives[0].dimensions = {0.608, 2.0, 1.0};
    collision_objects_[0].primitive_poses.resize(1);
    collision_objects_[0].primitive_poses[0].position.x = 0.676;
    collision_objects_[0].primitive_poses[0].position.y = 0.0;
    collision_objects_[0].primitive_poses[0].position.z = 0.5;
    collision_objects_[0].operation = moveit_msgs::msg::CollisionObject::ADD;

    // Table 2
    collision_objects_[1].id = "table2";
    collision_objects_[1].header.frame_id = "world";
    collision_objects_[1].primitives.resize(1);
    collision_objects_[1].primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
    collision_objects_[1].primitives[0].dimensions = {1.3, 0.8, 1.0};
    collision_objects_[1].primitive_poses.resize(1);
    collision_objects_[1].primitive_poses[0].position.x = 0.0;
    collision_objects_[1].primitive_poses[0].position.y = 0.69;
    collision_objects_[1].primitive_poses[0].position.z = 0.5;
    collision_objects_[1].operation = moveit_msgs::msg::CollisionObject::ADD;

    // Basement (cylinder)
    collision_objects_[2].id = "basement";
    collision_objects_[2].header.frame_id = "world";
    collision_objects_[2].primitives.resize(1);
    collision_objects_[2].primitives[0].type = shape_msgs::msg::SolidPrimitive::CYLINDER;
    collision_objects_[2].primitives[0].dimensions = {0.8, 0.2}; // height, radius
    collision_objects_[2].primitive_poses.resize(1);
    collision_objects_[2].primitive_poses[0].position.x = 0.0;
    collision_objects_[2].primitive_poses[0].position.y = 0.0;
    collision_objects_[2].primitive_poses[0].position.z = 0.4;
    collision_objects_[2].operation = moveit_msgs::msg::CollisionObject::ADD;

    // Object in planning scene (tall box)
    //collision_objects_[3].id = "object";
    //collision_objects_[3].header.frame_id = "world";
    //collision_objects_[3].primitives.resize(1);
    //collision_objects_[3].primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
    //collision_objects_[3].primitives[0].dimensions = {0.02, 0.02, 0.2};
    //collision_objects_[3].primitive_poses.resize(1);
    //collision_objects_[3].primitive_poses[0].position.x = 0.5;
    //collision_objects_[3].primitive_poses[0].position.y = 0.0;
    //collision_objects_[3].primitive_poses[0].position.z = 1.1;
    //collision_objects_[3].operation = moveit_msgs::msg::CollisionObject::ADD;

    planning_scene_interface_.applyCollisionObjects(collision_objects_);
    RCLCPP_INFO(logger_, "Collision objects added to the planning scene.");
  }

private:
  // ----------------------------
  // Helper: Create pick pose
  // ----------------------------
  geometry_msgs::msg::Pose makePickPose() const
  {
    geometry_msgs::msg::Pose pick_pose;
    tf2::Quaternion q;
    q.setRPY(1.57, 0.0, -3.14);
    pick_pose.orientation = tf2::toMsg(q);

    pick_pose.position.x = 0.495;
    pick_pose.position.y = -0.140;
    pick_pose.position.z = 0.318;

    return pick_pose;
  }

  // ----------------------------
  // Helper: Create place pose
  // ----------------------------
  geometry_msgs::msg::Pose makePlacePose() const
  {
    geometry_msgs::msg::Pose place_pose;
    tf2::Quaternion q;
    q.setRPY(-3.14, 0.0, -3.14);
    place_pose.orientation = tf2::toMsg(q);

    place_pose.position.x = 0.088;
    place_pose.position.y = 0.751;
    place_pose.position.z = 0.486;

    return place_pose;
  }

  // ----------------------------
  // Helper: Plan + execute to a pose target (normal planning)
  // ----------------------------
  bool planAndExecutePose(const geometry_msgs::msg::Pose& target,
                          const std::string& ee_link,
                          const std::string& label)
  {
    move_group_.setPoseTarget(target, ee_link);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const bool success = (move_group_.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);

    RCLCPP_INFO(logger_, "Planning to %s pose: %s", label.c_str(), success ? "SUCCESS" : "FAILED");

    if (!success)
      return false;

    move_group_.execute(plan);
    RCLCPP_INFO(logger_, "Execution to %s pose completed.", label.c_str());
    return true;
  }

  // ----------------------------
  // Helper: Cartesian approach from pre-pick to pick
  // ----------------------------
  bool executeCartesianDown(const geometry_msgs::msg::Pose& pre_pick,
                            const geometry_msgs::msg::Pose& pick)
  {
    // Cartesian path settings
    const double eef_step = 0.01;       // 1 cm resolution for interpolation
    const double jump_threshold = 0.0;  // 0.0 disables jump threshold checking (common in tutorials)

    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.reserve(2);

    // Keep the same orientation, just move linearly in Cartesian space
    waypoints.push_back(pre_pick);
    waypoints.push_back(pick);

    moveit_msgs::msg::RobotTrajectory trajectory;
    const double fraction = move_group_.computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);

    RCLCPP_INFO(logger_, "Cartesian path fraction: %.2f", fraction);

    // In practice, require a high fraction for a reliable straight approach
    if (fraction < 0.95)
    {
      RCLCPP_WARN(logger_, "Cartesian path not fully feasible (fraction < 0.95).");
      return false;
    }

    // Execute the cartesian trajectory
    moveit::planning_interface::MoveGroupInterface::Plan cart_plan;
    cart_plan.trajectory_ = trajectory;

    move_group_.execute(cart_plan);
    RCLCPP_INFO(logger_, "Cartesian approach execution completed.");
    return true;
  }

private:
  moveit::planning_interface::MoveGroupInterface move_group_;
  moveit::planning_interface::MoveGroupInterface gripper_;
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;

  std::vector<moveit_msgs::msg::CollisionObject> collision_objects_;

  rclcpp::Logger logger_;
  rclcpp::Node::SharedPtr node_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("pick_and_place_node");

  PickAndPlace pick_and_place(node);

  pick_and_place.addCollisionObjects();
  rclcpp::sleep_for(std::chrono::seconds(1));

  // Pick sequence (pre-pick + cartesian down + grasp)
  pick_and_place.pick();
  rclcpp::sleep_for(std::chrono::seconds(1));

  // Place sequence
  pick_and_place.place();
  rclcpp::sleep_for(std::chrono::seconds(1));

  // Release
  pick_and_place.open_gripper();
  rclcpp::sleep_for(std::chrono::seconds(1));

  rclcpp::shutdown();
  return 0;
}