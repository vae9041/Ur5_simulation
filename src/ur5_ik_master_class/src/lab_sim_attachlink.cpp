#include <memory>
#include <vector>
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <frame_transform/srv/frame_transform.hpp>
#include <linkattacher_msgs/srv/attach_link.hpp>
#include <linkattacher_msgs/srv/detach_link.hpp>

const double tau = 2 * M_PI; // tau = 2π handles the rotation more intuitively

static constexpr double kBaseLinkInWorldZ = 0.730;
static constexpr double kTableCenterZInWorld = 0.3585;
static constexpr double kTableThickness = 0.70;
static constexpr double kGraspClearanceAboveTable = 0.02;
static constexpr double kGraspZOffset = 0.15;

class PickAndPlace
{
private:
    rclcpp::Node::SharedPtr node_;
    moveit::planning_interface::MoveGroupInterface move_group_;
    moveit::planning_interface::MoveGroupInterface gripper_;
    rclcpp::Client<frame_transform::srv::FrameTransform>::SharedPtr client_picking_pose_;
    rclcpp::Client<linkattacher_msgs::srv::AttachLink>::SharedPtr attach_client_;
    rclcpp::Client<linkattacher_msgs::srv::DetachLink>::SharedPtr detach_client_;
    rclcpp::Logger logger_;
    std::vector<std::string> joint_names_;

public:
    PickAndPlace(rclcpp::Node::SharedPtr node)
        : node_(node),
          move_group_(node, "ur5_manipulator"),
          gripper_(node, "robotiq_gripper"),
          logger_(rclcpp::get_logger("PickAndPlace"))
    {
        // Create service clients
        client_picking_pose_ = node_->create_client<frame_transform::srv::FrameTransform>(
            "/get_position_base_link");
        attach_client_ = node_->create_client<linkattacher_msgs::srv::AttachLink>("/ATTACHLINK");
        detach_client_ = node_->create_client<linkattacher_msgs::srv::DetachLink>("/DETACHLINK");
        
        
        move_group_.setPoseReferenceFrame("base_link");
        
        // Configure OMPL planner
        move_group_.setPlanningTime(60.0);
        move_group_.setMaxVelocityScalingFactor(0.5);
        move_group_.setMaxAccelerationScalingFactor(0.5);
        move_group_.setPlannerId("RRTConnectkConfigDefault");
        move_group_.setNumPlanningAttempts(10);
        move_group_.allowReplanning(true);
        move_group_.setGoalTolerance(0.05);
        
        RCLCPP_INFO(logger_, "PickAndPlace initialized with OMPL configuration");
    }

    void close_gripper()
    {
        gripper_.setMaxVelocityScalingFactor(0.5);
        gripper_.setJointValueTarget("robotiq_85_left_knuckle_joint", 0.24);  // Close fully using action command value
        gripper_.move();
    }

    void open_gripper()
    {
        gripper_.setJointValueTarget("robotiq_85_left_knuckle_joint", 0.0);  // Open fully using action command value
        gripper_.move();
    }

    bool attachObject()
    {
        auto request = std::make_shared<linkattacher_msgs::srv::AttachLink::Request>();
        request->model1_name = "cobot";
        request->link1_name = "wrist_3_link";
        request->model2_name = "coke_can";
        request->link2_name = "link";

        if (!attach_client_->wait_for_service(std::chrono::seconds(5))) {
            RCLCPP_WARN(logger_, "AttachLink service not available");
            return false;
        }

        auto future = attach_client_->async_send_request(request);
        using namespace std::chrono_literals;
        if (future.wait_for(5s) != std::future_status::ready) {
            RCLCPP_WARN(logger_, "Failed to attach object (timeout)");
            return false;
        }

        future.get();
        RCLCPP_INFO(logger_, "Object attached successfully");
        return true;
    }

    bool detachObject()
    {
        auto request = std::make_shared<linkattacher_msgs::srv::DetachLink::Request>();
        request->model1_name = "cobot";
        request->link1_name = "wrist_3_link";
        request->model2_name = "coke_can";
        request->link2_name = "link";

        if (!detach_client_->wait_for_service(std::chrono::seconds(5))) {
            RCLCPP_WARN(logger_, "DetachLink service not available");
            return false;
        }

        auto future = detach_client_->async_send_request(request);
        using namespace std::chrono_literals;
        if (future.wait_for(5s) != std::future_status::ready) {
            RCLCPP_WARN(logger_, "Failed to detach object (timeout)");
            return false;
        }

        future.get();
        RCLCPP_INFO(logger_, "Object detached successfully");
        return true;
    }

    bool pick()
    {
        // ----------------------------
        // 0) Get pick position from service
        // ----------------------------
        if (!client_picking_pose_->wait_for_service(std::chrono::seconds(5))) {
            RCLCPP_ERROR(logger_, "Service /get_position_base_link not available");
            return false;
        }

        auto request = std::make_shared<frame_transform::srv::FrameTransform::Request>();
        auto future = client_picking_pose_->async_send_request(request);
        
        using namespace std::chrono_literals;
        if (future.wait_for(5s) != std::future_status::ready) {
            RCLCPP_ERROR(logger_, "Service call timeout");
            return false;
        }

        auto response = future.get();

        RCLCPP_INFO(logger_, "Service response (base_link): x=%.3f, y=%.3f, z=%.3f",
                    response->x_base_link_frame,
                    response->y_base_link_frame,
                    response->z_base_link_frame);

        // ----------------------------
        // 1) Move to safe home after target capture
        // ----------------------------
        RCLCPP_INFO(logger_, "Moving to safe home position first...");
        std::vector<double> home_joint_values = {0.0, -1.57, 1.57, -1.57, -1.57, 0.0};
        move_group_.setJointValueTarget(home_joint_values);
        
        moveit::planning_interface::MoveGroupInterface::Plan home_plan;
        if (move_group_.plan(home_plan) == moveit::core::MoveItErrorCode::SUCCESS) {
            move_group_.execute(home_plan);
            RCLCPP_INFO(logger_, "Moved to home position");
            rclcpp::sleep_for(std::chrono::seconds(1));
        } else {
            RCLCPP_WARN(logger_, "Failed to reach home, continuing anyway");
        }

        // ----------------------------
        // 2) Define pick and pre-pick
        // ----------------------------
        geometry_msgs::msg::Pose pick_position;
        tf2::Quaternion orientation;
        orientation.setRPY(-3.069, 0.009, -0.939);
        pick_position.orientation = tf2::toMsg(orientation);
        pick_position.position.x = response->x_base_link_frame;
        pick_position.position.y = response->y_base_link_frame;

        const double table_top_world_z = kTableCenterZInWorld + (kTableThickness / 2.0);
        const double table_top_base_link_z = table_top_world_z - kBaseLinkInWorldZ;
        const double min_pick_z = table_top_base_link_z + kGraspClearanceAboveTable;
        const double offset_pick_z = response->z_base_link_frame + kGraspZOffset;
        pick_position.position.z = std::max(offset_pick_z, min_pick_z);

        RCLCPP_INFO(logger_, "Pick position: x=%.3f, y=%.3f, z=%.3f",
                    pick_position.position.x, 
                    pick_position.position.y,
                    pick_position.position.z);

        if (offset_pick_z < min_pick_z) {
            RCLCPP_WARN(logger_,
                        "Offset pick Z (%.3f) is below table-top+clearance (%.3f) in base_link; clamping pick z.",
                        offset_pick_z,
                        min_pick_z);
        }

        // Pre-pick is 0.20 m above pick along +Z in the reference frame
        geometry_msgs::msg::Pose pre_pick_position = pick_position;
        pre_pick_position.position.z += 0.20;

        RCLCPP_INFO(logger_, "Pre-pick position: x=%.3f, y=%.3f, z=%.3f",
                    pre_pick_position.position.x, 
                    pre_pick_position.position.y,
                    pre_pick_position.position.z);
        
        auto current_pose = move_group_.getCurrentPose("tool0");
        RCLCPP_INFO(logger_, "Current pose: x=%.3f, y=%.3f, z=%.3f",
                    current_pose.pose.position.x,
                    current_pose.pose.position.y,
                    current_pose.pose.position.z);

        // ----------------------------
        // 3) Plan & execute to pre-pick
        // ----------------------------
        if (!planAndExecutePose(pre_pick_position, "tool0", "pre-pick")) {
            RCLCPP_ERROR(logger_, "Failed to reach pre-pick pose. Aborting pick.");
            return false;
        }

        // ----------------------------
        // 4) Cartesian linear move: pre-pick -> pick (straight down)
        // ----------------------------
        if (!executeCartesianDown(pre_pick_position, pick_position)) {
            RCLCPP_ERROR(logger_, "Cartesian approach to pick failed. Aborting pick.");
            return false;
        }

        // ----------------------------
        // 5) Close gripper
        // ----------------------------
        close_gripper();
        rclcpp::sleep_for(std::chrono::seconds(1));

        if (!attachObject()) {
            RCLCPP_ERROR(logger_, "Failed to attach object after grasp.");
            return false;
        }

        rclcpp::sleep_for(std::chrono::seconds(1));

        RCLCPP_INFO(logger_, "Pick motion completed successfully");
        return true;
    }

    bool place()
    {
        geometry_msgs::msg::Pose place_position;
        
        tf2::Quaternion orientation;
        orientation.setRPY(-3.069, 0.009, -0.939);
        place_position.orientation = tf2::toMsg(orientation);
        place_position.position.x = -0.570;
        place_position.position.y = 0.647;
        place_position.position.z = 0.256;

        RCLCPP_INFO(logger_, "Planning to place position...");
        RCLCPP_INFO(logger_, "Place pose: x=%.3f y=%.3f z=%.3f",
                    place_position.position.x, place_position.position.y, place_position.position.z);

        if (!planAndExecutePose(place_position, "tool0", "place")) {
            RCLCPP_ERROR(logger_, "Failed to execute place motion");
            return false;
        }

        if (!detachObject()) {
            RCLCPP_ERROR(logger_, "Failed to detach object at place pose");
            return false;
        }

        rclcpp::sleep_for(std::chrono::seconds(1));

        RCLCPP_INFO(logger_, "Place motion completed successfully");
        return true;
    }

    bool planAndExecutePose(const geometry_msgs::msg::Pose& target,
                            const std::string& ee_link,
                            const std::string& label)
    {
        move_group_.setPoseTarget(target, ee_link);

        moveit::planning_interface::MoveGroupInterface::Plan plan;
        const bool success = (move_group_.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);

        RCLCPP_INFO(logger_, "Planning to %s pose: %s", label.c_str(), success ? "SUCCESS" : "FAILED");

        if (!success) {
            return false;
        }

        const bool execute_success = (move_group_.execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
        if (!execute_success) {
            RCLCPP_ERROR(logger_, "Execution to %s pose failed.", label.c_str());
            return false;
        }

        RCLCPP_INFO(logger_, "Execution to %s pose completed.", label.c_str());
        return true;
    }

    bool executeCartesianDown(const geometry_msgs::msg::Pose& pre_pick,
                              const geometry_msgs::msg::Pose& pick)
    {
        const double eef_step = 0.01;
        const double jump_threshold = 0.0;

        std::vector<geometry_msgs::msg::Pose> waypoints;
        waypoints.reserve(2);
        waypoints.push_back(pre_pick);
        waypoints.push_back(pick);

        moveit_msgs::msg::RobotTrajectory trajectory;
        const double fraction = move_group_.computeCartesianPath(
            waypoints,
            eef_step,
            jump_threshold,
            trajectory);

        RCLCPP_INFO(logger_, "Cartesian path fraction: %.2f", fraction);

        if (fraction < 0.95) {
            RCLCPP_WARN(logger_, "Cartesian path not feasible enough (fraction < 0.95).");
            return false;
        }

        moveit::planning_interface::MoveGroupInterface::Plan cart_plan;
        cart_plan.trajectory_ = trajectory;

        const bool execute_success = (move_group_.execute(cart_plan) == moveit::core::MoveItErrorCode::SUCCESS);
        if (!execute_success) {
            RCLCPP_ERROR(logger_, "Cartesian approach execution failed.");
            return false;
        }

        RCLCPP_INFO(logger_, "Cartesian approach execution completed.");
        return true;
    }
};

void addCollisionObjects(moveit::planning_interface::PlanningSceneInterface& planning_scene_interface, rclcpp::Logger logger)
{
    std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
    collision_objects.resize(2);  // two objects: camera_mount and table

    // Camera mount 
    collision_objects[0].id = "camera_mount";
    collision_objects[0].header.frame_id = "base_link";
    collision_objects[0].primitives.resize(1);
    collision_objects[0].primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
    collision_objects[0].primitives[0].dimensions = {0.2, 0.4, 0.5};  // small box around camera
    collision_objects[0].primitive_poses.resize(1);
    collision_objects[0].primitive_poses[0].position.x = 0.23;
    collision_objects[0].primitive_poses[0].position.y = 0.43;
    collision_objects[0].primitive_poses[0].position.z = 1.0;
    collision_objects[0].operation = moveit_msgs::msg::CollisionObject::ADD;

    // Table
    collision_objects[1].id = "table";
    collision_objects[1].header.frame_id = "world";
    collision_objects[1].primitives.resize(1);
    collision_objects[1].primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
    collision_objects[1].primitives[0].dimensions = {1.625, 0.914, 0.70};
    collision_objects[1].primitive_poses.resize(1);
    collision_objects[1].primitive_poses[0].position.x = -0.045;
    collision_objects[1].primitive_poses[0].position.y = 0.375;
    collision_objects[1].primitive_poses[0].position.z = 0.3585;
    collision_objects[1].operation = moveit_msgs::msg::CollisionObject::ADD;

    planning_scene_interface.applyCollisionObjects(collision_objects);
    RCLCPP_INFO(logger, "Collision objects added to the planning scene.");
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions node_options;
    node_options.parameter_overrides({rclcpp::Parameter("use_sim_time", true)});
    node_options.arguments({
        "--ros-args",
        "--remap",
        "/joint_states:=/joint_states_moveit",
    });
    auto node = std::make_shared<rclcpp::Node>("ur5e_pick_and_place", node_options);

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    std::thread spinner([&executor]() { executor.spin(); });

    rclcpp::sleep_for(std::chrono::seconds(1));

    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
    
    addCollisionObjects(planning_scene_interface, node->get_logger());
    rclcpp::sleep_for(std::chrono::seconds(1));

    try {
        PickAndPlace pick_and_place(node);
        
        rclcpp::sleep_for(std::chrono::seconds(1));
        
        // Open gripper first
        pick_and_place.open_gripper();
        rclcpp::sleep_for(std::chrono::seconds(1));
        
        // Pick sequence (pre-pick + cartesian down + grasp)
        if (pick_and_place.pick()) {
            rclcpp::sleep_for(std::chrono::seconds(2));
        
            // Place sequence
            if (pick_and_place.place()) {
                rclcpp::sleep_for(std::chrono::seconds(1));
                
                // Open gripper to release
                pick_and_place.open_gripper();
                rclcpp::sleep_for(std::chrono::seconds(1));
                
                RCLCPP_INFO(node->get_logger(), "Pick and place completed successfully!");
            }
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node->get_logger(), "Error in pick and place: %s", e.what());
        return 1;
    }

    rclcpp::shutdown();
    spinner.join();
    return 0;
}