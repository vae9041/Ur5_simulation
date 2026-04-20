#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <frame_transform/srv/frame_transform.hpp>
#include <linkattacher_msgs/srv/attach_link.hpp>
#include <linkattacher_msgs/srv/detach_link.hpp>

const double tau = 2 * M_PI; // tau = 2π handles the rotation more intuitively

class PickAndPlace
{
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
        move_group_.setPlanningTime(60.0);
        move_group_.setMaxVelocityScalingFactor(1.0);
        move_group_.setMaxAccelerationScalingFactor(1.0);
        
        RCLCPP_INFO(logger_, "PickAndPlace initialized");
    }

    void close_gripper()
    {
        gripper_.setMaxVelocityScalingFactor(0.5);
        gripper_.setJointValueTarget("robotiq_85_left_knuckle_joint", 0.5);
        gripper_.move();
    }

    void open_gripper()
    {
        gripper_.setJointValueTarget("robotiq_85_left_knuckle_joint", 0.0);
        gripper_.move();
    }
    void attachObject()
    {
        auto request = std::make_shared<linkattacher_msgs::srv::AttachLink::Request>();
        request->model1_name = "cobot";
        request->link1_name = "wrist_3_link";
        request->model2_name = "red_cube";
        request->link2_name = "link_1";

        if (!attach_client_->wait_for_service(std::chrono::seconds(5)))
        {
            RCLCPP_WARN(logger_, "AttachLink service not available");
            return;
        }

        auto future = attach_client_->async_send_request(request);
        using namespace std::chrono_literals;
        if (future.wait_for(5s) == std::future_status::ready)
        {
            RCLCPP_INFO(logger_, "Object attached successfully");
        }
        else
        {
            RCLCPP_WARN(logger_, "Failed to attach object (timeout)");
        }
    }

    void detachObject()
    {
        auto request = std::make_shared<linkattacher_msgs::srv::DetachLink::Request>();
        request->model1_name = "cobot";
        request->link1_name = "wrist_3_link";
        request->model2_name = "red_cube";
        request->link2_name = "link_1";

        if (!detach_client_->wait_for_service(std::chrono::seconds(2)))
        {
            RCLCPP_WARN(logger_, "DetachLink service not available");
            return;
        }

        auto future = detach_client_->async_send_request(request);
        using namespace std::chrono_literals;
        if (future.wait_for(2s) == std::future_status::ready)
        {
            RCLCPP_INFO(logger_, "Object detached successfully");
        }
        else
        {
            RCLCPP_WARN(logger_, "Failed to detach object (timeout)");
        }
    }

    bool pick()
    {
        // Wait for service
        if (!client_picking_pose_->wait_for_service(std::chrono::seconds(5))) {
            RCLCPP_ERROR(logger_, "Service /get_position_base_link not available");
            return false;
        }

        // Create service request
        auto request = std::make_shared<frame_transform::srv::FrameTransform::Request>();
        
        // Call service
        auto future = client_picking_pose_->async_send_request(request);
        
        // Wait for response using wait_for instead of spin_until_future_complete
        using namespace std::chrono_literals;
        if (future.wait_for(5s) != std::future_status::ready)
        {
            RCLCPP_ERROR(logger_, "Service call timeout");
            return false;
        }

        auto response = future.get();
        
        // This is to set the pick position
        geometry_msgs::msg::Pose pick_position;
        tf2::Quaternion orientation;
        orientation.setRPY(-tau/2, 0, -tau/4);
        pick_position.orientation = tf2::toMsg(orientation);
        pick_position.position.x = response->x_base_link_frame;
        pick_position.position.y = response->y_base_link_frame;
        pick_position.position.z = response->z_base_link_frame;

        RCLCPP_INFO(logger_, "Pick position: x=%.3f, y=%.3f, z=%.3f",
                    pick_position.position.x, 
                    pick_position.position.y,
                    pick_position.position.z);

        move_group_.setPoseTarget(pick_position, "tool0");

        bool success = (move_group_.move() == moveit::core::MoveItErrorCode::SUCCESS);
        if (!success) {
            RCLCPP_ERROR(logger_, "Failed to execute pick motion");
            return false;
        }

        RCLCPP_INFO(logger_, "Pick motion completed successfully");
        return true;
    }

    bool place()
    {
        geometry_msgs::msg::Pose place_position;
        
        tf2::Quaternion orientation;
        orientation.setRPY(-tau/4, 0, 0);
        place_position.orientation = tf2::toMsg(orientation);
        place_position.position.x = 0.0;
        place_position.position.y = 0.8;
        place_position.position.z = 0.5;
        
        move_group_.setPoseTarget(place_position, "tool0");

        bool success = (move_group_.move() == moveit::core::MoveItErrorCode::SUCCESS);
        if (!success) {
            RCLCPP_ERROR(logger_, "Failed to execute place motion");
            return false;
        }

        RCLCPP_INFO(logger_, "Place motion completed successfully");
        return true;
    }

private:
    rclcpp::Node::SharedPtr node_;
    moveit::planning_interface::MoveGroupInterface move_group_;
    moveit::planning_interface::MoveGroupInterface gripper_;
    rclcpp::Logger logger_;
    rclcpp::Client<frame_transform::srv::FrameTransform>::SharedPtr client_picking_pose_;
    rclcpp::Client<linkattacher_msgs::srv::AttachLink>::SharedPtr attach_client_;
    rclcpp::Client<linkattacher_msgs::srv::DetachLink>::SharedPtr detach_client_;
};

void addCollisionObjects(moveit::planning_interface::PlanningSceneInterface& planning_scene_interface, rclcpp::Logger logger)
{
    std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
    collision_objects.resize(3);  // Changed from 4 to 3 since object is commented out

    collision_objects[0].id = "table1";
    collision_objects[0].header.frame_id = "world";
    collision_objects[0].primitives.resize(1);
    collision_objects[0].primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
    collision_objects[0].primitives[0].dimensions = {0.608, 2.0, 1.0};
    collision_objects[0].primitive_poses.resize(1);
    collision_objects[0].primitive_poses[0].position.x = 0.676;
    collision_objects[0].primitive_poses[0].position.y = 0.0;
    collision_objects[0].primitive_poses[0].position.z = 0.5;
    collision_objects[0].operation = moveit_msgs::msg::CollisionObject::ADD;

    collision_objects[1].id = "table2";
    collision_objects[1].header.frame_id = "world";
    collision_objects[1].primitives.resize(1);
    collision_objects[1].primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
    collision_objects[1].primitives[0].dimensions = {1.3, 0.8, 1.0};
    collision_objects[1].primitive_poses.resize(1);
    collision_objects[1].primitive_poses[0].position.x = 0.0;
    collision_objects[1].primitive_poses[0].position.y = 0.69;
    collision_objects[1].primitive_poses[0].position.z = 0.5;
    collision_objects[1].operation = moveit_msgs::msg::CollisionObject::ADD;

    collision_objects[2].id = "basement";
    collision_objects[2].header.frame_id = "world";
    collision_objects[2].primitives.resize(1);
    collision_objects[2].primitives[0].type = shape_msgs::msg::SolidPrimitive::CYLINDER;
    collision_objects[2].primitives[0].dimensions = {0.8, 0.2}; 
    collision_objects[2].primitive_poses.resize(1);
    collision_objects[2].primitive_poses[0].position.x = 0.0;
    collision_objects[2].primitive_poses[0].position.y = 0.0;
    collision_objects[2].primitive_poses[0].position.z = 0.4;
    collision_objects[2].operation = moveit_msgs::msg::CollisionObject::ADD;


    planning_scene_interface.applyCollisionObjects(collision_objects);
    RCLCPP_INFO(logger, "Collision objects added to the planning scene.");
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("ur5e_pick_and_place");

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    std::thread spinner([&executor]() { executor.spin(); });

    rclcpp::sleep_for(std::chrono::seconds(1));

    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
    
    addCollisionObjects(planning_scene_interface, node->get_logger());
    rclcpp::sleep_for(std::chrono::seconds(1));

    PickAndPlace pick_and_place(node);
    
    rclcpp::sleep_for(std::chrono::seconds(1));
    
    // Open gripper first
    pick_and_place.open_gripper();
    rclcpp::sleep_for(std::chrono::seconds(1));
    
    if (pick_and_place.pick()) {
        rclcpp::sleep_for(std::chrono::seconds(2));
        
        // Close gripper to grasp object
        pick_and_place.close_gripper();
        rclcpp::sleep_for(std::chrono::seconds(2));
        
        // Attach object in Gazebo
        pick_and_place.attachObject();
        rclcpp::sleep_for(std::chrono::seconds(1));
        
        if (pick_and_place.place()) {
            rclcpp::sleep_for(std::chrono::seconds(1));
            
            // Detach object in Gazebo
            pick_and_place.detachObject();
            rclcpp::sleep_for(std::chrono::seconds(1));
            
            // Open gripper to release
            pick_and_place.open_gripper();
            rclcpp::sleep_for(std::chrono::seconds(1));
            
            RCLCPP_INFO(node->get_logger(), "Pick and place completed successfully!");
        }
    }

    rclcpp::shutdown();
    spinner.join();
    return 0;
}
