#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <linkattacher_msgs/srv/attach_link.hpp>
#include <linkattacher_msgs/srv/detach_link.hpp>

const double tau = 2 * M_PI;

class PickAndPlace
{
public:
    PickAndPlace(rclcpp::Node::SharedPtr node)
        : move_group(node, "ur5_manipulator"),
          gripper(node, "robotiq_gripper"),
          planning_scene_interface(),
          logger(rclcpp::get_logger("PickAndPlace")),
          node_(node),
          tf_buffer(std::make_shared<tf2_ros::Buffer>(node_->get_clock())),
          tf_listener(std::make_shared<tf2_ros::TransformListener>(*tf_buffer))
    {
        move_group.setPoseReferenceFrame("base_link");
        attach_client = node_->create_client<linkattacher_msgs::srv::AttachLink>("/ATTACHLINK");
        detach_client = node_->create_client<linkattacher_msgs::srv::DetachLink>("/DETACHLINK");

        centroid_sub = node_->create_subscription<geometry_msgs::msg::Point>(
            "/centroid", 10,
            [this](const geometry_msgs::msg::Point::SharedPtr msg) {
                geometry_msgs::msg::PointStamped input_point, transformed_point;
                input_point.header.frame_id = "camera_optical_link";
                input_point.header.stamp = node_->now();
                input_point.point = *msg;

                try {
                    auto tf_stamped = tf_buffer->lookupTransform("base_link", "camera_optical_link", tf2::TimePointZero);
                    tf2::doTransform(input_point, transformed_point, tf_stamped);
                    latest_centroid = transformed_point.point;
                    centroid_received = true;
                    RCLCPP_INFO(logger, "Centroid transformed: x=%.3f y=%.3f z=%.3f", latest_centroid.x, latest_centroid.y, latest_centroid.z);
                } catch (tf2::TransformException &ex) {
                    RCLCPP_WARN(logger, "Transform failed: %s", ex.what());
                }
            });
    }

    bool isCentroidReceived() const { return centroid_received; }

    void close_gripper() 
    {   
        gripper.setJointValueTarget("robotiq_85_left_knuckle_joint", 0.2); 
        gripper.move(); 
    }
    void open_gripper() 
    { 
        gripper.setJointValueTarget("robotiq_85_left_knuckle_joint", 0.0); 
        gripper.move(); 
    }

    void pick()
    {
        if (!centroid_received) {
            RCLCPP_ERROR(logger, "Centroid not received yet, aborting pick.");
            return;
        }

        move_group.setMaxVelocityScalingFactor(0.5);
        move_group.setMaxAccelerationScalingFactor(0.5);
        move_group.setPlanningTime(10.0);
        move_group.allowReplanning(true);
        move_group.setGoalTolerance(0.03);

        geometry_msgs::msg::Pose pick_pose;
        tf2::Quaternion orientation;
        orientation.setRPY(1.57, 0, -3.14);
        pick_pose.orientation = tf2::toMsg(orientation);
        // pick_pose.position.x = latest_centroid.x;
        // pick_pose.position.y = latest_centroid.y;
        // pick_pose.position.z = latest_centroid.z;


        // Offset of 13 cm of positive z of tool0
        tf2::Transform offset_tf;
        offset_tf.setOrigin(tf2::Vector3(0.0, 0.0, -0.13));
        offset_tf.setRotation(orientation);

        tf2::Vector3 adjusted_pos(latest_centroid.x, latest_centroid.y, latest_centroid.z);
        tf2::Vector3 corrected_pos = adjusted_pos + tf2::quatRotate(orientation, offset_tf.getOrigin());

        pick_pose.position.x = corrected_pos.x();
        pick_pose.position.y = corrected_pos.y();
        pick_pose.position.z = corrected_pos.z();

        move_group.setPoseTarget(pick_pose, "tool0");
        moveit::planning_interface::MoveGroupInterface::Plan my_plan;
        bool success = (move_group.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
        RCLCPP_INFO(logger, "Visualizing pick plan: %s", success ? "SUCCESS" : "FAILED");

        if (success) 
        {
            move_group.move();
            RCLCPP_INFO(logger, "Pick motion execution completed.");

            close_gripper();
            rclcpp::sleep_for(std::chrono::seconds(1));

            attachObject();
            rclcpp::sleep_for(std::chrono::seconds(1));
        } 
        else 
        {
            RCLCPP_ERROR(logger, "Motion planning for pick failed!");
        }
    }

    void place()
    {
        move_group.setMaxVelocityScalingFactor(1);
        move_group.setMaxAccelerationScalingFactor(1);
        move_group.setPlanningTime(10.0);  
        move_group.allowReplanning(true);  
        move_group.setGoalTolerance(0.03); 

        // Creazione della posa target per il pick
        geometry_msgs::msg::Pose place_pose;
        tf2::Quaternion orientation;
        orientation.setRPY(-3.14, 0, -3.14);
        place_pose.orientation = tf2::toMsg(orientation);
        place_pose.position.x = 0.088;
        place_pose.position.y = 0.751;
        place_pose.position.z = 0.4;

        move_group.setPoseTarget(place_pose, "tool0");

        // Planning
        moveit::planning_interface::MoveGroupInterface::Plan my_plan;
        bool success = (move_group.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
        RCLCPP_INFO(logger, "Visualizing plan: %s", success ? "SUCCESS" : "FAILED");

        // Execution
        if (success)
        {
            move_group.move();
            RCLCPP_INFO(logger, "Motion execution completed.");
        }
        else
        {
            RCLCPP_ERROR(logger, "Motion planning failed!");
        }
    }

    void attachObject()
    {
        auto request = std::make_shared<linkattacher_msgs::srv::AttachLink::Request>();
        request->model1_name = "cobot";  // Nome del robot in Gazebo
        request->link1_name = "wrist_3_link"; // Nome del link del gripper
        request->model2_name = "red_cube"; // Nome dell'oggetto da afferrare
        request->link2_name = "link_1";    // Nome del link dell'oggetto

        while (!attach_client->wait_for_service(std::chrono::seconds(1)))
        {
            RCLCPP_WARN(logger, "Waiting for the AttachLink service...");
        }

        auto future = attach_client->async_send_request(request);
        if (rclcpp::spin_until_future_complete(node_, future) == rclcpp::FutureReturnCode::SUCCESS)
        {
            RCLCPP_INFO(logger, "Object attached successfully.");
        }
        else
        {
            RCLCPP_ERROR(logger, "Failed to attach object.");
        }
    }

    void detachObject()
    {
        auto request = std::make_shared<linkattacher_msgs::srv::DetachLink::Request>();
        request->model1_name = "cobot";  // Nome del robot in Gazebo
        request->link1_name = "wrist_3_link"; // Nome del link del gripper
        request->model2_name = "red_cube"; // Nome dell'oggetto da afferrare
        request->link2_name = "link_1";    // Nome del link dell'oggetto

        while (!detach_client->wait_for_service(std::chrono::seconds(1)))
        {
            RCLCPP_WARN(logger, "Waiting for the DetachLink service...");
        }

        auto future = detach_client->async_send_request(request);
        if (rclcpp::spin_until_future_complete(node_, future) == rclcpp::FutureReturnCode::SUCCESS)
        {
            RCLCPP_INFO(logger, "Object detached successfully.");
        }
        else
        {
            RCLCPP_ERROR(logger, "Failed to detach object.");
        }
    }




    void addCollisionObjects()
    {
        
        collision_objects.resize(3);


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

        
        // collision_objects[3].id = "object";
        // collision_objects[3].header.frame_id = "base_link";
        // collision_objects[3].primitives.resize(1);
        // collision_objects[3].primitives[0].type = shape_msgs::msg::SolidPrimitive::BOX;
        // collision_objects[3].primitives[0].dimensions = {0.02, 0.02, 0.2};
        // collision_objects[3].primitive_poses.resize(1);
        // collision_objects[3].primitive_poses[0].position.x = latest_centroid.x;
        // collision_objects[3].primitive_poses[0].position.y = latest_centroid.y;
        // collision_objects[3].primitive_poses[0].position.z = latest_centroid.z;
        // collision_objects[3].operation = moveit_msgs::msg::CollisionObject::ADD;

        
        planning_scene_interface.applyCollisionObjects(collision_objects);
        RCLCPP_INFO(logger, "Collision objects added to the planning scene.");
    }

    // void attachCollisionObject()
    // {
    //     //moveit_msgs::AttachedCollisionObject attached_object;
    //     attached_object.link_name = "wrist_3_link";
    //     attached_object.object = collision_objects[3];  

    //     attached_object.object.operation = attached_object.object.ADD;

    //     std::vector<std::string> touch_links;

    //     touch_links.push_back("robotiq_85_right_finger_tip_link");
    //     touch_links.push_back("robotiq_85_left_finger_tip_link");
    //     touch_links.push_back("wrist_3_link");

    //     attached_object.touch_links = touch_links;

    //     planning_scene_interface.applyAttachedCollisionObject(attached_object);
    // }

    // void detachCollisionObject()
    // {
    //     // Specify the link to which the object is currently attached
    //     attached_object.link_name = "wrist_3_link";
    //     // Define the operation as removing the attachment
    //     attached_object.object.operation = attached_object.object.REMOVE;
    //     // Apply the detachment operation to the planning scene
    //     planning_scene_interface.applyAttachedCollisionObject(attached_object);
    // }

private:
    moveit::planning_interface::MoveGroupInterface move_group;
    moveit::planning_interface::MoveGroupInterface gripper;
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
    std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
    moveit_msgs::msg::AttachedCollisionObject attached_object;
    rclcpp::Logger logger;
    rclcpp::Node::SharedPtr node_;
    rclcpp::Client<linkattacher_msgs::srv::AttachLink>::SharedPtr attach_client;
    rclcpp::Client<linkattacher_msgs::srv::DetachLink>::SharedPtr detach_client;

    geometry_msgs::msg::Point latest_centroid;
    bool centroid_received = false;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr centroid_sub;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("pick_and_place_node");
    PickAndPlace pick_and_place(node);

    pick_and_place.addCollisionObjects();
    rclcpp::sleep_for(std::chrono::seconds(1));


    rclcpp::Rate rate(10);
    int wait_sec = 5;
    int count = 0;
    while (rclcpp::ok() && !pick_and_place.isCentroidReceived() && count < wait_sec * 10) {
        RCLCPP_INFO(rclcpp::get_logger("main"), "Waiting for centroid...");
        rclcpp::spin_some(node);
        rate.sleep();
        count++;
    }

    if (pick_and_place.isCentroidReceived()) {
        pick_and_place.pick();
    } else {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "Timeout waiting for centroid. Aborting pick.");
    }

    // pick_and_place.pick();
    rclcpp::sleep_for(std::chrono::seconds(1));
    // pick_and_place.attachCollisionObject();
    // rclcpp::sleep_for(std::chrono::seconds(1));
    pick_and_place.place();
    rclcpp::sleep_for(std::chrono::seconds(1));
    pick_and_place.open_gripper();
    rclcpp::sleep_for(std::chrono::seconds(1));
    // pick_and_place.detachCollisionObject();
    // rclcpp::sleep_for(std::chrono::seconds(1));
    pick_and_place.detachObject();
    rclcpp::sleep_for(std::chrono::seconds(1));

    rclcpp::shutdown();
    return 0;
}