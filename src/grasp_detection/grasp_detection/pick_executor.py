#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
from control_msgs.action import GripperCommand
from shape_msgs.msg import SolidPrimitive
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from enum import Enum
import copy

class PickPlaceState(Enum):
    IDLE = 0
    APPROACHING = 1
    GRASPING = 2
    LIFTING = 3
    MOVING_TO_PLACE = 4
    RELEASING = 5
    RETREATING = 6
    DONE = 7

class PickExecutor(Node):
    def __init__(self):
        super().__init__('pick_executor')
        
        # Parameters
        self.declare_parameter('place_position.x', 0.3)
        self.declare_parameter('place_position.y', 0.3)
        self.declare_parameter('place_position.z', 0.5)
        self.declare_parameter('pre_grasp_offset', 0.15)  # 15cm before grasp
        self.declare_parameter('lift_height', 0.2)  # Lift 20cm
        
        # Callback group for parallel execution
        self.callback_group = ReentrantCallbackGroup()

        # Subscription
        self.subscription = self.create_subscription(
            PoseStamped,
            '/grasp_detection/pose',
            self.pose_callback,
            10,
            callback_group=self.callback_group
        )

        # Action clients
        self.move_group_client = ActionClient(
            self, 
            MoveGroup, 
            'move_action',
            callback_group=self.callback_group
        )
        
        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            '/gripper_position_controller/gripper_cmd',
            callback_group=self.callback_group
        )

        # State management
        self.state = PickPlaceState.IDLE
        self.current_grasp_pose = None
        self.is_executing = False

        self.get_logger().info('Pick and Place Executor ready!')

    def pose_callback(self, msg: PoseStamped):
        if self.is_executing:
            self.get_logger().warn('Already executing pick-place. Ignoring new pose.')
            return
        
        self.get_logger().info(f'Received grasp pose at [{msg.pose.position.x:.3f}, '
                              f'{msg.pose.position.y:.3f}, {msg.pose.position.z:.3f}]')
        
        self.current_grasp_pose = msg
        self.is_executing = True
        self.state = PickPlaceState.APPROACHING
        
        # Start the pick and place sequence
        self.execute_pick_and_place()

    def execute_pick_and_place(self):
        """Main state machine for pick and place"""
        
        # Step 1: Move to pre-grasp pose
        if self.state == PickPlaceState.APPROACHING:
            self.get_logger().info('=== Step 1: Approaching pre-grasp pose ===')
            pre_grasp_pose = self.get_pre_grasp_pose()
            if self.move_to_pose(pre_grasp_pose):
                self.state = PickPlaceState.GRASPING
            else:
                self.get_logger().error('Failed to approach grasp')
                self.reset_execution()
                return
        
        # Step 2: Open gripper and move to grasp pose
        if self.state == PickPlaceState.GRASPING:
            self.get_logger().info('=== Step 2: Opening gripper and moving to grasp ===')
            self.control_gripper(open=True)
            rclpy.spin_once(self, timeout_sec=1.0)
            
            if self.move_to_pose(self.current_grasp_pose):
                self.get_logger().info('=== Step 3: Closing gripper ===')
                self.control_gripper(open=False)
                rclpy.spin_once(self, timeout_sec=2.0)  # Wait for grasp
                self.state = PickPlaceState.LIFTING
            else:
                self.get_logger().error('Failed to reach grasp pose')
                self.reset_execution()
                return
        
        # Step 4: Lift object
        if self.state == PickPlaceState.LIFTING:
            self.get_logger().info('=== Step 4: Lifting object ===')
            lift_pose = self.get_lift_pose()
            if self.move_to_pose(lift_pose):
                self.state = PickPlaceState.MOVING_TO_PLACE
            else:
                self.get_logger().error('Failed to lift object')
                self.reset_execution()
                return
        
        # Step 5: Move to place location
        if self.state == PickPlaceState.MOVING_TO_PLACE:
            self.get_logger().info('=== Step 5: Moving to place location ===')
            place_pose = self.get_place_pose()
            if self.move_to_pose(place_pose):
                self.state = PickPlaceState.RELEASING
            else:
                self.get_logger().error('Failed to reach place location')
                self.reset_execution()
                return
        
        # Step 6: Release object
        if self.state == PickPlaceState.RELEASING:
            self.get_logger().info('=== Step 6: Releasing object ===')
            self.control_gripper(open=True)
            rclpy.spin_once(self, timeout_sec=1.0)
            self.state = PickPlaceState.RETREATING
        
        # Step 7: Retreat
        if self.state == PickPlaceState.RETREATING:
            self.get_logger().info('=== Step 7: Retreating ===')
            retreat_pose = self.get_retreat_pose()
            if self.move_to_pose(retreat_pose):
                self.state = PickPlaceState.DONE
                self.get_logger().info('=== Pick and Place COMPLETED ===')
            else:
                self.get_logger().warn('Failed to retreat, but task completed')
            
            self.reset_execution()

    def move_to_pose(self, target_pose: PoseStamped) -> bool:
        """Send pose goal to MoveIt and wait for result"""
        
        if not self.move_group_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("MoveGroup action server not available")
            return False

        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = 'manipulator'
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = 0.3
        goal_msg.request.max_acceleration_scaling_factor = 0.2

        # Create constraints
        constraints = Constraints()

        # Position constraint
        pos_constraint = PositionConstraint()
        pos_constraint.header = target_pose.header
        pos_constraint.link_name = "tool0"
        
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.02, 0.02, 0.02]  # 2cm tolerance
        
        pos_constraint.constraint_region.primitives.append(box)
        pos_constraint.constraint_region.primitive_poses.append(target_pose.pose)
        pos_constraint.weight = 1.0
        constraints.position_constraints.append(pos_constraint)

        # Orientation constraint
        ori_constraint = OrientationConstraint()
        ori_constraint.header = target_pose.header
        ori_constraint.link_name = "tool0"
        ori_constraint.orientation = target_pose.pose.orientation
        ori_constraint.absolute_x_axis_tolerance = 0.2
        ori_constraint.absolute_y_axis_tolerance = 0.2
        ori_constraint.absolute_z_axis_tolerance = 0.2
        ori_constraint.weight = 1.0
        constraints.orientation_constraints.append(ori_constraint)

        goal_msg.request.goal_constraints.append(constraints)

        # Send goal and wait
        self.get_logger().info(f'Moving to: [{target_pose.pose.position.x:.3f}, '
                              f'{target_pose.pose.position.y:.3f}, '
                              f'{target_pose.pose.position.z:.3f}]')
        
        future = self.move_group_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        
        if not future.result():
            self.get_logger().error('Goal rejected')
            return False
        
        goal_handle = future.result()
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)
        
        if result_future.result():
            self.get_logger().info('Movement succeeded!')
            return True
        else:
            self.get_logger().error('Movement failed or timed out')
            return False

    def control_gripper(self, open: bool):
        """Control gripper - open or close"""
        
        if not self.gripper_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("Gripper action server not available")
            return
        
        goal = GripperCommand.Goal()
        goal.command.position = 0.085 if open else 0.0  # Robotiq 85 range
        goal.command.max_effort = 100.0
        
        action = "Opening" if open else "Closing"
        self.get_logger().info(f'{action} gripper...')
        
        self.gripper_client.send_goal_async(goal)

    def get_pre_grasp_pose(self) -> PoseStamped:
        """Create pre-grasp pose (offset before actual grasp)"""
        pre_grasp = copy.deepcopy(self.current_grasp_pose)
        offset = self.get_parameter('pre_grasp_offset').value
        
        # Move back along approach direction (assume Z-axis approach)
        pre_grasp.pose.position.z += offset
        
        return pre_grasp

    def get_lift_pose(self) -> PoseStamped:
        """Create lift pose (grasp pose + vertical offset)"""
        lift = copy.deepcopy(self.current_grasp_pose)
        lift_height = self.get_parameter('lift_height').value
        lift.pose.position.z += lift_height
        
        return lift

    def get_place_pose(self) -> PoseStamped:
        """Create place pose from parameters"""
        place = PoseStamped()
        place.header.frame_id = 'base_link'
        place.header.stamp = self.get_clock().now().to_msg()
        
        place.pose.position.x = self.get_parameter('place_position.x').value
        place.pose.position.y = self.get_parameter('place_position.y').value
        place.pose.position.z = self.get_parameter('place_position.z').value
        
        # Copy orientation from grasp pose
        place.pose.orientation = self.current_grasp_pose.pose.orientation
        
        return place

    def get_retreat_pose(self) -> PoseStamped:
        """Create retreat pose (place pose + vertical offset)"""
        retreat = self.get_place_pose()
        retreat.pose.position.z += 0.15  # Retreat 15cm up
        
        return retreat

    def reset_execution(self):
        """Reset execution state"""
        self.state = PickPlaceState.IDLE
        self.is_executing = False
        self.current_grasp_pose = None
        self.get_logger().info('Ready for next grasp pose')

def main(args=None):
    rclpy.init(args=args)
    node = PickExecutor()
    
    # Use MultiThreadedExecutor for parallel callbacks
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
