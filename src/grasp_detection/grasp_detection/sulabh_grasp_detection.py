#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
import torch
import sys
sys.path.append('/home/eeee784/robotic-grasping')
import torch.nn.functional as F
import cv2
from inference.models.grconvnet import GenerativeResnet
from utils.dataset_processing.grasp import detect_grasps

class SulabhGraspNode(Node):
    def __init__(self):
        super().__init__('sulabh_grasp_node')
        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.annotated_pub = self.create_publisher(Image, '/grasp_detection/image_annotated', 10)


        # Camera intrinsics — update these!!
        self.fx = 528.43
        self.fy = 528.43
        self.cx = 320.0
        self.cy = 240.0

        # Subscribe to RGB and Depth topics
        self.create_subscription(Image, '/camera/image_raw', self.rgb_callback, 10)
        self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)

        # Publisher for grasp pose
        self.pose_pub = self.create_publisher(PoseStamped, '/grasp_detection/pose', 10)

        # Load pretrained GR-ConvNet model directly from checkpoint
        model_path = '/home/eeee784/robotic-grasping/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98'
        self.model = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model.eval()

        self.get_logger().info("Sulabh GR-ConvNet loaded and ready!")

    def rgb_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.try_infer()

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        self.try_infer()

    def try_infer(self):
        if self.rgb_image is None or self.depth_image is None:
            return

        # Resize both to 300x300 (GR-ConvNet default)
        rgb = cv2.resize(self.rgb_image, (300, 300))
        depth = cv2.resize(self.depth_image, (300, 300))

        # Normalize
        rgb_tensor = torch.from_numpy(rgb.transpose((2, 0, 1))).float() / 255.0
        depth_tensor = torch.from_numpy(depth).float().unsqueeze(0) / 1000.0  # Convert mm to meters

        # Stack into RGB-D input [4 x H x W]
        input_tensor = torch.cat((rgb_tensor, depth_tensor), dim=0).unsqueeze(0)

        # Inference
        with torch.no_grad():
            q_img, ang_img, width_img, _ = self.model(input_tensor)

        # Convert to numpy
        q_img = q_img.squeeze().cpu().numpy()
        ang_img = ang_img.squeeze().cpu().numpy()
        width_img = width_img.squeeze().cpu().numpy()


        # Find best grasp
        grasps = detect_grasps(q_img, ang_img, width_img, no_grasps=1)
        if not grasps:
            self.get_logger().warn("No valid grasp found.")
            return

        g = grasps[0]
        u, v = g.center
        theta = g.angle

        # Get depth at grasp center
        z = depth[int(v), int(u)]
        if z <= 0.0 or np.isnan(z):
            self.get_logger().warn("Invalid depth at grasp center")
            return

        # Back-project to 3D (camera frame)
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        # Construct pose message
        pose = PoseStamped()
        pose.header.frame_id = 'camera_link'  # Use tf to transform later
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)

        # Orientation from theta (around z-axis)
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = np.sin(theta / 2.0)
        pose.pose.orientation.w = np.cos(theta / 2.0)

        # Publish
        self.pose_pub.publish(pose)
        self.get_logger().info(f"Published grasp at (x={x:.3f}, y={y:.3f}, z={z:.3f})")

        # Draw grasp on image (rotated rectangle + center/axis)
        img_annotated = rgb.copy()
        try:
            cx_i, cy_i = int(round(u)), int(round(v))
            # Determine rectangle width (pixels) from grasp object or width_img
            rect_w = None
            if hasattr(g, 'width') and g.width is not None:
                try:
                    rect_w = int(round(g.width))
                except Exception:
                    rect_w = None
            if rect_w is None:
                try:
                    rect_w = int(round(width_img[int(round(v)), int(round(u))]))
                except Exception:
                    rect_w = None
            if rect_w is None or rect_w <= 0:
                rect_w = 40
            rect_h = max(8, rect_w // 4)

            # angle in degrees for OpenCV
            angle_deg = float(theta * 180.0 / np.pi)
            box = ((cx_i, cy_i), (rect_w, rect_h), angle_deg)
            pts = cv2.boxPoints(box).astype(int)
            cv2.drawContours(img_annotated, [pts], 0, (0, 255, 0), 2)

            # draw axis line and center
            dx = (rect_w / 2.0) * np.cos(theta)
            dy = (rect_w / 2.0) * np.sin(theta)
            pt1 = (int(round(cx_i - dx)), int(round(cy_i - dy)))
            pt2 = (int(round(cx_i + dx)), int(round(cy_i + dy)))
            cv2.line(img_annotated, pt1, pt2, (0, 255, 255), 2)
            cv2.circle(img_annotated, (cx_i, cy_i), 3, (0, 0, 255), -1)
        except Exception as e:
            self.get_logger().warn(f"Failed to draw grasp annotation: {e}")

        # Convert to ROS Image and publish
        msg_img = self.bridge.cv2_to_imgmsg(img_annotated, encoding='bgr8')
        msg_img.header = pose.header
        self.annotated_pub.publish(msg_img)

        # Clear images after inference
        self.rgb_image = None
        self.depth_image = None

def main(args=None):
    rclpy.init(args=args)
    node = SulabhGraspNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()


