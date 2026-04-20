#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class GraspDetector(Node):
    def __init__(self):
        super().__init__('grasp_detector')

        # Subscribes to the wrist RGB camera
        self.sub = self.create_subscription(
            Image,
            '/wrist_rgbd_depth_sensor/image_raw',
            self.image_callback,
            10
        )

        # Publisher to send annotated image (view in rqt_image_view)
        self.image_pub = self.create_publisher(Image, '/grasp_detection/image_annotated', 10)

        self.bridge = CvBridge()
        self.get_logger().info("✅ GraspDetector Node Running!")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Just draw dummy box for testing
            h, w, _ = frame.shape
            cv2.rectangle(frame, (int(w*0.3), int(h*0.3)), (int(w*0.7), int(h*0.7)), (0, 255, 0), 2)

            # Convert back and publish
            annotated_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.image_pub.publish(annotated_msg)

        except Exception as e:
            self.get_logger().error(f"Processing failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = GraspDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

