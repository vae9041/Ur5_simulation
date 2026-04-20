#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from rclpy.duration import Duration
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import tf2_ros
import tf2_geometry_msgs.tf2_geometry_msgs
import cv2
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import Marker
from linkattacher_msgs.srv import AttachLink, DetachLink


# NumPy compatibility fix for models saved with NumPy 2.x
import sys
if not hasattr(np, '_core'):
    # Create the _core module structure
    class _CoreModule:
        multiarray = np.core.multiarray
        umath = np.core.umath
        numerictypes = np.core.numerictypes
    
    np._core = _CoreModule()
    sys.modules['numpy._core'] = np._core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    sys.modules['numpy._core.umath'] = np.core.umath
    sys.modules['numpy._core.numerictypes'] = np.core.numerictypes

from ament_index_python.packages import get_package_share_directory

class FasterRCNNNode(Node):
    def __init__(self):
        super().__init__('faster_rcnn_node')

        # Subscriptions
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.rgb_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )

        # Publishers
        self.image_pub = self.create_publisher(Image, '/grasp_detection/image_annotated', 10)
        
        # Publisher for object position in camera optical frame
        self.camera_position_pub = self.create_publisher(
            PointStamped, 
            '/grasp_detection/object_position_camera_frame', 
            10
        )

        self.bridge = CvBridge()
        self.latest_depth_image = None
        self.latest_depth_frame = None
        self.latest_depth_stamp = None
        self.latest_depth_encoding = None

        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        model_path = os.path.join(
    	get_package_share_directory('grasp_detection'),
    	'Model',
    	'best_model.pth'
	)

        # Set device (GPU if available, else CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        self.model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        
        # Load checkpoint and extract model state dict
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Check if it's a checkpoint dict or just state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            self.get_logger().info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)  # Move model to GPU
        self.model.eval()

        self.transform = T.ToTensor()
        # Camera intrinsics from Gazebo camera plugin
        self.declare_parameter('camera.fx', 528.433756558705)  # From Gazebo camera plugin
        self.declare_parameter('camera.fy', 528.433756558705)
        self.declare_parameter('camera.cx', 320.5)  
        self.declare_parameter('camera.cy', 240.5)  
        self.fx = float(self.get_parameter('camera.fx').value)
        self.fy = float(self.get_parameter('camera.fy').value)
        self.cx = float(self.get_parameter('camera.cx').value)
        self.cy = float(self.get_parameter('camera.cy').value)

        self.get_logger().info(f'Camera intrinsics: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}')
        self.get_logger().info('Faster R-CNN model loaded and ready!')

    def depth_callback(self, msg):
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_depth_encoding = msg.encoding
            # remember the frame id of depth images so we can transform from it later
            try:
                self.latest_depth_frame = msg.header.frame_id
            except Exception:
                self.latest_depth_frame = None
            try:
                self.latest_depth_stamp = msg.header.stamp
            except Exception:
                self.latest_depth_stamp = None
            # log arrival for debugging
            try:
                shape = self.latest_depth_image.shape
            except Exception:
                shape = None
            self.get_logger().info(
                f"Depth image received (frame='{self.latest_depth_frame}', encoding='{self.latest_depth_encoding}', shape={shape})"
            )
        except Exception as e:
            self.get_logger().error(f"Depth image conversion failed: {e}")

    def rgb_callback(self, msg):
        if self.latest_depth_image is None:
            self.get_logger().warn("Waiting for depth image...")
            return

        try:
            if self.latest_depth_stamp is not None:
                rgb_t = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
                depth_t = float(self.latest_depth_stamp.sec) + float(self.latest_depth_stamp.nanosec) * 1e-9
                if abs(rgb_t - depth_t) > 0.2:
                    self.get_logger().warn(
                        f"Depth/RGB timestamps too far apart (rgb={rgb_t:.3f}, depth={depth_t:.3f}); skipping frame"
                    )
                    return
        except Exception:
            pass

        try:
            # Convert RGB image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            input_tensor = self.transform(cv_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                predictions = self.model(input_tensor)[0]

            # Find best box
            best_score = 0.0
            best_box = None

            for box, score in zip(predictions['boxes'], predictions['scores']):
                if score > 0.7 and score > best_score:
                    best_score = score
                    best_box = box.int().tolist()

            if best_box:
                x1, y1, x2, y2 = best_box
                self.get_logger().info(
                    f"Selected bbox score={best_score:.3f} box=({x1}, {y1}) -> ({x2}, {y2})"
                )
                bbox_cx, bbox_cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Validate and extract depth
                h, w = self.latest_depth_image.shape[:2]
                if bbox_cx < 0 or bbox_cx >= w or bbox_cy < 0 or bbox_cy >= h:
                    self.get_logger().warn(f"Projection pixel ({bbox_cx},{bbox_cy}) outside depth image bounds {w}x{h}")
                    return

                depth_cx = bbox_cx
                depth_cy = bbox_cy
                depth_raw = float(self.latest_depth_image[depth_cy, depth_cx])

                if not np.isfinite(depth_raw) or depth_raw <= 0.0:
                    self.get_logger().warn(
                        f"Invalid depth at projection pixel ({depth_cx},{depth_cy}) = {depth_raw}"
                    )
                    return

                depth_val = depth_raw
                if self.latest_depth_encoding == '16UC1':
                    depth_val = depth_val / 1000.0
                elif depth_val > 50.0:
                    depth_val = depth_val / 1000.0

                self.get_logger().info(
                    f"Using center depth at ({depth_cx},{depth_cy}) = {depth_raw}"
                )
                self.get_logger().info(f"Using depth (meters): {depth_val:.6f}")

                # Annotate image
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(cv_image, (bbox_cx, bbox_cy), 5, (0, 0, 255), -1)
                cv2.circle(cv_image, (depth_cx, depth_cy), 4, (255, 0, 0), -1)
                cv2.putText(cv_image, f"Depth: {depth_val:.2f}m", (bbox_cx, bbox_cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Camera optical frame coordinates.
                # Gazebo depth cameras often publish depth as range along the viewing ray.
                # Convert that to optical-axis depth (Z) before computing X/Y.
                Z_cam = depth_val
                X_cam = ((bbox_cx - self.cx) * Z_cam) / self.fx
                Y_cam = ((bbox_cy - self.cy) * Z_cam) / self.fy
                
                self.get_logger().info(
                    f"Projection pixel: ({bbox_cx}, {bbox_cy}), depth sample: ({depth_cx}, {depth_cy}), Depth: {Z_cam:.3f}m"
                )
                self.get_logger().info(f"Camera optical frame: X={X_cam:.3f} Y={Y_cam:.3f} Z={Z_cam:.3f}")
                
                # Publish position in camera optical frame
                camera_point = PointStamped()
                camera_point.header.stamp = self.get_clock().now().to_msg()
                camera_point.header.frame_id = "camera_optical_link"
                camera_point.point.x = float(X_cam)
                camera_point.point.y = float(Y_cam)
                camera_point.point.z = float(Z_cam)
                self.camera_position_pub.publish(camera_point)
                
                self.get_logger().info("Published object position in camera optical frame")

                # Publish annotated image
                annotated_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
                self.image_pub.publish(annotated_msg)

        except Exception as e:
            self.get_logger().error(f"Image processing failed: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

def main(args=None):
    rclpy.init(args=args)
    node = FasterRCNNNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

