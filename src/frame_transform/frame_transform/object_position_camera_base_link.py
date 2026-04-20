#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from geometry_msgs.msg import PointStamped
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_point
from frame_transform.srv import FrameTransform

class ConversionFrameServer(Node):
    def __init__(self):
        super().__init__('frame_conversion_server')
        self.declare_parameter('workspace.min_x', 0.0)
        self.declare_parameter('workspace.max_x', 0.8)
        self.declare_parameter('workspace.min_y', -0.6)
        self.declare_parameter('workspace.max_y', 0.6)
        self.declare_parameter('workspace.min_z', -0.2)
        self.declare_parameter('workspace.max_z', 0.15)
        self.min_x = float(self.get_parameter('workspace.min_x').value)
        self.max_x = float(self.get_parameter('workspace.max_x').value)
        self.min_y = float(self.get_parameter('workspace.min_y').value)
        self.max_y = float(self.get_parameter('workspace.max_y').value)
        self.min_z = float(self.get_parameter('workspace.min_z').value)
        self.max_z = float(self.get_parameter('workspace.max_z').value)
        
        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Subscriber
        self.frame_sub = self.create_subscription(
            PointStamped,
            '/grasp_detection/object_position_camera_frame',
            self.conversion_callback,
            10
        )
        
        # Service server
        self.server = self.create_service(
            FrameTransform,
            '/get_position_base_link',
            self.handle_conversion
        )
        
        self.point_base_link = None
        self.point_base_link_stamp = None
        self.source_point_camera = None
        self.source_point_frame_id = None
        self.accepted_point_sequence = 0
        
        self.get_logger().info('Frame conversion server ready')

    def conversion_callback(self, data):
        try:
            # Wait for transform to be available
            transform = self.tf_buffer.lookup_transform(
                'base_link',
                data.header.frame_id,
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0)
            )
            
            # Log transform details
            self.get_logger().info(
                f'Transform translation: x={transform.transform.translation.x:.3f}, '
                f'y={transform.transform.translation.y:.3f}, '
                f'z={transform.transform.translation.z:.3f}'
            )
            self.get_logger().info(
                f'Transform rotation: x={transform.transform.rotation.x:.3f}, '
                f'y={transform.transform.rotation.y:.3f}, '
                f'z={transform.transform.rotation.z:.3f}, w={transform.transform.rotation.w:.3f}'
            )
            
            # Log original point
            self.get_logger().info(
                f'Original point in {data.header.frame_id}: x={data.point.x:.3f}, '
                f'y={data.point.y:.3f}, z={data.point.z:.3f}'
            )
            
            # Transform point from camera_optical_link to base_link
            transformed_point = do_transform_point(data, transform)
            
            self.get_logger().info(
                f'Transformed point: x={transformed_point.point.x:.3f}, '
                f'y={transformed_point.point.y:.3f}, '
                f'z={transformed_point.point.z:.3f}'
            )

            point_is_plausible = (
                self.min_x <= transformed_point.point.x <= self.max_x and
                self.min_y <= transformed_point.point.y <= self.max_y and
                self.min_z <= transformed_point.point.z <= self.max_z
            )

            if not point_is_plausible:
                self.get_logger().warn(
                    f'Transformed point outside workspace bounds; accepting anyway: '
                    f'x={transformed_point.point.x:.3f}, '
                    f'y={transformed_point.point.y:.3f}, '
                    f'z={transformed_point.point.z:.3f}'
                )

            self.point_base_link = transformed_point
            self.point_base_link_stamp = self.get_clock().now()
            self.source_point_camera = PointStamped()
            self.source_point_camera.header = data.header
            self.source_point_camera.point = data.point
            self.source_point_frame_id = data.header.frame_id
            self.accepted_point_sequence += 1
            self.get_logger().info('Accepted transformed point as current service target')
            
        except Exception as e:
            self.get_logger().error(f'Transform failed: {e}')

    def handle_conversion(self, request, response):
        if self.point_base_link is not None:
            response.x_base_link_frame = self.point_base_link.point.x
            response.y_base_link_frame = self.point_base_link.point.y
            response.z_base_link_frame = self.point_base_link.point.z
            point_age_seconds = 0.0
            if self.point_base_link_stamp is not None:
                point_age_seconds = (self.get_clock().now() - self.point_base_link_stamp).nanoseconds / 1e9
            
            self.get_logger().info(
                f'Object position in base_link frame: '
                f'x={response.x_base_link_frame:.3f}, '
                f'y={response.y_base_link_frame:.3f}, '
                f'z={response.z_base_link_frame:.3f}, '
                f'age={point_age_seconds:.3f}s'
            )
            if self.source_point_camera is not None:
                self.get_logger().info(
                    f'Service source #{self.accepted_point_sequence} from {self.source_point_frame_id}: '
                    f'x={self.source_point_camera.point.x:.3f}, '
                    f'y={self.source_point_camera.point.y:.3f}, '
                    f'z={self.source_point_camera.point.z:.3f}'
                )
        else:
            self.get_logger().warn('No point transformed yet')
            response.x_base_link_frame = 0.0
            response.y_base_link_frame = 0.0
            response.z_base_link_frame = 0.0
        
        return response


def main(args=None):
    rclpy.init(args=args)
    conversion_server = ConversionFrameServer()
    
    try:
        rclpy.spin(conversion_server)
    except KeyboardInterrupt:
        print("Shutting down")
    finally:
        conversion_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()