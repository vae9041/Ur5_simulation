#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointStatesFilter(Node):
    def __init__(self):
        super().__init__("joint_states_filter")

        self.declare_parameter("input_topic", "/joint_states")
        self.declare_parameter("output_topic", "/joint_states_moveit")
        self.declare_parameter("drop_suffix", "_mimic")

        input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self._drop_suffix = self.get_parameter("drop_suffix").get_parameter_value().string_value

        self._pub = self.create_publisher(JointState, output_topic, 10)
        self._sub = self.create_subscription(JointState, input_topic, self._cb, 10)

    def _cb(self, msg: JointState) -> None:
        if not msg.name:
            self._pub.publish(msg)
            return

        keep_indices = [
            i for i, name in enumerate(msg.name)
            if not (self._drop_suffix and name.endswith(self._drop_suffix))
        ]

        if len(keep_indices) == len(msg.name):
            self._pub.publish(msg)
            return

        out = JointState()
        out.header = msg.header
        out.name = [msg.name[i] for i in keep_indices]

        if msg.position:
            out.position = [msg.position[i] for i in keep_indices]
        if msg.velocity:
            out.velocity = [msg.velocity[i] for i in keep_indices]
        if msg.effort:
            out.effort = [msg.effort[i] for i in keep_indices]

        self._pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JointStatesFilter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
