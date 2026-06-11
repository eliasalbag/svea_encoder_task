#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)

from mavros_msgs.msg import WheelOdomStamped, RCIn
from geometry_msgs.msg import TwistWithCovarianceStamped


qos_pubber = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

qos_subber = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)


class MavrosEncoderToTwist(Node):
    def __init__(self):
        super().__init__("mavros_encoder_to_twist")

        # Parameters
        self.declare_parameter("wheel_track", 0.2)
        self.declare_parameter("frame_id", "wheel_encoder")
        self.declare_parameter("distance_topic", "/mavros/wheel_odometry/distance")
        self.declare_parameter("rc_topic", "/mavros/rc/in")
        self.declare_parameter("output_topic", "/encoder/twist")

        # RC mapping
        self.declare_parameter("rc_throttle_channel", 1)
        self.declare_parameter("rc_neutral", 1513)
        self.declare_parameter("rc_deadband", 50)

        # Covariance values
        self.declare_parameter("vx_covariance", 0.05)
        self.declare_parameter("yaw_rate_covariance", 0.1)

        self.wheel_track = float(self.get_parameter("wheel_track").value)
        self.frame_id = self.get_parameter("frame_id").value

        self.distance_topic = self.get_parameter("distance_topic").value
        self.rc_topic = self.get_parameter("rc_topic").value
        self.output_topic = self.get_parameter("output_topic").value

        self.rc_throttle_channel = int(self.get_parameter("rc_throttle_channel").value)
        self.rc_neutral = int(self.get_parameter("rc_neutral").value)
        self.rc_deadband = int(self.get_parameter("rc_deadband").value)

        self.vx_covariance = float(self.get_parameter("vx_covariance").value)
        self.yaw_rate_covariance = float(
            self.get_parameter("yaw_rate_covariance").value
        )

        # State
        self.prev_left = None
        self.prev_right = None
        self.prev_time = None

        # Direction from RC:
        # +1 forward, -1 backward, 0 neutral
        self.rc_direction = 0
        self.last_rc_throttle = None

        # Publisher
        self.encoder_pub = self.create_publisher(
            TwistWithCovarianceStamped,
            self.output_topic,
            qos_pubber,
        )

        # Subscribers
        self.distance_sub = self.create_subscription(
            WheelOdomStamped,
            self.distance_topic,
            self.distance_callback,
            qos_subber,
        )

        self.rc_sub = self.create_subscription(
            RCIn,
            self.rc_topic,
            self.rc_callback,
            qos_subber,
        )

        self.get_logger().info("MAVROS encoder-to-twist node started")
        self.get_logger().info(f"Distance input: {self.distance_topic}")
        self.get_logger().info(f"RC input: {self.rc_topic}")
        self.get_logger().info(f"Twist output: {self.output_topic}")
        self.get_logger().info(
            f"RC throttle channel={self.rc_throttle_channel}, "
            f"neutral={self.rc_neutral}, deadband={self.rc_deadband}"
        )

    def rc_callback(self, msg):
        if len(msg.channels) <= self.rc_throttle_channel:
            self.get_logger().warn(
                f"RC message has only {len(msg.channels)} channels, "
                f"cannot read channel {self.rc_throttle_channel}"
            )
            return

        throttle = int(msg.channels[self.rc_throttle_channel])
        self.last_rc_throttle = throttle

        upper = self.rc_neutral + self.rc_deadband
        lower = self.rc_neutral - self.rc_deadband

        if throttle > upper:
            self.rc_direction = 1
        elif throttle < lower:
            self.rc_direction = -1
        else:
            self.rc_direction = 0

    def stamp_to_sec(self, stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def distance_callback(self, msg):
        if len(msg.data) < 2:
            self.get_logger().warn("WheelOdomStamped has fewer than 2 distance values")
            return

        left = float(msg.data[0])
        right = float(msg.data[1])
        current_time = self.stamp_to_sec(msg.header.stamp)

        if self.prev_time is None:
            self.prev_left = left
            self.prev_right = right
            self.prev_time = current_time
            self.get_logger().info("First wheel distance message received")
            return

        dt = current_time - self.prev_time

        if dt <= 0.0:
            self.get_logger().warn(f"Invalid dt={dt:.6f}, skipping message")
            return

        d_left = left - self.prev_left
        d_right = right - self.prev_right

        self.prev_left = left
        self.prev_right = right
        self.prev_time = current_time

        # MAVROS wheel distance only increases, independent of direction.
        # Therefore the encoder gives speed magnitude only.
        v_left_unsigned = abs(d_left / dt)
        v_right_unsigned = abs(d_right / dt)

        # Direction comes from RC throttle.
        direction = self.rc_direction

        v_left = direction * v_left_unsigned
        v_right = direction * v_right_unsigned

        linear_x = 0.5 * (v_left + v_right)
        angular_z = (v_right - v_left) / self.wheel_track

        out = TwistWithCovarianceStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id

        out.twist.twist.linear.x = linear_x
        out.twist.twist.linear.y = 0.0
        out.twist.twist.linear.z = 0.0

        out.twist.twist.angular.x = 0.0
        out.twist.twist.angular.y = 0.0
        out.twist.twist.angular.z = angular_z

        # Covariance matrix order:
        # x, y, z, rot_x, rot_y, rot_z
        out.twist.covariance[0] = self.vx_covariance
        out.twist.covariance[7] = 999.0
        out.twist.covariance[14] = 999.0
        out.twist.covariance[21] = 999.0
        out.twist.covariance[28] = 999.0
        out.twist.covariance[35] = self.yaw_rate_covariance

        self.encoder_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = MavrosEncoderToTwist()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()