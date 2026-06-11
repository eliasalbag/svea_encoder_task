#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from geometry_msgs.msg import TransformStamped
from mocap4r2_msgs.msg import RigidBodies
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
import tf_transformations as tr


qos_subber = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)

qos_pubber = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class MocapToOdom(Node):
    def __init__(self):
        super().__init__("mocap_to_odom")

        self.declare_parameter("input_topic", "/rigid_bodies")
        self.declare_parameter("rigid_body_name", "svea3")
        self.declare_parameter("output_topic", "/svea3/odometry/mocap")
        self.declare_parameter("frame_id", "")
        self.declare_parameter("child_frame_id", "svea3/base_link")
        self.declare_parameter("two_d_mode", True)
        self.declare_parameter("publish_tf", False)

        self.declare_parameter("x_covariance", 0.0025)
        self.declare_parameter("y_covariance", 0.0025)
        self.declare_parameter("z_covariance", 999.0)
        self.declare_parameter("roll_covariance", 999.0)
        self.declare_parameter("pitch_covariance", 999.0)
        self.declare_parameter("yaw_covariance", 0.01)

        self.input_topic = self.get_parameter("input_topic").value
        self.rigid_body_name = self.get_parameter("rigid_body_name").value
        self.output_topic = self.get_parameter("output_topic").value
        self.frame_id = self.get_parameter("frame_id").value
        self.child_frame_id = self.get_parameter("child_frame_id").value
        self.two_d_mode = bool(self.get_parameter("two_d_mode").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)

        self.pose_covariance = [0.0] * 36
        self.pose_covariance[0] = float(self.get_parameter("x_covariance").value)
        self.pose_covariance[7] = float(self.get_parameter("y_covariance").value)
        self.pose_covariance[14] = float(self.get_parameter("z_covariance").value)
        self.pose_covariance[21] = float(self.get_parameter("roll_covariance").value)
        self.pose_covariance[28] = float(self.get_parameter("pitch_covariance").value)
        self.pose_covariance[35] = float(self.get_parameter("yaw_covariance").value)

        self.odom_pub = self.create_publisher(Odometry, self.output_topic, qos_pubber)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None
        self.rb_sub = self.create_subscription(
            RigidBodies,
            self.input_topic,
            self.rigid_bodies_callback,
            qos_subber,
        )

        self.get_logger().info(
            f"Publishing rigid body '{self.rigid_body_name}' from "
            f"{self.input_topic} as {self.output_topic}"
        )
        self.get_logger().info(
            f"child_frame_id={self.child_frame_id}, "
            f"two_d_mode={self.two_d_mode}, publish_tf={self.publish_tf}"
        )

    def rigid_bodies_callback(self, msg):
        for rigid_body in msg.rigidbodies:
            if rigid_body.rigid_body_name == self.rigid_body_name:
                self.publish_odometry(msg, rigid_body.pose)
                return

    def publish_odometry(self, msg, pose):
        frame_id = self.frame_id or msg.header.frame_id

        odom = Odometry()
        odom.header = msg.header
        odom.header.frame_id = frame_id
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose = pose
        odom.pose.covariance = self.pose_covariance

        if self.two_d_mode:
            odom.pose.pose.position.z = 0.0
            yaw = self.yaw_from_quaternion(odom.pose.pose.orientation)
            q = tr.quaternion_from_euler(0.0, 0.0, yaw)
            odom.pose.pose.orientation.x = q[0]
            odom.pose.pose.orientation.y = q[1]
            odom.pose.pose.orientation.z = q[2]
            odom.pose.pose.orientation.w = q[3]

        self.odom_pub.publish(odom)

        if self.tf_broadcaster is not None:
            transform = TransformStamped()
            transform.header = odom.header
            transform.child_frame_id = odom.child_frame_id
            transform.transform.translation.x = odom.pose.pose.position.x
            transform.transform.translation.y = odom.pose.pose.position.y
            transform.transform.translation.z = odom.pose.pose.position.z
            transform.transform.rotation = odom.pose.pose.orientation
            self.tf_broadcaster.sendTransform(transform)

    @staticmethod
    def yaw_from_quaternion(quaternion):
        _, _, yaw = tr.euler_from_quaternion(
            [
                quaternion.x,
                quaternion.y,
                quaternion.z,
                quaternion.w,
            ]
        )
        if math.isnan(yaw):
            return 0.0
        return yaw


def main(args=None):
    rclpy.init(args=args)
    node = MocapToOdom()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
