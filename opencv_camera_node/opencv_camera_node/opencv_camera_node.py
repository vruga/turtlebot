#!/usr/bin/env python3
"""
OpenCV camera node for TurtleBot on Raspberry Pi.
Publishes sensor_msgs/Image on /camera/image_raw (and CompressedImage on /camera/image_compressed).
Integrates with SLAM/navigation; no RVIZ required.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2


class OpenCVCameraNode(Node):
    def __init__(self):
        super().__init__('opencv_camera_node')

        self.declare_parameter('device_id', 0)
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('publish_compressed', True)

        pv = self.get_parameter('device_id').get_parameter_value()
        try:
            device_id = int(pv.integer_value)
        except (AttributeError, TypeError):
            device_id = int(getattr(pv, 'string_value', '0') or '0')
        width = self.get_parameter('width').get_parameter_value().integer_value
        height = self.get_parameter('height').get_parameter_value().integer_value
        fps = self.get_parameter('fps').get_parameter_value().double_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        publish_compressed = self.get_parameter('publish_compressed').get_parameter_value().bool_value

        self.image_pub = self.create_publisher(Image, 'camera/image_raw', 10)
        if publish_compressed:
            self.compressed_pub = self.create_publisher(
                CompressedImage, 'camera/image_compressed', 10
            )
        else:
            self.compressed_pub = None

        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            self.get_logger().error(
                f'Could not open camera device_id={device_id}. '
                'Try device_id=0 or 2 (e.g. /dev/video0, /dev/video2).'
            )
            raise RuntimeError('Camera open failed')

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        period_ms = int(1000.0 / fps) if fps > 0 else 33
        self.timer = self.create_timer(1.0 / fps, self.timer_callback)

        self.get_logger().info(
            f'OpenCV camera node started: /camera/image_raw '
            f'({width}x{height} @ {fps} Hz, frame_id={self.frame_id})'
        )

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to read frame', throttle_duration_sec=5.0)
            return

        stamp = self.get_clock().now().to_msg()
        header = Header(stamp=stamp, frame_id=self.frame_id)

        try:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header = header
            self.image_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}', throttle_duration_sec=5.0)
            return

        if self.compressed_pub is not None:
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if jpeg is not None:
                cmsg = CompressedImage()
                cmsg.header = header
                cmsg.format = 'jpeg'
                cmsg.data = jpeg.tobytes()
                self.compressed_pub.publish(cmsg)

    def destroy_node(self, *args, **kwargs):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        super().destroy_node(*args, **kwargs)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = OpenCVCameraNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, RuntimeError):
        pass
    finally:
        if rclpy.ok():
            try:
                node.destroy_node()
            except NameError:
                pass
            rclpy.shutdown()


if __name__ == '__main__':
    main()
