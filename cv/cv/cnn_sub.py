import rclpy
from rclpy.node import Node
from interfaces.msg import HandChess
from interfaces.srv import ComputeIK
import time

class ChessSubscriber(Node):
    def __init__(self):
        super().__init__('chess_subscriber')
        self.create_subscription(HandChess, 'hand_gesture', self.callback, 10)
        self.get_logger().info("Chess Subscriber Node has been started.")
        self.IK_client = self.create_client(ComputeIK, "computation_service")
        self.get_logger().info("IK Client has been created.")

        self.last_msg = None
        self.last_msg_time = None
        self.stable_msg = None
        self.timer = self.create_timer(0.5, self.check_stable_msg)

    def callback(self, msg):
        # Only compare relevant fields for stability
        msg_tuple = (msg.coordinate, msg.left_hand, msg.right_hand)
        now = self.get_clock().now().nanoseconds / 1e9
        if msg_tuple != self.last_msg:
            self.last_msg = msg_tuple
            self.last_msg_time = now
            self.stable_msg = None  # Reset stability
        self.get_logger().info(f"Square: {msg.coordinate} (Left: {msg.left_hand}, Right: {msg.right_hand})")

    def check_stable_msg(self):
        if self.last_msg is None:
            return
        now = self.get_clock().now().nanoseconds / 1e9
        if self.last_msg_time is not None and (now - self.last_msg_time) >= 3.0:
            if self.stable_msg != self.last_msg:
                self.stable_msg = self.last_msg
                self.get_logger().info(f"Stable message for 3 seconds: {self.stable_msg}. Requesting IK...")
                self.call_compute_IK_from_msg(self.stable_msg)

    def square_to_xyz(self, square):
        """
        Convert chessboard square (e.g., 'a1', 'h8') to (x, y, z) coordinates.
        Example: a1 = (0.0, 0.0, 0.0), b1 = (1.0, 0.0, 0.0), a2 = (0.0, 1.0, 0.0), etc.
        Adjust the mapping and scaling as needed for your setup.
        """
        if len(square) != 2:
            return 0.0, 0.0, 0.0  # Default/fallback

        file = square[0].lower()
        rank = square[1]

        files = 'abcdefgh'
        try:
            x = float((int(rank) - 1)*5 + 13)
            y = float(22 - (files.index(file)*5))
            z = 2.0  # Set z as needed, or use another field if needed
        except Exception:
            x, y, z = 0.0, 0.0, 0.0
        return x, y, z

    def call_compute_IK_from_msg(self, msg_tuple):
        coordinate, left_hand, right_hand = msg_tuple
        x, y, z = self.square_to_xyz(coordinate)
        req = ComputeIK.Request()
        req.x = x
        req.y = y
        req.z = z
        req.initial_guess_joint_angles = [0.0] * 6  # Or use a better guess if available

        future = self.IK_client.call_async(req)
        future.add_done_callback(self.ik_response_callback)

    def ik_response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f"IK Response: {response.desired_joint_angles}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ChessSubscriber()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()