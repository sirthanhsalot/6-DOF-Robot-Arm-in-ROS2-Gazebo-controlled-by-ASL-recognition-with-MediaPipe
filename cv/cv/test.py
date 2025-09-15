import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp

class HandLandmarkNode(Node):
    def __init__(self):
        super().__init__('hand_landmark_node')
        self.publisher_ = self.create_publisher(Image, 'hand_landmark_image', 10)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to capture frame')
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_hands.HAND_CONNECTIONS
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher_.publish(img_msg)
        cv2.imshow('Hand Landmarks', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = HandLandmarkNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()