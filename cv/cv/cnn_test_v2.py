#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import csv
import copy
import itertools
from collections import deque, Counter

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier


class HandGesturePublisher(Node):
    def __init__(self):
        super().__init__('hand_gesture_publisher')
        self.publisher_ = self.create_publisher(String, 'hand_gesture', 10)

        # Camera and model setup
        self.cap_device = 0
        self.cap_width = 960
        self.cap_height = 540
        self.use_static_image_mode = False
        self.min_detection_confidence = 0.7
        self.min_tracking_confidence = 0.5

        self.cap = cv.VideoCapture(self.cap_device)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.cap_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.cap_height)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.use_static_image_mode,
            max_num_hands=2,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.keypoint_classifier = KeyPointClassifier()
        with open('cv/model/keypoint_classifier/keypoint_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

        self.cvFpsCalc = CvFpsCalc(buffer_len=10)
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)
        self.finger_gesture_history = deque(maxlen=self.history_length)

        self.timer = self.create_timer(0.03, self.timer_callback)  # ~30 FPS

    def timer_callback(self):
        fps = self.cvFpsCalc.get()
        key = cv.waitKey(10)
        if key == 27:  # ESC
            return
        mode = 0
        use_brect = True    

        number, mode = self.select_mode(key, mode)

        history_length = 16
        point_history = deque(maxlen=history_length)

        ret, image = self.cap.read()
        if not ret:
            self.get_logger().error('Failed to capture image')
            return

        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True
        
        gesture = "Unknown"
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = self.calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = self.pre_process_landmark(landmark_list)
                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)

                debug_image = self.draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = self.draw_landmarks(debug_image, landmark_list)

                if hand_sign_id != -1:
                    gesture = self.keypoint_classifier_labels[hand_sign_id]
                    debug_image = self.draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        gesture,
                    )
                else:
                    gesture = "Unknown"
                    debug_image = self.draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        'Unknown',
                    )
                # Continue to process and publish gesture for each detected hand

        debug_image = self.draw_point_history(debug_image, point_history)
        debug_image = self.draw_info(debug_image, fps, mode, number)

        cv.imshow('Hand Gesture Recognition', debug_image)

        msg = String()
        msg.data = gesture
        self.publisher_.publish(msg)

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])
        return landmark_point

    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)
        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]
            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        max_value = max(list(map(abs, temp_landmark_list))) if temp_landmark_list else 1
        temp_landmark_list = [n / max_value for n in temp_landmark_list]
        return temp_landmark_list

    def draw_bounding_rect(self,use_brect, image, brect):
        if use_brect:
            # Outer rectangle
            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                        (0, 0, 0), 1)
        return image

    def draw_info_text(self, image, brect, handedness, hand_sign_text):
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                    (0, 0, 0), -1)

        info_text = handedness.classification[0].label[0:]
        if hand_sign_text != "":
            info_text = info_text + ':' + hand_sign_text

        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        return image

    def draw_point_history(self,image, point_history):
        for index, point in enumerate(point_history):
            if point[0] != 0 and point[1] != 0:
                cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                        (152, 251, 152), 2)
        return image
    
    def draw_info(self, image, fps, mode, number):
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv.LINE_AA)

        mode_string = ['Logging Key Point', 'Logging Point History']
        if 1 <= mode <= 2:
            cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
                    cv.LINE_AA)
            if 0 <= number <= 7:
                cv.putText(image, "NUM:" + str(number+1), (10, 110),
                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
                        cv.LINE_AA)
            elif 8 <= number <= 15:
                letter = chr(ord('a') + number - 8)
                cv.putText(image, "LET: " + letter, (10, 110),
                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
                        cv.LINE_AA)
        return image

    def calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv.boundingRect(landmark_array)

        return [x, y, x + w, y + h]
    
    def select_mode(self, key, mode):
        number = -1
        if 49 <= key <= 56:  # ord('1')=49, ord('8')=56
            number = key - 49  # 0–7
        elif 97 <= key <= 104:  # ord('a')=97, ord('h')=104
            number = key - 97 + 8  # 8–15

        if key == 110:  # n
            mode = 0
        if key == 107:  # k
            mode = 1
        # if key == 104:  # h
        #     mode = 2
        return number, mode
    
    def draw_landmarks(self, image, landmark_point):
        if len(landmark_point) > 0:
            # Thumb
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (255, 255, 255), 2)

            # Index finger
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    (255, 255, 255), 2)

            # Middle finger
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                    (255, 255, 255), 2)

            # Ring finger
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                    (255, 255, 255), 2)

            # Little finger
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (255, 255, 255), 2)

            # Palm
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (255, 255, 255), 2)
        return image

def main(args=None):
    rclpy.init(args=args)
    node = HandGesturePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        cv.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

