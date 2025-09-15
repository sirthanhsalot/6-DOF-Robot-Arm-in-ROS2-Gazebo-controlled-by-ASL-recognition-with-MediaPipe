#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from interfaces.msg import HandChess

import csv
import copy
import itertools
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from cv.utils import CvFpsCalc
from cv.model import KeyPointClassifier


class HandGesturePublisher(Node):
    def __init__(self):
        super().__init__('hand_gesture_publisher')
        self.publisher_ = self.create_publisher(HandChess, 'hand_gesture', 10)

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

        # Persist history across frames
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)

        # ---------------- Debounce / stabilization state ----------------
        self.debounce_N = 6           # frames
        self.debounce_ms = 180        # milliseconds
        self._candidate = None        # (left_hand:str, right_hand:int)
        self._candidate_count = 0
        self._candidate_started = self.get_clock().now()
        self._stable = None           # last stable pair
        self._stable_valid = False

        # Last rendered status for on-screen overlay
        self._ui_left = None
        self._ui_right = None
        self._ui_coord = ""

        self.timer = self.create_timer(0.03, self.timer_callback)  # ~30 FPS

    # ------------------------ Main loop ------------------------
    def timer_callback(self):
        fps = self.cvFpsCalc.get()
        key = cv.waitKey(1)
        if key == 27:  # ESC
            return

        ret, image = self.cap.read()
        if not ret:
            self.get_logger().error('Failed to capture image')
            return

        # Mirror the image for a selfie-view UI
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        # MediaPipe processing
        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        # ---------------- Detect both hands & classify ----------------
        left_letter = None   # 'a'..'h'
        right_number = None  # 1..8

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = self.calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = self.calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = self.pre_process_landmark(landmark_list)
                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)

                # Draw rect + landmarks
                debug_image = self.draw_bounding_rect(True, debug_image, brect)
                debug_image = self.draw_landmarks(debug_image, landmark_list)

                # Handedness label from MediaPipe is relative to the (flipped) image.
                # After cv.flip(image, 1), MediaPipe's "Left" appears on the user's RIGHT hand.
                user_hand = handedness.classification[0].label  # 'Left' or 'Right'
                # 'Left'/'Right' (user perspective)

                # Map classifier id -> label string
                gesture_label = "Unknown"
                if hand_sign_id != -1:
                    gesture_label = self.keypoint_classifier_labels[hand_sign_id].strip().lower()

                # Put overlay text for that hand
                debug_image = self.draw_info_text(debug_image, brect, handedness, gesture_label if gesture_label else "Unknown")

                # Parse to letter/number depending on which hand it is
                letter, number = self.parse_label(gesture_label)

                if user_hand == 'Left' and letter is not None and letter in "abcdefgh":
                    left_letter = letter
                if user_hand == 'Right' and number is not None and 1 <= number <= 8:
                    right_number = number

        # --------------- Debounce: require stable pair ---------------
        valid, changed, pair = self._update_stable_pair(left_letter, right_number)

        # Build + publish message
        msg = HandChess()
        if valid:
            lh, rh = pair  # lh: 'a'..'h', rh: 1..8
            coord = f"{lh}{rh}"
            msg.left_hand = lh
            msg.right_hand = rh
            msg.coordinate = coord
            self.publisher_.publish(msg)

            # For UI overlay
            self._ui_left = lh
            self._ui_right = rh
            self._ui_coord = coord
        else:
            # Publish nothing when unstable; just keep UI informative
            pass

        # ------------------- UI overlays (FPS + status) -------------------
        debug_image = self.draw_info(debug_image, fps)

        cv.imshow('Hand Gesture Recognition', debug_image)

    # ------------------------ Debounce core ------------------------
    def _update_stable_pair(self, left_hand, right_hand):
        """
        Returns (valid, changed, stable_pair)
          valid: bool – stable pair detected
          changed: bool – stable_pair changed
          stable_pair: tuple (left_hand:str, right_hand:int) or None
        """
        if left_hand is None or right_hand is None:
            self._candidate = None
            self._candidate_count = 0
            self._stable_valid = False
            return False, False, None

        pair = (left_hand, right_hand)
        now = self.get_clock().now()

        if self._candidate != pair:
            self._candidate = pair
            self._candidate_count = 1
            self._candidate_started = now
            self._stable_valid = False
            return False, False, None
        else:
            self._candidate_count += 1
            elapsed_ms = (now - self._candidate_started).nanoseconds / 1e6

            if self._candidate_count >= self.debounce_N or elapsed_ms >= self.debounce_ms:
                changed = (self._stable != pair) or (not self._stable_valid)
                self._stable = pair
                self._stable_valid = True
                return True, changed, pair

            return False, False, None

    # ------------------------ Helpers ------------------------

    def parse_label(self, label):
        """
        Parse a classifier label into (letter, number).
        Accept 'a'..'h' for letters and '1'..'8' (or 'one'..'eight') for numbers.
        Returns (letter:str or None, number:int or None).
        """
        if not label:
            return None, None

        # Normalize
        t = label.strip().lower()

        # Try letter
        if len(t) == 1 and 'a' <= t <= 'h':
            return t, None

        # Try number as digit
        if t.isdigit():
            try:
                v = int(t)
                if 1 <= v <= 8:
                    return None, v
            except ValueError:
                pass

        # Try number as word
        words_to_num = {
            'one':1, 'two':2, 'three':3, 'four':4,
            'five':5, 'six':6, 'seven':7, 'eight':8
        }
        if t in words_to_num:
            return None, words_to_num[t]

        return None, None

    # ------------------------ Drawing & geometry ------------------------
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
            temp_landmark_list[index][0] -= base_x
            temp_landmark_list[index][1] -= base_y
        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        max_value = max(list(map(abs, temp_landmark_list))) if temp_landmark_list else 1
        temp_landmark_list = [n / max_value for n in temp_landmark_list]
        return temp_landmark_list

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

    def draw_bounding_rect(self, use_brect, image, brect):
        if use_brect:
            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
        return image

    def draw_info_text(self, image, brect, handedness, hand_sign_text):
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
        info_text = handedness.classification[0].label[0:]
        if hand_sign_text != "":
            info_text = info_text + ':' + hand_sign_text
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        return image

    def draw_info(self, image, fps):
        # FPS
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

        # Current stable output (if any)
        y = 60
        if self._ui_left is not None:
            cv.putText(image, f"LEFT (letter): {self._ui_left}", (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv.LINE_AA)
            cv.putText(image, f"LEFT (letter): {self._ui_left}", (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv.LINE_AA)
            y += 24
        if self._ui_right is not None:
            cv.putText(image, f"RIGHT (number): {self._ui_right}", (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv.LINE_AA)
            cv.putText(image, f"RIGHT (number): {self._ui_right}", (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv.LINE_AA)
            y += 24
        if self._ui_coord:
            cv.putText(image, f"COORD: {self._ui_coord}", (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv.LINE_AA)
            cv.putText(image, f"COORD: {self._ui_coord}", (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv.LINE_AA)
        return image

    def draw_landmarks(self, image, landmark_point):
        if len(landmark_point) > 0:
            # Thumb
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)

            # Index finger
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 2)

            # Middle finger
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 2)

            # Ring finger
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 2)

            # Little finger
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 2)

            # Palm
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 2)
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
