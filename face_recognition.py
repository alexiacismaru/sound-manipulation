# import numpy as np
# import math
# import cv2 as cv

# cap = cv.VideoCapture(0)

# while True: 
#     _, img = cap.read()
#     cv.rectangle(img, (300, 300), (100, 100), (0, 255, 0), 0)
#     crop_img = img[100:300, 100:300]
#     grey = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
#     value = (35, 35)
#     blurred_ = cv.GaussianBlur(grey, value, 0)
#     _, thresholded = cv.threshold(blurred_, 127, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    
#     contours,hierachy = cv.findContours(thresholded.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#     count1 = max(contours, key=lambda x: cv.contourArea(x))
#     x, y, w, h = cv.boundingRect(count1)
#     cv.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)
#     hull = cv.convexHull(count1)
#     drawing = np.zeros(crop_img.shape, np.uint8)
#     cv.drawContours(drawing, [count1], 0, (0, 255, 0), 0)
#     cv.drawContours(drawing, [hull], 0, (0, 0, 255), 0)
#     hull = cv.convexHull(count1, returnPoints=False)
#     defects = cv.convexityDefects(count1, hull)

#     count_defects = 0
#     cv.drawContours(thresholded, contours, -1, (0, 255, 0), 3)

#     for i in range(defects.shape[0]):
#         s, e, f, d = defects[i, 0]
#         start = tuple(count1[s][0])
#         end = tuple(count1[e][0])
#         far = tuple(count1[f][0])
#         a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
#         b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
#         c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
#         angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

#         if angle <= 90:
#             count_defects += 1
#             cv.circle(crop_img, far, 1, [0, 0, 255], -1)

#         cv.line(crop_img, start, end, [0, 255, 0], 2)

#         if count_defects == 1:
#             cv.putText(img, "2 fingers", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
#         elif count_defects == 2:
#             str = "3 fingers"
#             cv.putText(img, str, (5, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
#         elif count_defects == 3:
#             cv.putText(img, "4 fingers", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
#         elif count_defects == 4:
#             cv.putText(img, "5 fingers", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
#         elif count_defects == 0:
#             cv.putText(img, "one", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))

#             cv.imshow('main window', img)
#             all_img = np.hstack((drawing, crop_img))
#     # Press 'q' to exit the loop
#     if cv.waitKey(1) == ord('q'):
#         break

# # Release the capture 
# cap.release() 
# cv.destroyAllWindows()

import csv
import copy 
import itertools 
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
 
from model import KeyPointClassifier 

def main():  
    cap = cv.VideoCapture(0)
    use_brect = True 

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands( 
        max_num_hands=1, 
    )

    keypoint_classifier = KeyPointClassifier() 

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ] 

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # # Finger gesture history ################################################
    # finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0

    while True: 
        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                # pre_processed_point_history_list = pre_process_point_history(
                #     debug_image, point_history) 

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                # finger_gesture_id = 0
                # point_history_len = len(pre_processed_point_history_list) 

                # Calculates the gesture IDs in the latest detection
                # finger_gesture_history.append(finger_gesture_id) 

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                # debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id], 
                )
        else:
            point_history.append([0, 0])
 
        debug_image = draw_info(debug_image, mode, number)

        # Screen reflection 
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1) 
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


# def draw_landmarks(image, landmark_point):
#     if len(landmark_point) > 0:
#         # Thumb
#         cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
#                 (255, 255, 255), 2)

#         # Index finger
#         cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
#                 (255, 255, 255), 2)

#         # Middle finger
#         cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
#                 (255, 255, 255), 2)

#         # Ring finger
#         cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
#                 (255, 255, 255), 2)

#         # Little finger
#         cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
#                 (255, 255, 255), 2)

#         # Palm
#         cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
#                 (255, 255, 255), 2)
#         cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
#                 (0, 0, 0), 6)
#         cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
#                 (255, 255, 255), 2)

#     # Key Points
#     for index, landmark in enumerate(landmark_point):
#         if index == 0:  # 手首1
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 1:  # 手首2
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 2:  # 親指：付け根
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 3:  # 親指：第1関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 4:  # 親指：指先
#             cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
#         if index == 5:  # 人差指：付け根
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 6:  # 人差指：第2関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 7:  # 人差指：第1関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 8:  # 人差指：指先
#             cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
#         if index == 9:  # 中指：付け根
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 10:  # 中指：第2関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 11:  # 中指：第1関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 12:  # 中指：指先
#             cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
#         if index == 13:  # 薬指：付け根
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 14:  # 薬指：第2関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 15:  # 薬指：第1関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 16:  # 薬指：指先
#             cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
#         if index == 17:  # 小指：付け根
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 18:  # 小指：第2関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 19:  # 小指：第1関節
#             cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
#         if index == 20:  # 小指：指先
#             cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
#                       -1)
#             cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

#     return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image  

def draw_info(image, mode, number): 
    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
