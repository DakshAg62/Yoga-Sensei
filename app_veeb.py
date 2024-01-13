from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# reference_image = cv2.imread('veerbasic.jpg')
reference_image = cv2.imread('d:\One Drive\OneDrive\Pictures\Camera Roll\WIN_20231218_17_10_45_Pro.jpg')
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
reference_results = pose.process(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))

reference_landmarks = []
if reference_results.pose_landmarks:
    for i, landmark in enumerate(reference_results.pose_landmarks.landmark):
        reference_landmarks.append((landmark.x, landmark.y, i))

app = Flask(__name__)

def generate_frames():
    cap = cv2.VideoCapture(0)
    keypoint_labels = {
    "Perfect": "Perfect! Keep Going :)",
    "HandsNotAtRightPosition": "Hands not at right position",
    "HandsNotAt90": "Bring your hands\nin a straight line across shoulders",
    "LegNot90": "front leg should make\n90 degrees with hip and ankle",
    "LegsWrong": "Please bend to perform\nthe asana",
    "Idle": "Idle, please perform asana",
    "LegNotStraight": "straighten your other leg",
    "ShoulderWrong": "stand straight in\nfront of the camera"
}
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_results = pose.process(frame_rgb)

        frame_landmarks = []
        if frame_results.pose_landmarks:
            for i, landmark in enumerate(frame_results.pose_landmarks.landmark):
                frame_landmarks.append((landmark.x, landmark.y, i))

        if len(frame_landmarks) > 0 and len(reference_landmarks) > 0:
            initial_offset = np.array(reference_landmarks[0][:2]) - np.array(frame_landmarks[0][:2])

        frame_copy = frame.copy()
        leg_not_90 = False
        hands_not_at_right_position = False
        hands_not_at_90 = False
        shoulder_wrong = False
        leg_not_straight = False

        for frame_landmark in frame_landmarks:
            frame_point = frame_landmark[:2]
            frame_keypoint_number = frame_landmark[2]

            adjusted_reference_point = (reference_landmarks[frame_keypoint_number][0] - initial_offset[0], reference_landmarks[frame_keypoint_number][1] - initial_offset[1])
            distance = calculate_distance(adjusted_reference_point, frame_point)
            threshold = 0.08  ###############################################################################################  

            if distance < threshold:
                cv2.circle(frame_copy, (int(frame_point[0] * frame.shape[1]), int(frame_point[1] * frame.shape[0])), 5, (0, 255, 0), -1)
            else:
                if frame_keypoint_number in [26, 28, 30, 32]:
                    leg_not_90 = True
                if frame_keypoint_number in [25, 27, 29, 31]:
                    leg_not_straight = True
                if frame_keypoint_number == 13 or frame_keypoint_number == 14:
                    hands_not_at_90 = True
                if frame_keypoint_number in [16, 18, 20, 22] or frame_keypoint_number in [15, 17, 19, 21]:
                    hands_not_at_right_position = True
                if frame_keypoint_number in [11] or frame_keypoint_number in [12]:
                    shoulder_wrong = True
                cv2.circle(frame_copy, (int(frame_point[0] * frame.shape[1]), int(frame_point[1] * frame.shape[0])), 5, (0, 0, 255), -1)

        if shoulder_wrong:
            label = keypoint_labels["ShoulderWrong"]
        elif leg_not_90:
            if leg_not_straight:
                if hands_not_at_90:
                    label = keypoint_labels["Idle"]
                else:
                    label = keypoint_labels["LegsWrong"]
            else:
                label = keypoint_labels["LegNot90"]
        elif leg_not_straight:
            label = keypoint_labels["LegNotStraight"]
        elif hands_not_at_right_position:
            if hands_not_at_90:
                label = keypoint_labels["HandsNotAt90"]
            else:
                label = keypoint_labels["HandsNotAtRightPosition"]
        else:
            label = keypoint_labels["Perfect"]
        label_lines = label.split('\n')
        for i, line in enumerate(label_lines):
            cv2.putText(frame_copy, line, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame_copy)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('veer_basic.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True,port=5003)
