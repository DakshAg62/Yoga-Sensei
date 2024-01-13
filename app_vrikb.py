from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

app = Flask(__name__)

pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

reference_image = cv2.imread('d:\One Drive\OneDrive\Pictures\Camera Roll\WIN_20231218_17_09_29_Pro.jpg')
reference_results = pose.process(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))
reference_landmarks = []

if reference_results.pose_landmarks:
    for i, landmark in enumerate(reference_results.pose_landmarks.landmark):
        reference_landmarks.append((landmark.x, landmark.y, i))

keypoint_labels = {
    "Perfect": "Perfect! Keep Going :)",
    "HandsNotAtRightPosition": "Hands not at right position",
    "HandsNotAt90": "Hands not at 90 degree",
    "LegsNotTriangle": "Legs not triangle",
    "Hands_legs_wrong": "Hands and leg both wrong",
    "LegDown": "Leg Down",
    "Idle": "Idle, please perform asana",
    "BentRight": "Bent right, \nstraighten yourself",
    "BentLeft": "Bent left, \nstraighten yourself",
    "BentForward": "Bent forward, straighten yourself",
    "WrongPosture" : "Straighten yourself"
}

def analyze_posture(frame, initial_offset, frame_landmarks):
    leg_down = False
    hands_not_at_right_position = False
    hands_not_at_90 = False
    bent_right = False
    bent_left = False
    bent_forward = False
    left_shoulder_y = None
    right_shoulder_y = None

    for frame_landmark in frame_landmarks:
        frame_point = frame_landmark[:2]
        frame_keypoint_number = frame_landmark[2]

        if frame_keypoint_number == 11:  # Left Shoulder
            left_shoulder_y = frame_point[1]
        elif frame_keypoint_number == 12:  # Right Shoulder
            right_shoulder_y = frame_point[1]
        elif frame_keypoint_number == 13:  # Left Elbow
            left_elbow_y = frame_point[1]
        elif frame_keypoint_number == 14:  # Right Elbow
            right_elbow_y = frame_point[1]
        
        adjusted_left_shoulder_point = (reference_landmarks[11][0] - initial_offset[0], reference_landmarks[11][1] - initial_offset[1])
        adjusted_right_shoulder_point = (reference_landmarks[12][0] - initial_offset[0], reference_landmarks[12][1] - initial_offset[1])

        # Obtain y-coordinates of the adjusted reference frame for left and right shoulders
        adjusted_left_shoulder_y = adjusted_left_shoulder_point[1]
        adjusted_right_shoulder_y = adjusted_right_shoulder_point[1]

        adjusted_reference_point = (reference_landmarks[frame_keypoint_number][0] - initial_offset[0], reference_landmarks[frame_keypoint_number][1] - initial_offset[1])
        distance = calculate_distance(adjusted_reference_point, frame_point)
        threshold = 0.05

        if distance < threshold:
            cv2.circle(frame, (int(frame_point[0] * frame.shape[1]), int(frame_point[1] * frame.shape[0])), 5, (0, 255, 0), -1)
        else:
            if frame_keypoint_number in [28, 30, 32] or frame_keypoint_number in [27, 29, 31] or frame_keypoint_number == 25 or frame_keypoint_number == 26:
                leg_down = True
            if frame_keypoint_number == 13 or frame_keypoint_number == 14:
                hands_not_at_90 = True
            if frame_keypoint_number in [16, 18, 20, 22] or frame_keypoint_number in [15, 17, 19, 21]:
                hands_not_at_right_position = True
            if frame_keypoint_number == 11:
                bent_left = True
            if frame_keypoint_number == 12:
                bent_right = True
                
            cv2.circle(frame, (int(frame_point[0] * frame.shape[1]), int(frame_point[1] * frame.shape[0])), 5, (0, 0, 255), -1)

    if bent_left:
        if bent_right:
            label = keypoint_labels["WrongPosture"]
        else:
            label = keypoint_labels["BentLeft"]
    elif bent_right:
        label = keypoint_labels["BentRight"]
    elif leg_down:
        if hands_not_at_right_position:
            if not hands_not_at_90:
                label = keypoint_labels["Idle"]
            else:
                label = keypoint_labels["Hands_legs_wrong"]
        else:
            label = keypoint_labels["LegDown"]
    elif hands_not_at_90:
        label = keypoint_labels["HandsNotAt90"]
    elif hands_not_at_right_position:
        label = keypoint_labels["HandsNotAtRightPosition"]
    else:
        label = keypoint_labels["Perfect"]

    label_lines = label.split('\n')
    for i, line in enumerate(label_lines):
        cv2.putText(frame, line, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

@app.route('/')
def index():
    return render_template('vrikb.html')

def generate_frames():
    cap = cv2.VideoCapture(0)

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
        analyze_posture(frame_copy, initial_offset, frame_landmarks)

        ret, buffer = cv2.imencode('.jpg', frame_copy)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5002)
