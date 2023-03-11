import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import queue


# To play audio text-to-speech during execution
class TTSThread(threading.Thread):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self.daemon = True
        self.start()

    def run(self):
        tts_engine = pyttsx3.init()
        tts_engine.startLoop(False)
        while True:
            data = self.queue.get()
            if data == "exit":
                break
            else:
                tts_engine.say(data)
                tts_engine.iterate()
        tts_engine.endLoop()



q = queue.Queue()
tts_thread = TTSThread(q)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

window_name = 'WORKOUT COUNTER'

cap = cv2.VideoCapture(0)

# cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# grab the width and height of the frames in the video stream
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# set the fps
fps = 5

# initialize the FourCC codec and create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

# Counter Variables
# left_bicep_counter and right_bicep_counter are used to store the count of bicep curls
# left_bicep_stage and right_bicep_stage are used to store the stage of the bicep curl
left_bicep_counter, right_bicep_counter = 0, 0
left_bicep_stage, right_bicep_stage = None, None

# Squat Counter Variables
# angle_min_right_knee and angle_min_right_hip are used to store the minimum angle for the right knee and hip
# angle_min_left_knee and angle_min_left_hip are used to store the minimum angle for the left knee and hip
angle_min_right_knee, angle_min_right_hip = [], []
angle_min_left_knee, angle_min_left_hip = [], []

# squat_counter is used to store the count of squats
squat_counter = 0

# min_ang_right_knee and max_ang_right_knee are used to store the minimum and maximum angle for the right knee
# min_ang_right_hip and max_ang_right_hip are used to store the minimum and maximum angle for the right hip
# min_ang_left_knee and max_ang_left_knee are used to store the minimum and maximum angle for the left knee
# min_ang_left_hip and max_ang_left_hip are used to store the minimum and maximum angle for the left hip
min_ang_right_knee, max_ang_right_knee = 0, 0
min_ang_right_hip, max_ang_right_hip = 0, 0
min_ang_left_knee, max_ang_left_knee = 0, 0
min_ang_left_hip, max_ang_left_hip = 0, 0

# squat_stage and display_squat_stage are used to store the stage of the squat
squat_stage, display_squat_stage = None, None

# Push up Variables
# push_up_counter is used to store the count of push-ups
push_up_counter = 0

# push_up_stage and display_push_up_stage are used to store the stage of the push-up
push_up_stage, display_push_up_stage = None, None

# calculate_angle is a function to calculate the angle between three points a, b, and c
def calculate_angle(a, b, c):
    # Calculate the difference between two points a and b and store it in ab
    ab = np.array(a) - np.array(b)
    # Calculate the difference between two points c and b and store it in cb
    cb = np.array(c) - np.array(b)
    # Calculate the difference in radians between the angles of ab and cb
    radians = np.arctan2(cb[1], cb[0]) - np.arctan2(ab[1], ab[0])
    # Convert the radians to degrees
    angle = np.abs(np.rad2deg(radians)) % 360
    # Return the minimum between the angle and 360 minus the angle
    return min(angle, 360 - angle)




## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()
        ret, frame = cap.read()

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
            right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y

            left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y

            # Landmark visibility
            left_shoulder_visibility = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
            left_elbow_visibility = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility
            left_wrist_visibility = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility

            right_shoulder_visibility = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
            right_elbow_visibility = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility
            right_wrist_visibility = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility

            right_hip_visibility = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility
            right_knee_visibility = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility
            right_ankle_visibility = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility

            left_hip_visibility = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility
            left_knee_visibility = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
            left_ankle_visibility = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility

            # Calculate angle for Bicep and Push Up
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            angle_right_knee = calculate_angle(right_hip, right_knee, right_ankle)  # Knee joint angle
            angle_right_knee = round(angle_right_knee, 2)

            angle_left_knee = calculate_angle(left_hip, left_knee, left_ankle)  # Knee joint angle
            angle_left_knee = round(angle_left_knee, 2)

            angle_right_hip = calculate_angle(right_shoulder, right_hip, right_knee)
            angle_right_hip = round(angle_right_hip, 2)

            angle_left_hip = calculate_angle(left_shoulder, left_hip, left_knee)
            angle_left_hip = round(angle_left_hip, 2)

            # right_hip_angle = 180 - angle_right_hip
            # right_knee_angle = 180 - angle_right_knee

            angle_min_right_knee.append(angle_right_knee)
            angle_min_right_hip.append(angle_right_hip)

            angle_min_left_knee.append(angle_left_knee)
            angle_min_left_hip.append(angle_left_hip)

            # Visualize angle for bicep curls
            # print("left_elbow: " + left_elbow)
            # cv2.putText(image, str(left_bicep_angle),
            #             tuple(np.multiply(left_elbow, [720, 720]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #             )
            #
            # cv2.putText(image, str(right_bicep_angle),
            #             tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #             )
            #
            # # # Visualize angle for squats
            # cv2.putText(image, str(angle_right_knee),
            #             tuple(np.multiply(right_knee, [640, 480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #             )
            #
            # cv2.putText(image, str(angle_right_hip),
            #             tuple(np.multiply(right_hip, [640, 480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #             )

            # Curl counter logic
            if (left_angle > 160 and (left_hip_y < 0.6 and right_hip_y < 0.6)
                    and (left_shoulder_visibility > 0.8 and left_elbow_visibility > 0.8
                         and left_wrist_visibility > 0.8)):
                left_bicep_stage = "DOWN"
            if (left_angle < 30 and left_bicep_stage == 'DOWN' and (left_hip_y < 0.6 and right_hip_y < 0.6)
                    and (left_shoulder_visibility > 0.8 and left_elbow_visibility > 0.8
                         and left_wrist_visibility > 0.8)):
                left_bicep_stage = "UP"
                left_bicep_counter += 1
                q.put(str(left_bicep_counter) + " left bicep curl")
                print("left bicep counter: ", left_bicep_counter)

            if (right_angle > 160 and (left_hip_y < 0.6 and right_hip_y < 0.6)
                    and (right_shoulder_visibility > 0.8 and right_elbow_visibility > 0.8
                         and right_wrist_visibility > 0.8)):
                right_bicep_stage = "DOWN"
            if (right_angle < 30 and right_bicep_stage == 'DOWN' and (left_hip_y < 0.6 and right_hip_y < 0.6)
                    and (right_shoulder_visibility > 0.8 and right_elbow_visibility > 0.8
                         and right_wrist_visibility > 0.8)):
                right_bicep_stage = "UP"
                right_bicep_counter += 1
                q.put(str(right_bicep_counter) + " right bicep curl")
                print("right bicep counter: ", right_bicep_counter)

            # Push up counter logic
            if ((left_angle > 160 and right_angle > 160) and (push_up_stage is None)
                    and (left_shoulder_y > 0.5 and right_shoulder_y > 0.5)
                    and (left_shoulder_visibility > 0.5 and right_shoulder_visibility > 0.5
                         and left_elbow_visibility > 0.5 and right_elbow_visibility > 0.5
                         and left_wrist_visibility > 0.5 and right_wrist_visibility > 0.5)):
                push_up_stage = 'UP'
                display_push_up_stage = 'UP'
            if ((left_angle <= 90 and right_angle <= 90)
                    and (left_shoulder_y > 0.5 and right_shoulder_y > 0.5)
                    and push_up_stage == 'UP'
                    and (left_shoulder_visibility > 0.8 and right_shoulder_visibility > 0.8
                         and left_elbow_visibility > 0.8 and right_elbow_visibility > 0.8
                         and left_wrist_visibility > 0.8 and right_wrist_visibility > 0.8)):
                push_up_stage = 'DOWN'
                display_push_up_stage = 'DOWN'
            if ((left_angle > 160 and right_angle > 160)
                    and (left_shoulder_y > 0.5 and right_shoulder_y > 0.5)
                    and push_up_stage == 'DOWN'
                    and (left_shoulder_visibility > 0.8 and right_shoulder_visibility > 0.8
                         and left_elbow_visibility > 0.8 and right_elbow_visibility > 0.8
                         and left_wrist_visibility > 0.8 and right_wrist_visibility > 0.8)):
                push_up_stage = 'UP'
                display_push_up_stage = 'UP'
                push_up_counter += 1
                q.put(str(push_up_counter) + " push up")
                push_up_stage = None

            # Squat counter logic
            print("left_shoulder_y: ", left_shoulder_y)
            print("left_hip_y: ", left_hip_y)
            print("right_shoulder_y: ", right_shoulder_y)
            print("right_hip_y: ", right_hip_y)
            if (((angle_right_knee > 170 or angle_left_knee > 170) and (squat_stage is None)
                 and (left_hip_y < 0.6 and right_hip_y < 0.6)
                 and (left_shoulder_y < 0.4 and right_shoulder_y < 0.4)
                 and ((right_hip_visibility > 0.5 or left_knee_visibility > 0.5)
                      and (right_knee_visibility > 0.5 or left_knee_visibility > 0.5)
                      and (right_ankle_visibility > 0.5 or left_ankle_visibility > 0.5)))):
                squat_stage = "UP"
                display_squat_stage = "UP"
            if (((angle_right_knee <= 90 or angle_left_knee <= 90) and (left_hip_y > 0.6 and right_hip_y > 0.6)
                 and squat_stage == 'UP'
                 and ((right_hip_visibility > 0.5 or left_hip_visibility > 0.5)
                      and (right_knee_visibility > 0.5 or left_knee_visibility > 0.5)
                      and (right_ankle_visibility > 0.5 or left_ankle_visibility > 0.5)))):
                squat_stage = "DOWN"
                display_squat_stage = "DOWN"
            if (((angle_right_knee > 170 or angle_left_knee > 170) and (squat_stage == 'DOWN') and (left_hip_y < 0.6 and right_hip_y < 0.6)
                 and (left_shoulder_y < 0.4 and right_shoulder_y < 0.4)
                 and ((right_hip_visibility > 0.5 or left_knee_visibility > 0.5)
                      and (right_knee_visibility > 0.5 or left_knee_visibility > 0.5)
                      and (right_ankle_visibility > 0.5 or left_ankle_visibility > 0.5)))):
                squat_stage = "UP"
                display_squat_stage = "UP"
                squat_counter += 1
                q.put(str(squat_counter) + " squat")
                print("squat counter: ", squat_counter)
                squat_stage = None

                min_ang_right_knee = min(angle_min_right_knee)
                max_ang_right_knee = max(angle_min_right_knee)

                min_ang_left_knee = min(angle_min_left_knee)
                max_ang_left_knee = max(angle_min_left_knee)

                min_ang_right_hip = min(angle_min_right_hip)
                max_ang_right_hip = max(angle_min_right_hip)

                min_ang_left_hip = min(angle_min_left_hip)
                max_ang_left_hip = max(angle_min_left_hip)

                # print("Right Knee Angle Set: ", min(angle_min_right_knee), " _ ", max(angle_min_right_knee))
                # print("Right Hip Angle Set: ", min(angle_min_right_hip), " _ ", max(angle_min_right_hip))
                #
                # print("Left Knee Angle Set: ", min(angle_min_right_knee), " _ ", max(angle_min_right_knee))
                # print("Left Hip Angle Set: ", min(angle_min_right_hip), " _ ", max(angle_min_right_hip))

                angle_min_right_knee = []
                angle_min_right_hip = []

                angle_min_left_knee = []
                angle_min_left_hip = []

        except:
            pass

        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0, 0), (2000, 80), (245, 117, 16), -1)

        # Rep data
        cv2.putText(image, 'BICEP LEFT REPS: ' + str(left_bicep_counter), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, ' BICEP RIGHT REPS: ' + str(right_bicep_counter), (350, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, "SQUAT REPS : " + str(squat_counter),
                    (750, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, "PUSHUP REPS : " + str(push_up_counter),
                    (1100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Stage data
        cv2.putText(image, 'POSITION: ' + str(left_bicep_stage), (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, 'POSITION: ' + str(right_bicep_stage), (365, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, 'POSITION: ' + str(display_squat_stage), (750, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, 'POSITION: ' + str(display_push_up_stage), (1100, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # # Knee angle:
        # cv2.putText(image, "Knee-joint angle : " + str(min_ang),
        #             (30, 100),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        #
        # # Hip angle:
        # cv2.putText(image, "Hip-joint angle : " + str(min_ang_hip),
        #             (30, 140),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow(window_name, image)
        # write the frame to the output file
        output.write(image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    output.release()
    cv2.destroyAllWindows()
