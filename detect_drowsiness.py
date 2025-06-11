import cv2
import dlib
from scipy.spatial import distance
from imutils import face_utils
import pygame
import time

# EAR calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# MAR calculation
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[13], mouth[19])  # 63-67
    B = distance.euclidean(mouth[14], mouth[18])  # 64-66
    C = distance.euclidean(mouth[12], mouth[16])  # 62-66
    return (A + B) / (2.0 * C)

# Thresholds
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20
MOUTH_AR_THRESH = 0.75
MOUTH_AR_CONSEC_FRAMES = 15

# Counters
COUNTER_EYE = 0
COUNTER_YAWN = 0
ALARM_ON = False

# Initialize sound
pygame.mixer.init()
pygame.mixer.music.load("aleart.aiff")

# Load models
print("[INFO] Loading predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

print("[INFO] Starting webcam...")
vs = cv2.VideoCapture(0)
time.sleep(1.0)

while True:
    ret, frame = vs.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # Draw eye and mouth
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255, 0, 0), 1)

        eyes_closed = False
        is_yawning = False

        # Eye Drowsiness Detection
        if ear < EYE_AR_THRESH:
            COUNTER_EYE += 1
            eyes_closed = True
        else:
            COUNTER_EYE = 0

        # Yawn Detection
        if mar > MOUTH_AR_THRESH:
            COUNTER_YAWN += 1
            is_yawning = True
        else:
            COUNTER_YAWN = 0

        # Trigger Alarm if either condition is true
        if COUNTER_EYE >= EYE_AR_CONSEC_FRAMES or COUNTER_YAWN >= MOUTH_AR_CONSEC_FRAMES:
            if not ALARM_ON:
                ALARM_ON = True
                pygame.mixer.music.play(-1)

            if COUNTER_EYE >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if COUNTER_YAWN >= MOUTH_AR_CONSEC_FRAMES:
                cv2.putText(frame, "YAWNING ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if ALARM_ON:
                pygame.mixer.music.stop()
                ALARM_ON = False

        # Print current eye/yawn status
        if eyes_closed:
            cv2.putText(frame, "Eyes Closed", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if is_yawning:
            cv2.putText(frame, "Yawning", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Show EAR & MAR values
        cv2.putText(frame, f"EAR: {ear:.2f}", (500, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"MAR: {mar:.2f}", (500, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Driver Drowsiness + Yawn Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()