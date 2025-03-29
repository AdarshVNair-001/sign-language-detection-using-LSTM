import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import threading
from tensorflow.keras.models import load_model

# Load the trained model and actions
model = load_model('model.h5')
actions = np.load('actions.npy', allow_pickle=True)

# Initialize Mediapipe holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Colors for visualization
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

# Function to detect landmarks using Mediapipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to draw landmarks on the image
def draw_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# Function to extract keypoints from Mediapipe results
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

# Optimized Video Capture Class
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
    
    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.stream.read()
    
    def read(self):
        return self.frame
    
    def stop(self):
        self.stopped = True
        self.stream.release()

# Initialize variables
sequence, sentence, predictions = [], [], []
threshold, frame_skip, frame_count = 0.5, 3, 0

# Start video capture
cap = VideoStream().start()

with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
    while True:
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        
        frame = cap.read()
        if frame is None:
            continue
        
        # Resize frame for speed optimization
        frame = cv2.resize(frame, (320, 240))
        
        # Run Mediapipe detection
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)
        
        keypoints = extract_keypoints(results)
        if keypoints.shape[0] != 1662:
            continue
        
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predicted_action = actions[np.argmax(res)]
            predictions.append(np.argmax(res))
            
            if len(predictions) >= 10 and np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold and (not sentence or predicted_action != sentence[-1]):
                    sentence.append(predicted_action)
            
            if len(sentence) > 5:
                sentence = sentence[-5:]
        
        # Display detected action
        cv2.rectangle(image, (0, 0), (320, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Optimized Detection', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.stop()
cv2.destroyAllWindows()