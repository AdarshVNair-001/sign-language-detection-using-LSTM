import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert color space
    image.flags.writeable = False  # Reduce computation
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Restore write permissions
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to OpenCV format
    return image, results

# Function to draw landmarks on the image
def draw_landmarks(image, results):
    # Draw face landmarks
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),  # Green color
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)   # Green color
    )
    # Draw pose landmarks
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),  # Green color
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)   # Green color
    )
    # Draw left hand landmarks
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),  # Green color
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)   # Green color
    )
    # Draw right hand landmarks
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),  # Green color
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)   # Green color
    )

# Function to extract keypoints from Mediapipe results
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    
    keypoints = np.concatenate([pose, face, lh, rh])
    
    if keypoints.shape[0] != 1662:
        print(f"Warning: Keypoint shape mismatch! Got {keypoints.shape[0]}, expected 1662.")
    
    return keypoints

# Function to visualize prediction probabilities
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# Initialize variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

# Open webcam
cap = cv2.VideoCapture(0)

# Load Mediapipe holistic model for real-time detection
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Run Mediapipe detection
        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks on image
        draw_landmarks(image, results)

        # Extract keypoints **(DO NOT SAVE THEM)**
        keypoints = extract_keypoints(results)
        if keypoints.shape[0] != 1662:
            print("Skipping frame: Incorrect keypoint shape")
            continue  # Skip frames with incorrect data

        # Append keypoints to sequence for prediction
        sequence.append(keypoints)
        sequence = sequence[-30:]  # Keep last 30 frames

        # Predict when we have enough frames
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predicted_action = actions[np.argmax(res)]
            print(f"Detected Action: {predicted_action}")

            predictions.append(np.argmax(res))

            # Ensure enough predictions before deciding
            if len(predictions) >= 10:
                if np.unique(predictions[-10:])[0] == np.argmax(res):  
                    if res[np.argmax(res)] > threshold:  
                        if len(sentence) == 0 or predicted_action != sentence[-1]:
                            sentence.append(predicted_action)

            # Keep only the last 5 recognized actions
            if len(sentence) > 5:
                sentence = sentence[-5:]

            

        # Show recognized gesture
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display output
        cv2.imshow('OpenCV Feed', image)

        # Break if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()


