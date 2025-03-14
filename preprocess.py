from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

# Define the actions to be detected
try:

    actions = ['hello', 'thanks', 'iloveyou']
    DATA_PATH = "MP_Data"

except ImportError:
    print("Error: Could not import actions. Ensure both files are in the same directory.")
    exit()

# Define the label map
label_map = {label: num for num, label in enumerate(actions)}

# Initialize sequences and labels
sequences, labels = [], []

# Define the sequence length
sequence_length = 30

# Check if DATA_PATH exists
if not os.path.exists(DATA_PATH):
    print(f"Error: DATA_PATH '{DATA_PATH}' does not exist. Run slt3.py first.")
    exit()

# Loop through each action and sequence to load the data
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        print(f"Warning: No data found for action '{action}'")
        continue

    try:
        sequences_list = np.array(os.listdir(action_path)).astype(int)
        if sequences_list.size == 0:
            print(f"Warning: No sequences found for action '{action}'")
            continue
    except ValueError:
        print(f"Error: Invalid sequence names in '{action_path}'. Ensure they are numerical.")
        continue

    for sequence in sequences_list:
        sequence_path = os.path.join(action_path, str(sequence))
        
        if not os.path.exists(sequence_path):
            print(f"Warning: Missing sequence folder '{sequence_path}'")
            continue

        window = []
        for frame_num in range(sequence_length):
            frame_path = os.path.join(sequence_path, f"{frame_num}.npy")
            
            if not os.path.exists(frame_path):
                print(f"Warning: Missing frame '{frame_path}', skipping sequence {sequence}")
                break  # Skip this sequence
            
            res = np.load(frame_path)
            window.append(res)

        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])

# Convert sequences and labels to numpy arrays
if len(sequences) == 0:
    print("Error: No valid data found. Check if slt3.py was run correctly.")
    exit()

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y = to_categorical(labels).astype(int)

