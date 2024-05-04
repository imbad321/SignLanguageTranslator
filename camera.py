import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Check if a GPU is available and if not, fallback to CPU
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print('GPU device not found, using CPU instead.')
else:
  print('Found GPU at: {}'.format(device_name))

with tf.device(device_name):
    # Load the trained model
    model = tf.keras.models.load_model('hand_gesture_model.h5')

    # Define the list of class names in the same order as used during training
    class_names = ['A', 'B','C','D','E','ily']

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    # Initialize MediaPipe Drawing Utilities
    mp_drawing = mp.solutions.drawing_utils

    # Capture video from the webcam
    cap = cv2.VideoCapture(1)

    frame_count = 0  # Add a frame counter

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1  # Increment the frame counter
        original_frame = frame.copy()
        # Only process every 10th frame
        if frame_count != 0:
            # Create a copy of the original frame

            # Flip the image horizontally for a later selfie-view display
            # and convert the BGR image to RGB.
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            frame.flags.writeable = False
            results = hands.process(frame)

            # Draw the hand annotations on the image.
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:  # If a hand is detected
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get the bounding box coordinates
                    x_min = max(0, int(min([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1] - 50))  # Subtract 50 pixels
                    x_max = min(frame.shape[1], int(max([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1] + 50))  # Add 50 pixels
                    y_min = max(0, int(min([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0] - 50))  # Subtract 50 pixels
                    y_max = min(frame.shape[0], int(max([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0] + 50))  # Add 50 pixels

                    # Crop the image to the bounding box of the hand
                    cropped_image = frame[y_min:y_max, x_min:x_max]

                    # Preprocess the cropped image to match the input shape used during training
                    img = cv2.resize(cropped_image, (150, 150))
                    img = img / 255.0  # normalize pixel values
                    img = np.expand_dims(img, axis=0)  # expand dimensions for model prediction

                    # Use the model to predict the class of the hand gesture
                    predictions = model.predict(img)
                    predicted_class = class_names[np.argmax(predictions)]
                    prediction_probability = np.max(predictions)  # Get the maximum probability

                    # Display the prediction and probability on the original frame
                    cv2.putText(original_frame, f'{predicted_class}: {prediction_probability*100:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Show the cropped image in a separate window
                    cv2.imshow('Prediction Capture', cropped_image)

        # Show the original frame
        cv2.imshow('Original Camera Feed', original_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object and destroy windows
    cap.release()
    cv2.destroyAllWindows()