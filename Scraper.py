import cv2
import os
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize MediaPipe Drawing Utilities
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)  # 0 is the default camera

count = 0  # Counter for image filenames
categories = ['A', 'B','C','D','E','ily']  # List of categories
category_index = 0  # Index of the current category

while cap.isOpened():
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break

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
        print("Hand detected")  # Debugging line
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the bounding box coordinates
            x_min = max(0, int(min([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1] - 50))  # Subtract 50 pixels
            x_max = min(frame.shape[1], int(max([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1] + 50))  # Add 50 pixels
            y_min = max(0, int(min([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0] - 50))  # Subtract 50 pixels
            y_max = min(frame.shape[0], int(max([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0] + 50))  # Add 50 pixels

            # Crop the image to the bounding box of the hand and save it
            cropped_image = frame[y_min:y_max, x_min:x_max]

            # Create a directory for the category if it doesn't exist
            os.makedirs(f'training_images/{categories[category_index]}', exist_ok=True)

            # Save the image in the category directory
            cv2.imwrite(f'training_images/{categories[category_index]}/image_{count}.jpg', cropped_image)
            print(f'Image saved: training_images/{categories[category_index]}/image_{count}.jpg')  # Debugging line
            count += 1

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # If 300 images have been saved, switch to the next category
    if count % 300 == 0 and count != 0:
        category_index += 1
        if category_index >= len(categories):
            break
        print("Switching to category:", categories[category_index])
        print("Current time:", time.ctime())
        time.sleep(2)  # Pause for 2 seconds

# After the loop release the cap object and destroy windows
cap.release()
cv2.destroyAllWindows()