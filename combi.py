import cv2
import mediapipe as mp
import pyautogui
import time
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize hand tracking
hands = mp_hands.Hands(max_num_hands=1)

# Set up camera capture
cap = cv2.VideoCapture(0)

# Previous hand landmarks
prev_landmarks = None

# Cooldown timer
cooldown_timer = time.time()
cooldown_duration = 0.5  # Adjust the cooldown duration as needed

# Threshold distance for further movement
threshold_distance = 50  # Adjust the threshold distance as needed

# Flag to indicate if the hand has moved far enough
hand_moved = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(frame_rgb)

    # If hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark positions
            landmarks = hand_landmarks.landmark

            # Get coordinates of specific landmarks
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            
            # Calculate Euclidean distance between index and middle finger tips
            distance = math.sqrt((index_tip.x - middle_tip.x) ** 2 + (index_tip.y - middle_tip.y) ** 2)

            # Detect hand movement and trigger key press
            if prev_landmarks is not None:
                dx = index_tip.x - prev_landmarks[8].x
                dy = index_tip.y - prev_landmarks[8].y

                # Check if cooldown period has elapsed
                if time.time() - cooldown_timer > cooldown_duration:
                    if distance > threshold_distance:
                        # Move left
                        if dx < -0.05:
                            pyautogui.press('left')
                            cooldown_timer = time.time()  # Reset cooldown timer
                        # Move right
                        elif dx > 0.05:
                            pyautogui.press('right')
                            cooldown_timer = time.time()  # Reset cooldown timer
                        # Move up
                        elif dy < -0.05:
                            pyautogui.press('up')
                            cooldown_timer = time.time()  # Reset cooldown timer
                        # Move down
                        elif dy > 0.05:
                            pyautogui.press('down')
                            cooldown_timer = time.time()  # Reset cooldown timer
                        hand_moved = False
                    else:
                        hand_moved = True

            # Update previous landmarks
            prev_landmarks = hand_landmarks.landmark

    # If hand was in near point, but hasn't moved far enough, reset cooldown timer
    if hand_moved:
        cooldown_timer = time.time()

    # Display the frame
    cv2.imshow('Hand Tracking', cv2.flip(frame, 1))

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
