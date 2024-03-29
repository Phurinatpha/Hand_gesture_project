import cv2
import mediapipe as mp
import pyautogui
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize hand tracking
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)

# Set up camera capture
cap = cv2.VideoCapture(0)

# Previous hand landmarks
prev_landmarks = None

# Cooldown timer
cooldown_timer = time.time()
cooldown_duration = 0.2  # Adjust the cooldown duration as needed

movement_threshold = 0.03

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
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            # middle_tip = landmarks[12]
            # ring_tip = landmarks[16]
            # pinky_tip = landmarks[20]

            # Detect hand movement
            if prev_landmarks is not None:
                # Calculate differences in x and y coordinates
                dx = index_tip.x - prev_landmarks[8].x
                dy = index_tip.y - prev_landmarks[8].y

                # Check if cooldown period has elapsed
                if time.time() - cooldown_timer > cooldown_duration:
                    # Move left right
                    if dx < -movement_threshold:
                        pyautogui.press('right')
                        cooldown_timer = time.time()  
                    elif dx > movement_threshold:
                        pyautogui.press('left')
                        cooldown_timer = time.time()  

                    # Move up down
                    elif dy < -movement_threshold:
                        pyautogui.press('up')
                        cooldown_timer = time.time() 
                    elif dy > movement_threshold:
                        pyautogui.press('down')
                        cooldown_timer = time.time()
                
            # Update previous landmarks
            prev_landmarks = hand_landmarks.landmark

    # Display the frame
    cv2.imshow('Hand Tracking', cv2.flip(frame, 1))

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
