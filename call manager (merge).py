import cv2
import mediapipe as mp
import speech_recognition as sr
import threading
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Gesture and Voice Command Labels
ANSWER_GESTURE = "Answer"
DECLINE_GESTURE = "Decline"
ANSWER_COMMANDS = ["answer", "pick up", "yes", "accept the call"]
DECLINE_COMMANDS = ["decline", "reject", "no", "hang up"]

# Shared State
call_action = None  # Tracks whether the call is answered or declined

# Function to classify hand gestures
def classify_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    # Example gestures: Thumb up -> Answer, Index up -> Decline
    if thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y:
        return ANSWER_GESTURE
    elif index_tip.y < thumb_tip.y and index_tip.y < middle_tip.y:
        return DECLINE_GESTURE
    else:
        return "Unknown"

# Function to classify voice commands
def classify_command(command):
    command = command.lower()
    if any(phrase in command for phrase in ANSWER_COMMANDS):
        return "Answer"
    elif any(phrase in command for phrase in DECLINE_COMMANDS):
        return "Decline"
    else:
        return "Unknown"

# Thread 1: Hand Gesture Recognition
def gesture_recognition():
    global call_action
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Flip and process the frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Classify the gesture
                gesture = classify_gesture(hand_landmarks)
                if gesture in [ANSWER_GESTURE, DECLINE_GESTURE]:
                    call_action = gesture
                    print(f"Gesture Detected: {gesture}")
                    break

        # Display the frame
        cv2.imshow("Gesture Recognition", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Thread 2: Voice Command Recognition
def voice_recognition():
    global call_action
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    while True:
        print("Listening for a voice command (say 'Answer' or 'Decline')...")
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio)
                print(f"Voice Command: {command}")
                action = classify_command(command)
                if action in ["Answer", "Decline"]:
                    call_action = action
                    print(f"Voice Command Detected: {action}")
                    break
            except sr.UnknownValueError:
                print("Sorry, I couldn't understand that.")
            except sr.RequestError:
                print("Error: Could not connect to the speech recognition service.")
            except sr.WaitTimeoutError:
                print("No voice input detected.")

# Main Function to Handle Calls
def handle_call():
    global call_action
    while True:
        if call_action == "Answer":
            print("Call Answered!")
            call_action = None
            break
        elif call_action == "Decline":
            print("Call Declined!")
            call_action = None
            break
        time.sleep(0.1)

# Run Gesture and Voice Recognition in Parallel
if __name__ == "__main__":
    # Start threads for gesture and voice recognition
    gesture_thread = threading.Thread(target=gesture_recognition)
    voice_thread = threading.Thread(target=voice_recognition)

    gesture_thread.start()
    voice_thread.start()

    # Monitor for call action
    handle_call()

    # Wait for threads to complete
    gesture_thread.join()
    voice_thread.join()

    print("Call handling system terminated.")
