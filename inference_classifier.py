import pickle
import mediapipe as mp
import numpy as np
import pygame
import streamlit as st

# Initialize pygame
pygame.init()

# Load the MP3 file
pygame.mixer.music.load(r"danger-alarm-23793.mp3")

# Load the model
model_dict = pickle.load(open(r"model_xgboost.p", 'rb'))
model = model_dict['model']

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary for labels
labels_dict = {2: 'C'}  # Only 'C' gesture for system disarm

# Variables for system status
system_armed = True
disarm_timer = 0

# Variable to keep track of whether the sound is playing
sound_playing = False

# Streamlit app function
def main():
    st.title("Sign Language Detector")
    st.write("Showing live sign language detection")

    # Video capture
    cap = cv2.VideoCapture(0)

    # Infinite loop to capture video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video.")
            break

        data_aux = []
        x_ = []
        y_ = []

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Calculate bounding box
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Predict gesture
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict.get(int(prediction[0]), None)

            # Display the predicted character and disarm if 'C' is detected
            if predicted_character == 'C':
                disarm_timer += 5
                st.write("Danger")
                if disarm_timer > 30:  # Disarm for 30 frames
                    system_armed = False
                    disarm_timer = 0
                    if not sound_playing:
                        pygame.mixer.music.play(-1)  # Loop the music
                        sound_playing = True
            else:
                disarm_timer = 0
                if system_armed:
                    pass  # Do nothing
                else:
                    st.write("")  # Display nothing
                    if sound_playing:
                        pygame.mixer.music.stop()
                        sound_playing = False

        # Display the frame
        st.image(frame, channels="BGR", use_column_width=True)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the Streamlit app
if __name__ == "__main__":
    main()

