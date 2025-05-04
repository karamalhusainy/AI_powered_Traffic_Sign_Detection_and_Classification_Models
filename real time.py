import cv2
import numpy as np
from tensorflow import keras
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Load the pre-trained CNN model
model = keras.models.load_model('traffic_classifier1.h5')

threshold = 0.85
font = cv2.FONT_HERSHEY_SIMPLEX
# Define the class labels
class_labels = ["No Entry", "Hump", "Stop", "Pedestrian Cross", "No Stop", "Give Way", "Pass Either"]

# Initialize video capture
cap = cv2.VideoCapture(0)



while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Preprocess the frame
    if not ret:
        continue  # Skip to the next iteration if the frame was not read properly

    resized = cv2.resize(frame, (30, 30))
    resized = np.expand_dims(resized, axis=0)

    # Perform traffic sign classification
    prediction = model.predict(resized) 
    class_index = np.argmax(prediction)
    class_label = class_labels[class_index]

    probabilityValue = np.amax(prediction)
    if probabilityValue > threshold:
        cv2.putText(frame, str(class_label), (120, 35), font, 0.75, (0, 0, 255),
                    2,
                    cv2.LINE_AA)
        cv2.putText(frame, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2,
                    cv2.LINE_AA)


    cv2.imshow("Result", frame)

    # Check for key press (e.g., 'q' to exit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()




