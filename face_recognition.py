import threading
import cv2
from deepface import DeepFace
from queue import Queue

# Initialize video capture
cap = cv2.VideoCapture(1)  # Use the second camera (change index if needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution if needed 320 and 240
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load reference images
reference_img1 = cv2.imread('pic3.png')


# Queue for processing frames
frame_queue = Queue(maxsize=1)
face_match = False

# Function to process frames
def process_frame():
    global face_match
    while True:
        frame = frame_queue.get()
        try:
            face_match = DeepFace.verify(frame, reference_img1)["verified"]
        except:
            face_match = False

# Start processing thread
threading.Thread(target=process_frame, daemon=True).start()

# Main loop for video capture
counter = 0
while True:
    ret, frame = cap.read()
    if ret:
        # Add frame to queue every 60 frames
        if counter % 60 == 0 and not frame_queue.full():
            frame_queue.put(frame.copy())

        # Display result on the video
        if face_match:
            cv2.putText(frame, "PERFECT MATCH COMRADE", (20, 450),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "Nope, NOT A MATCH", (20, 450),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

        cv2.imshow("MIKE SOLOMAN'S FACE RECOGNITION APP", frame)
        counter += 1

    # Break on 'q' key press
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()