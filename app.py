import ultralytics
from ultralytics import YOLO
# import queue
import cv2
import streamlit as st
import av
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# Classes
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Confidence threshold
conf_threshold = 0.3

# Set initial values for the frame width and height
frame_width = 640
frame_height = 480

# Set up the Streamlit application
st.set_page_config(
    page_title='Human Emotion Detector',
    page_icon=':smiley:',
    layout='centered'
)

# Load the model
@st.cache_resource
def load_model():
    return YOLO('weights/emotion.pt')

model = load_model()

# Set title
st.title('Human Emotion Detector')


# Define a callback function to process each frame
def callback(frame: av.VideoFrame) -> av.VideoFrame:

    # Resize the frame to the desired width and height
    resized_frame = cv2.resize(frame.to_ndarray(format="bgr24"), (frame_width, frame_height))

    # Detect objects in the current frame using a custom-trained model
    detections = model(resized_frame, stream=True)

    # Draw bounding boxes and labels for each detected object
    for detection in detections:
        bounding_boxes = detection.boxes

        for box in bounding_boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = box.conf[0]
            cls = int(box.cls[0])

            # If object confidence is above the threshold, draw bounding box and label
            if conf > conf_threshold:
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                text_size, _ = cv2.getTextSize(f'{classes[cls]} {conf:.2f}', cv2.FONT_HERSHEY_PLAIN, 1, 2)
                text_width, text_height = text_size

                if y1 - 30 < 0:
                    y1 = 30
                if x1 + text_width + 5 > resized_frame.shape[1]:
                    x1 = resized_frame.shape[1] - text_width - 5
                if y1 - text_height - 10 < 0:
                    y1 = text_height + 10

                cv2.rectangle(resized_frame, (x1, y1 - 30), (x1 + text_width + 5, y1 - 5), (0, 255, 0), -1)

                cv2.putText(resized_frame, f'{classes[cls]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

    # Return the processed frame
    return av.VideoFrame.from_ndarray(resized_frame, format="bgr24")

# Add sliders to adjust the frame width and height
frame_width = st.slider('Frame Width', 320, 1280, 640, 10)
frame_height = st.slider('Frame Height', 240, 960, 480, 10)

# Stream the video using WebRTC
webrtc_streamer(
    key="human_emotion_detection",
    video_frame_callback=callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False}

)


