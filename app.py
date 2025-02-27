import cv2
import numpy as np
import time
from collections import deque
import streamlit as st

# Streamlit app title
st.title("Crowd Analysis and Stampede Detection")

# Sidebar inputs for user settings
st.sidebar.header("Settings")
motion_sensitivity = st.sidebar.slider("Motion Sensitivity", 0.1, 1.0, 0.4, 0.1)
min_sustained_frames = st.sidebar.slider("Minimum Sustained Frames", 10, 200, 45, 5)
skip_frames = st.sidebar.slider("Skip Frames (1 = no skip)", 1, 10, 1, 1)

# Upload video file
uploaded_file = st.file_uploader("Upload a video file (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

# Function to process video frames
def process_video(video_path, motion_sensitivity, min_sustained_frames, skip_frames):
    cap = cv2.VideoCapture(video_path)
    
    # Get video FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default FPS if unavailable
    frame_delay = 1 / fps  # Delay in seconds

    # Parameters
    MIN_SUSTAINED_FRAMES = int(fps * (min_sustained_frames / 30))
    MOTION_SENSITIVITY = motion_sensitivity

    # Variables for tracking
    frame_count = 0
    prev_gray = None
    in_stampede = False
    stampede_count = 0
    stampede_start_time = None
    detection_queue = deque(maxlen=MIN_SUSTAINED_FRAMES * 2)

    # Placeholder for displaying frames
    stframe = st.empty()

    while cap.isOpened():
        start_time = time.time()  # Start timing the frame processing

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames based on user setting
        if frame_count % skip_frames != 0:
            continue

        frame_display = frame.copy()

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            # Optical flow calculation
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=1, winsize=15,  # Reduce levels for faster computation
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0
            )
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            avg_motion = np.mean(magnitude)

            angles = np.arctan2(flow[..., 1], flow[..., 0])
            hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
            max_direction = np.max(hist) / (np.sum(hist) + 1e-5)

            # Stampede detection logic
            is_stampede_frame = avg_motion > MOTION_SENSITIVITY and max_direction > 0.3
            detection_queue.append(1 if is_stampede_frame else 0)

            consecutive_detections = sum(detection_queue)

            if consecutive_detections >= MIN_SUSTAINED_FRAMES * 0.6 and not in_stampede:
                in_stampede = True
                stampede_start_time = time.time()
                stampede_count += 1

            elif consecutive_detections < MIN_SUSTAINED_FRAMES * 0.3 and in_stampede:
                in_stampede = False

            # Display status
            if in_stampede:
                duration = time.time() - stampede_start_time
                status_text = f"STAMPEDE DETECTED! Duration: {duration:.1f}s"
                color = (0, 0, 255)
            else:
                detection_ratio = consecutive_detections / MIN_SUSTAINED_FRAMES
                if detection_ratio > 0.8:
                    status_text = f"IMMINENT STAMPEDE WARNING! ({detection_ratio*100:.0f}%)"
                    color = (0, 80, 255)
                elif detection_ratio > 0.5:
                    status_text = f"SUSPICIOUS ACTIVITY ({detection_ratio*100:.0f}%)"
                    color = (0, 165, 255)
                elif detection_ratio > 0.3:
                    status_text = f"Heightened Movement ({detection_ratio*100:.0f}%)"
                    color = (0, 255, 255)
                else:
                    status_text = f"Normal Activity"
                    color = (0, 255, 0)

            # Add status text to frame
            cv2.putText(frame_display, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        prev_gray = gray.copy()

        # Convert frame to RGB for Streamlit display
        frame_display = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        stframe.image(frame_display, channels="RGB", use_container_width=True)

        # Measure processing time and adjust delay
        processing_time = time.time() - start_time
        delay = max(0, frame_delay - processing_time)  # Ensure no negative delay
        time.sleep(delay)

    cap.release()
    return stampede_count

# If a video is uploaded, process it
if uploaded_file is not None:
    with st.spinner("Processing video..."):
        # Save uploaded video to a temporary file
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        video_path = "temp_video.mp4"

        # Call the processing function
        stampede_count = process_video(video_path, motion_sensitivity, min_sustained_frames, skip_frames)

    st.success(f"Detection completed. Found {stampede_count} stampede events.")
else:
    st.info("Please upload a video file to start detection.")