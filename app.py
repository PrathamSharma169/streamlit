import streamlit as st
import cv2
import numpy as np
import time
from collections import deque
import requests
import tempfile

st.title("Stampede Detection")

# Create a session state flag for stopping detection
if "stop_detection" not in st.session_state:
    st.session_state["stop_detection"] = False

# Button placed outside the loop to set the flag
if st.button("Stop Detection", key="stop_detection_button"):
    st.session_state["stop_detection"] = True

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    # Save the uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error: Could not open video file {video_path}")
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    MIN_SUSTAINED_FRAMES = int(fps * 1.5)
    MOTION_SENSITIVITY = 0.4

    frame_count = 0
    prev_gray = None
    in_stampede = False
    stampede_count = 0
    stampede_start_time = None
    detection_queue = deque(maxlen=MIN_SUSTAINED_FRAMES * 2)

    st.write("Starting HIGHLY SENSITIVE stampede detection...")
    st.write(f"Only {MIN_SUSTAINED_FRAMES} frames ({MIN_SUSTAINED_FRAMES/fps:.1f} seconds) needed to trigger alert")

    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check the stop flag set outside the loop.
        if st.session_state["stop_detection"]:
            break

        frame_count += 1
        frame_display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.1, flags=0
            )

            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            avg_motion = np.mean(magnitude)

            angles = np.arctan2(flow[..., 1], flow[..., 0])
            hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
            max_direction = np.max(hist) / (np.sum(hist) + 1e-5)

            is_stampede_frame = avg_motion > MOTION_SENSITIVITY and max_direction > 0.3

            detection_queue.append(1 if is_stampede_frame else 0)
            consecutive_detections = 0
            for i in range(len(detection_queue) - 1, -1, -1):
                if detection_queue[i] == 1:
                    consecutive_detections += 1
                else:
                    if consecutive_detections < 3:
                        consecutive_detections = 0
                    else:
                        break

            if consecutive_detections >= MIN_SUSTAINED_FRAMES * 0.6 and not in_stampede:
                in_stampede = True
                stampede_start_time = time.time()
                stampede_count += 1
                st.write(f"*** STAMPEDE DETECTED! ({consecutive_detections} frames) ***")
                st.write(f"Motion: {avg_motion:.2f}, Direction: {max_direction:.2f}")

                try:
                    alert_payload = {
                        "alert": "Stampede detected",
                        "frames": consecutive_detections,
                        "motion": float(avg_motion),        # Convert numpy.float32 to float
                        "direction": float(max_direction),  # Convert numpy.float32 to float
                        "timestamp": time.time()
                    }
                    response = requests.post("http://localhost:5000/api/alert", json=alert_payload)
                    if response.ok:
                        st.write("Backend alert sent successfully.")
                    else:
                        st.write("Failed to send backend alert.", response.status_code)
                except Exception as e:
                    st.write("Error sending backend alert:", e)

            elif consecutive_detections < MIN_SUSTAINED_FRAMES * 0.3 and in_stampede:
                in_stampede = False
                duration = time.time() - stampede_start_time if stampede_start_time else 0
                st.write(f"Stampede ended after {duration:.1f} seconds")

            if in_stampede:
                duration = time.time() - stampede_start_time if stampede_start_time else 0
                status_text = f"STAMPEDE DETECTED! Duration: {duration:.1f}s"
                color = (0, 0, 255)  # Red
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
                    status_text = "Normal Activity"
                    color = (0, 255, 0)

            cv2.putText(frame_display, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            metrics_text = f"Motion: {avg_motion:.2f}, Direction: {max_direction:.2f}, Frames: {consecutive_detections}/{MIN_SUSTAINED_FRAMES}"
            cv2.putText(frame_display, metrics_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        prev_gray = gray.copy()
        frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
        time.sleep(1 / fps)

    cap.release()
    st.write(f"Detection completed. Found {stampede_count} stampede events.")