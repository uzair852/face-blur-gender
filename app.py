import streamlit as st
import cv2
import tempfile
import os
import subprocess
from deepface import DeepFace

# ---- CONFIG ----
DETECT_EVERY_N_FRAMES = 4
BLUR_STRENGTH = 51
EXPAND_RATIO = 0.5
DETECTION_SCALE = 0.5

_windows_ffmpeg = r"C:\Users\uzair\Downloads\ffmpeg-9.0.1-essentials_build\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
FFMPEG_PATH = _windows_ffmpeg if os.path.exists(_windows_ffmpeg) else "ffmpeg"


def blur_region(frame, x, y, w, h, expand_ratio=EXPAND_RATIO):
    frame_h, frame_w = frame.shape[:2]
    expand_w = int(w * expand_ratio)
    expand_h_top = int(h * expand_ratio * 1.5)
    expand_h_bottom = int(h * expand_ratio * 0.5)

    x1 = max(0, x - expand_w)
    y1 = max(0, y - expand_h_top)
    x2 = min(frame_w, x + w + expand_w)
    y2 = min(frame_h, y + h + expand_h_bottom)

    face_region = frame[y1:y2, x1:x2]
    if face_region.size == 0:
        return frame
    blurred = cv2.GaussianBlur(face_region, (BLUR_STRENGTH, BLUR_STRENGTH), 0)
    frame[y1:y2, x1:x2] = blurred
    return frame


def detect_and_classify(frame):
    results = []
    small_frame = cv2.resize(frame, None, fx=DETECTION_SCALE, fy=DETECTION_SCALE)
    try:
        analysis = DeepFace.analyze(
            img_path=small_frame,
            actions=['gender'],
            detector_backend='retinaface',
            enforce_detection=False
        )
        if isinstance(analysis, dict):
            analysis = [analysis]
        for face in analysis:
            region = face['region']
            if region['w'] <= 0 or region['h'] <= 0:
                continue
            x = int(region['x'] / DETECTION_SCALE)
            y = int(region['y'] / DETECTION_SCALE)
            w = int(region['w'] / DETECTION_SCALE)
            h = int(region['h'] / DETECTION_SCALE)
            dominant_gender = face['dominant_gender']
            confidence = face['gender'][dominant_gender]
            print(f"Detected: {dominant_gender} (confidence: {confidence:.1f}%)")
            results.append((x, y, w, h, dominant_gender))
    except Exception as e:
        print(f"Detection error on frame: {e}")
    return results


def process_video(input_path, output_path, progress_callback=None):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    last_detections = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % DETECT_EVERY_N_FRAMES == 0:
            new_detections = detect_and_classify(frame)
            if new_detections:
                last_detections = new_detections

        for (x, y, w, h, gender) in last_detections:
            if gender == "Woman":
                frame = blur_region(frame, x, y, w, h)

        out.write(frame)
        frame_idx += 1

        if progress_callback and total_frames > 0:
            progress_callback(frame_idx / total_frames)

    cap.release()
    out.release()


def merge_audio(video_path, original_path, final_path):
    cmd = [
        FFMPEG_PATH, "-y",
        "-i", video_path,
        "-i", original_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        final_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)


# ---- STREAMLIT UI ----
st.set_page_config(
    page_title="Face Blur by Gender",
    page_icon="🎭",
    layout="centered"
)

st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #6366f1;
        color: white;
        font-weight: 600;
        padding: 0.6rem;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #4f46e5;
        color: white;
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #6366f1;
        border-radius: 10px;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🎭 Face Blur by Gender</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: gray;'>Upload a video and automatically blur faces classified as women, powered by DeepFace + OpenCV.</p>",
    unsafe_allow_html=True
)

st.divider()

uploaded_file = st.file_uploader("📤 Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = os.path.join(tmp_dir, "input.mp4")
        blurred_path = os.path.join(tmp_dir, "blurred.mp4")
        final_path = os.path.join(tmp_dir, "final.mp4")

        with open(input_path, "wb") as f:
            f.write(uploaded_file.read())

        st.markdown("#### Original Video")
        st.video(input_path)

        st.markdown("")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            process_clicked = st.button("✨ Process Video")

        if process_clicked:
            st.markdown("#### Processing")
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(pct):
                progress_bar.progress(min(pct, 1.0))
                status_text.markdown(f"🔍 Detecting faces and blurring... **{int(pct * 100)}%**")

            status_text.markdown("🔍 Detecting faces and blurring...")
            process_video(input_path, blurred_path, progress_callback=update_progress)

            status_text.markdown("🎧 Merging audio...")
            try:
                merge_audio(blurred_path, input_path, final_path)
                final_output = final_path
            except subprocess.CalledProcessError:
                st.warning("Audio merge failed — showing video without audio.")
                final_output = blurred_path

            status_text.markdown("✅ **Done!**")
            progress_bar.progress(1.0)

            st.divider()
            st.markdown("#### Result")
            st.video(final_output)

            with open(final_output, "rb") as f:
                st.download_button(
                    label="⬇️ Download Processed Video",
                    data=f.read(),
                    file_name="face_blurred_output.mp4",
                    mime="video/mp4"
                )
else:
    st.info("👆 Upload a video to get started.")