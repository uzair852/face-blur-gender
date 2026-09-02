import cv2
from deepface import DeepFace

# ---- CONFIG ----
INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "output.mp4"
DETECT_EVERY_N_FRAMES = 5   # run detection every 5th frame, reuse boxes in between
BLUR_STRENGTH = 51          # must be odd number, higher = blurrier
EXPAND_RATIO = 0.3          # how much to expand the box beyond the face (covers hair)

def blur_region(frame, x, y, w, h, expand_ratio=EXPAND_RATIO):
    """Apply Gaussian blur to a rectangular region, expanded to cover hair/more area."""
    frame_h, frame_w = frame.shape[:2]

    expand_w = int(w * expand_ratio)
    expand_h_top = int(h * expand_ratio * 1.5)   # more expansion upward for hair
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
    """Run DeepFace on a frame, return list of (x, y, w, h, gender)."""
    results = []
    try:
        analysis = DeepFace.analyze(
            img_path=frame,
            actions=['gender'],
            detector_backend='retinaface',
            enforce_detection=False
        )
        if isinstance(analysis, dict):
            analysis = [analysis]
        for face in analysis:
            region = face['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            dominant_gender = face['dominant_gender']
            results.append((x, y, w, h, dominant_gender))
    except Exception as e:
        print(f"Detection error on frame: {e}")
    return results

def main():
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"Error: could not open {INPUT_VIDEO}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    last_detections = []
    frame_idx = 0

    print(f"Processing {total_frames} frames at {fps} fps...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % DETECT_EVERY_N_FRAMES == 0:
            last_detections = detect_and_classify(frame)

        for (x, y, w, h, gender) in last_detections:
            if gender == "Woman":
                frame = blur_region(frame, x, y, w, h)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")

    cap.release()
    out.release()
    print(f"Done. Output saved to {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()