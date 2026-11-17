# Face Blur by Gender — Video Pipeline

A video-processing app that detects every face in a video, classifies each one by gender, and blurs only the faces classified as women, while keeping the original audio. It's designed for content that has to follow cultural or modesty guidelines, where showing women's faces isn't appropriate but the rest of the footage should stay untouched.

Built during my AI Engineering internship at e& (Etisalat), June–August 2026.

**Live demo:** https://face-blur-gender-752m9p2wgdrxsf44xzaca4.streamlit.app/

*Runs on Streamlit's free tier, so use short clips (5–10 seconds). The app may take a minute to wake up on first visit.*

## How It Works

```
Uploaded video
     │
     ▼
Read frame by frame (OpenCV)
     │
     ├── every 4th frame ──► Downscale 50% → detect faces (RetinaFace)
     │                              │
     │                              ▼
     │                       Classify gender (DeepFace)
     │                              │
     │                              ▼
     │                       Update blur boxes
     │                       (cleared after 2 detection passes with no faces)
     │
     ▼
Gaussian-blur faces classified as "Woman"
(box expanded to cover hair and head)
     │
     ▼
Write blurred video → merge original audio back (ffmpeg, H.264 + AAC)
     │
     ▼
Preview and download in the browser
```

### Design choices

- **Detection every 4th frame:** face detection is the slow step, so it runs on every 4th frame and the boxes are reused in between. This makes processing roughly 4× faster.
- **Downscaled detection:** frames are shrunk to 50% before detection, then the boxes are scaled back up, which speeds things up further.
- **Expanded blur box:** the blur covers more than the detected face, with extra room above for hair, so the person isn't recognisable.
- **Ghost-blur prevention:** if no faces are found for two detection passes in a row, the blur boxes are cleared, so no blur is left behind after someone leaves the frame.
- **Audio preserved:** OpenCV writes video without sound, so ffmpeg merges the original audio track back into the final file.

## Tech Stack

| Component | Technology |
|---|---|
| Face detection | RetinaFace (via DeepFace) |
| Gender classification | DeepFace |
| Video processing | OpenCV |
| Audio merge and encoding | ffmpeg |
| Interface | Streamlit |
| Deployment | Streamlit Community Cloud |

## Project Structure

```
├── app.py                        # Streamlit app and processing pipeline
├── requirements.txt              # Python dependencies
├── packages.txt                  # System packages for Streamlit Cloud (ffmpeg)
├── runtime.txt                   # Python version for Streamlit Cloud
├── test.jpg                      # Test image for face/gender detection
└── Task4_Solution_Document.pdf   # Full solution write-up
```

## Running Locally

1. Clone the repo and install dependencies (Python 3.10):
   ```
   git clone https://github.com/uzair852/face-blur-gender.git
   cd face-blur-gender
   pip install -r requirements.txt
   ```
2. Install [ffmpeg](https://ffmpeg.org/download.html). Either add it to your PATH, or set the `FFMPEG_PATH` environment variable to the location of `ffmpeg.exe`.
3. Start the app:
   ```
   streamlit run app.py
   ```

The DeepFace models (about 650 MB) download automatically on the first run.

## Known Limitations

- **Classification isn't perfect:** gender classification sometimes gets a face wrong in individual frames, so a woman's face can briefly appear unblurred. For a production system where this matters, results would need human review, or the blur would need to be kept on a face once it's been classified as a woman.
- **Memory-heavy:** the models need roughly 1 GB of memory together, which is close to the free Streamlit tier's limit. Long videos can crash the live demo; running locally avoids this.
- **Speed:** processing is CPU-bound and much slower than real time on the free tier.
