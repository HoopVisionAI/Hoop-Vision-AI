import streamlit as st
import tempfile
import torch
import numpy as np
from moviepy.editor import VideoFileClip
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

st.set_page_config(page_title="Hoop Vision AI", layout="wide")
st.title("🏀 Hoop Vision AI - Headless Stats Tracker")

# ------------------------------
# Upload video
# ------------------------------
uploaded = st.file_uploader("Upload a basketball game video", type=["mp4","mov","mkv"])

# ------------------------------
# Calibration inputs
# ------------------------------
st.subheader("Court Calibration")
hoop_x1 = st.number_input("Hoop X1", min_value=0, value=400)
hoop_y1 = st.number_input("Hoop Y1", min_value=0, value=50)
hoop_x2 = st.number_input("Hoop X2", min_value=0, value=500)
hoop_y2 = st.number_input("Hoop Y2", min_value=0, value=100)
hoop_box = [hoop_x1, hoop_y1, hoop_x2, hoop_y2]

three_pt_y = st.number_input("3PT Line Y-coordinate", min_value=0, value=200)
frame_skip = st.slider("Frame Skip (every N frames)", min_value=1, max_value=10, value=5)

# ------------------------------
# Main processing
# ------------------------------
if uploaded and st.button("Analyze Video"):
    st.video(uploaded)
    with st.spinner("Analyzing video..."):
        # Save temp video
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded.read())
        video_path = temp_video.name

        # ------------------------------
        # Load YOLO models
        # ------------------------------
        device = "cuda" if torch.cuda.is_available() else "cpu"
        player_model = YOLO("yolov8n.pt")
        ball_model = YOLO("yolov8n.pt")  # replace with basketball-trained if available

        # ------------------------------
        # Load TrOCR
        # ------------------------------
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)

        tracker = DeepSort(max_age=30)
        player_stats = {}
        last_shot_frame = -30
        frame_number = 0

        clip = VideoFileClip(video_path)
        for frame in clip.iter_frames(fps=int(clip.fps / frame_skip)):
            frame_number += 1
            frame_rgb = np.array(frame)

            # -------------------------
            # Player detection & tracking
            # -------------------------
            results = player_model(frame_rgb)
            players = []  # fill with YOLO output boxes
            tracks = tracker.update_tracks([p for p in players], frame=frame_rgb)

            # -------------------------
            # Jersey recognition using TrOCR
            # -------------------------
            for tr in tracks:
                if 'player_id' not in tr:
                    x1, y1, x2, y2 = tr[1]
                    crop = frame_rgb[int(y1):int(y2), int(x1):int(x2)]
                    pil_img = Image.fromarray(crop)
                    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
                    generated_ids = trocr_model.generate(pixel_values)
                    jersey_number = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    if jersey_number:
                        tr['player_id'] = jersey_number
                        if jersey_number not in player_stats:
                            player_stats[jersey_number] = {
                                "points":0,"makes":0,"attempts":0,"three_pt_makes":0,"three_pt_attempts":0,
                                "rebounds":0,"assists":0
                            }

            # -------------------------
            # Ball detection
            # -------------------------
            ball_results = ball_model(frame_rgb)
            ball_bbox = None  # extract bounding box

            # -------------------------
            # Shot detection
            # -------------------------
            if ball_bbox and (frame_number - last_shot_frame > 10):
                bx = (ball_bbox[0]+ball_bbox[2])/2
                by = (ball_bbox[1]+ball_bbox[3])/2
                x1,y1,x2,y2 = hoop_box
                if bx>x1 and bx<x2 and by>y1 and by<y2:
                    nearest_player = min(tracks, key=lambda t: abs((t[1][0]+t[1][2])/2 - bx))
                    if 'player_id' in nearest_player:
                        pid = nearest_player['player_id']
                        player_stats[pid]["makes"] += 1
                        player_stats[pid]["attempts"] += 1
                        if ((nearest_player[1][1]+nearest_player[1][3])/2) < three_pt_y:
                            player_stats[pid]["points"] += 3
                            player_stats[pid]["three_pt_makes"] += 1
                            player_stats[pid]["three_pt_attempts"] += 1
                        else:
                            player_stats[pid]["points"] += 2
                        last_shot_frame = frame_number

            # -------------------------
            # Rebound detection (simplified)
            # -------------------------
            if ball_bbox and (frame_number - last_shot_frame <= 5):
                for tr in tracks:
                    if 'player_id' in tr:
                        pid = tr['player_id']
                        player_stats[pid]["rebounds"] += 1

        # -------------------------
        # Display stats
        # -------------------------
        st.subheader("📊 Player Stats")
        for pid, stats in player_stats.items():
            st.write(f"Player {pid}: {stats}")

    st.success("Analysis complete!")
