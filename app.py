import streamlit as st
import tempfile
import os
import torch
import numpy as np
from moviepy.editor import VideoFileClip
import easyocr
from deep_sort_realtime.deepsort_tracker import DeepSort

st.set_page_config(page_title="Hoop Vision AI", layout="wide")
st.title("🏀 Hoop Vision AI - Real Stats Tracker (Headless)")

# ------------------------------
# Upload video
# ------------------------------
uploaded = st.file_uploader("Upload a basketball game video", type=["mp4", "mov", "mkv"])

# ------------------------------
# Calibration inputs
# ------------------------------
st.subheader("Court Calibration")
st.write("Set hoop box and 3PT line Y-coordinate (pixels)")

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
        # Save uploaded file
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded.read())
        video_path = temp_video.name

        # -------------------------
        # Load YOLO models (PyTorch only)
        # -------------------------
        device = "cuda" if torch.cuda.is_available() else "cpu"
        player_model = torch.hub.load('ultralytics/yolov8', 'yolov8n.pt', pretrained=True).to(device)
        ball_model = torch.hub.load('ultralytics/yolov8', 'yolov8n.pt', pretrained=True).to(device)  # Replace with basketball-trained model if available

        reader = easyocr.Reader(['en'])
        tracker = DeepSort(max_age=30)

        player_stats = {}
        last_shot_frame = -30
        frame_number = 0

        clip = VideoFileClip(video_path)
        for frame in clip.iter_frames(fps=int(clip.fps / frame_skip)):
            frame_number += 1
            frame_rgb = np.array(frame)  # MoviePy gives RGB frames

            # -------------------------
            # 1) Player Detection (YOLO)
            # -------------------------
            frame_tensor = torch.from_numpy(frame_rgb).permute(2,0,1).float()/255.0
            frame_tensor = frame_tensor.unsqueeze(0).to(device)
            player_results = player_model(frame_tensor)[0].cpu().detach().numpy()  # simplified
            players = []  # fill with bboxes
            # (Implement extraction from YOLO output here)

            # Track players
            tracks = tracker.update_tracks([p for p in players], frame=frame_rgb)

            # -------------------------
            # 2) Jersey OCR
            # -------------------------
            for tr in tracks:
                if 'player_id' not in tr:
                    x1, y1, x2, y2 = tr[1]
                    crop = frame_rgb[int(y1):int(y2), int(x1):int(x2)]
                    try:
                        result = reader.readtext(crop)
                        if result:
                            tr['player_id'] = result[0][1]
                            pid = tr['player_id']
                            if pid not in player_stats:
                                player_stats[pid] = {
                                    "points":0, "makes":0, "attempts":0,
                                    "three_pt_makes":0, "three_pt_attempts":0,
                                    "rebounds":0, "assists":0
                                }
                    except:
                        continue

            # -------------------------
            # 3) Ball Detection (YOLO)
            # -------------------------
            ball_results = ball_model(frame_tensor)[0].cpu().detach().numpy()
            ball_bbox = None
            # Extract bounding box of basketball from ball_results here

            # -------------------------
            # 4) Shot & 3PT Detection
            # -------------------------
            if ball_bbox and (frame_number - last_shot_frame > 10):
                bx = (ball_bbox[0]+ball_bbox[2])/2
                by = (ball_bbox[1]+ball_bbox[3])/2
                x1,y1,x2,y2 = hoop_box
                if bx>x1 and bx<x2 and by>y1 and by<y2:
                    # Assign nearest player as shooter
                    nearest_player = min(tracks, key=lambda t: abs((t[1][0]+t[1][2])/2 - bx))
                    if 'player_id' in nearest_player:
                        pid = nearest_player['player_id']

                        # Update stats
                        player_stats[pid]["makes"] += 1
                        player_stats[pid]["attempts"] += 1

                        # 3PT check
                        player_y = (nearest_player[1][1]+nearest_player[1][3])/2
                        if player_y < three_pt_y:
                            player_stats[pid]["points"] += 3
                            player_stats[pid]["three_pt_makes"] += 1
                            player_stats[pid]["three_pt_attempts"] += 1
                        else:
                            player_stats[pid]["points"] += 2

                        last_shot_frame = frame_number

            # -------------------------
            # 5) Rebound & Assist Detection (simplified)
            # -------------------------
            if ball_bbox and (frame_number - last_shot_frame <= 5):
                for tr in tracks:
                    if 'player_id' in tr:
                        pid = tr['player_id']
                        player_stats[pid]["rebounds"] += 1

        # -------------------------
        # Display Stats
        # -------------------------
        st.subheader("📊 Player Stats")
        for pid, stats in player_stats.items():
            st.write(f"Player {pid}: {stats}")

    st.success("Analysis complete!")
