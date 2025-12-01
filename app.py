import streamlit as st
import os
import tempfile

from video_ingest import VideoIngest
from detector import Detector
from tracker import TrackerWrapper
from jersey_ocr import JerseyOCR
from event_classifier import EventClassifier
from stats_aggregator import StatsAggregator
from highlight_maker import make_highlights
from report import make_html_report

st.set_page_config(page_title="Hoop Vision AI", layout="wide")

st.title("🏀 Hoop Vision AI")
st.subheader("Upload any basketball game. Get stats, breakdowns, and automatic highlights.")

uploaded = st.file_uploader("Upload game video", type=["mp4", "mov", "mkv"])

if uploaded:
    st.video(uploaded)
    if st.button("Analyze Video"):
        with st.spinner("Analyzing game…"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                temp_video.write(uploaded.read())
                video_path = temp_video.name

            out_dir = tempfile.mkdtemp()

            vi = VideoIngest(video_path)
            det = Detector('yolov8n.pt')
            tracker = TrackerWrapper()
            ocr = JerseyOCR(gpu=False)
            ec = EventClassifier()
            sa = StatsAggregator()

            events = []

            for idx, t, frame in vi.frames():
                dets = det.detect(frame)
                tracks = tracker.update(dets, frame)

                # jersey numbers
                for tr in tracks:
                    if 'player_id' not in tr:
                        num, conf = ocr.read_number(frame, tr['bbox'])
                        if num and conf > 0.4:
                            tr['player_id'] = num

                # ball detection
                ball_bbox = None
                for d in dets:
                    if d['name'] in ('ball', 'basketball'):
                        ball_bbox = d['bbox']
                        break

                evt = ec.update(t, tracks, ball_bbox)
                if evt:
                    events.append(evt)
                    sa.record_event(evt)

            vi.release()

            highlights = make_highlights(video_path, events, os.path.join(out_dir, "highlights"))
            report_path = os.path.join(out_dir, "report.html")
            make_html_report(sa.get_player_stats(), events, highlights, report_path)

        st.success("Done!")

        st.subheader("📊 Game Breakdown")
        st.markdown(f"[Open Full Report]({report_path})")

        st.subheader("🎬 Highlight Clips")
        for h in highlights:
            st.video(h["file"])
