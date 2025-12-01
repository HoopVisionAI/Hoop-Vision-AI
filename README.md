# 🏀 Hoop Vision AI

**Hoop Vision AI** is a real-time basketball stat tracking application that analyzes uploaded game videos to extract detailed player stats and event data. It detects shots, makes, attempts, three-point shots, rebounds, and assists using AI models and computer vision.

The app is built using **Python**, **Streamlit**, **YOLOv8**, **DeepSORT**, and **EasyOCR** for jersey number recognition.

---

## **Features**

- Upload a basketball game video for analysis
- Detect players and track them across the court
- Recognize jersey numbers to attribute stats to individual players
- Detect ball and identify when a shot goes through the hoop
- Determine three-point shots using a calibrated 3PT line
- Track statistics per player:
  - Points
  - Field Goals Made / Attempted
  - Three-Point Makes / Attempts
  - Rebounds
  - Assists
- Calibrate hoop box and 3PT line per video for accuracy
- Optimized frame sampling for faster processing

---

## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/hoop-vision-ai.git
cd hoop-vision-ai
