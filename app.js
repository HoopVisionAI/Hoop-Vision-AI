// ---------------------------
// Hoop Vision AI (Single File)
// ---------------------------

const express = require("express");
const multer = require("multer");
const ffmpeg = require("fluent-ffmpeg");
const ffmpegPath = require("ffmpeg-static");
ffmpeg.setFfmpegPath(ffmpegPath);

const path = require("path");
const fs = require("fs");

const app = express();
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// Storage for uploads
const upload = multer({ dest: "uploads/" });

// -----------------------------
// FRONTEND (HTML IN ONE FILE)
// -----------------------------
app.get("/", (req, res) => {
  res.send(`
    <html>
      <head>
        <title>Hoop Vision AI</title>
        <style>
          body { font-family: Arial; margin: 40px; background:#111; color:white; }
          .container { max-width: 600px; margin:auto; padding:20px; background:#222; border-radius:10px; }
          input { margin-top:10px; }
          button { margin-top:20px; padding:10px; width:100%; background:#ff5500; color:white; border:none; font-size:18px; cursor:pointer; }
          button:hover { background:#ff7733; }
          h1 { text-align:center; }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>🏀 Hoop Vision AI</h1>
          <p>Upload a basketball game clip to auto-generate stats, breakdowns, and highlights.</p>

          <form action="/analyze" method="POST" enctype="multipart/form-data">
            <input type="file" name="video" accept="video/*" required />
            <button type="submit">Analyze Game</button>
          </form>
        </div>
      </body>
    </html>
  `);
});

// -------------------------------------------------------
// MAIN LOGIC — Stats, Highlights, Frame Extraction in ONE
// -------------------------------------------------------

app.post("/analyze", upload.single("video"), async (req, res) => {
  const file = req.file;

  // Folder to hold frames
  const framesFolder = `frames_${Date.now()}`;
  fs.mkdirSync(framesFolder);

  // 1) Extract frames every 1 second
  await new Promise((resolve, reject) => {
    ffmpeg(file.path)
      .output(`${framesFolder}/frame_%04d.jpg`)
      .outputOptions(["-vf fps=1"])
      .on("end", resolve)
      .on("error", reject)
      .run();
  });

  // 2) Simulated AI analysis (replace with real model later)
  let stats = {
    points: Math.floor(Math.random() * 80),
    rebounds: Math.floor(Math.random() * 30),
    assists: Math.floor(Math.random() * 20),
    turnovers: Math.floor(Math.random() * 15),
    threesMade: Math.floor(Math.random() * 20),
  };

  // Fake highlight detection
  let highlights = [
    "Dunk at 00:23",
    "Deep 3-pointer at 01:15",
    "Chase-down block at 02:07",
  ];

  // 3) RESPONSE PAGE
  res.send(`
      <html>
        <head>
          <title>Hoop Vision AI – Results</title>
          <style>
            body { font-family: Arial; margin: 40px; background:#111; color:white; }
            .container { max-width: 700px; margin:auto; padding:20px; background:#222; border-radius:10px; }
            h1, h2 { text-align:center; }
            .stat-box { background:#333; padding:15px; border-radius:10px; margin:10px 0; }
          </style>
        </head>

        <body>
          <div class="container">
            <h1>🏀 Hoop Vision AI Results</h1>

            <h2>📊 Game Stats</h2>
            <div class="stat-box">
              <p><b>Points:</b> ${stats.points}</p>
              <p><b>Rebounds:</b> ${stats.rebounds}</p>
              <p><b>Assists:</b> ${stats.assists}</p>
              <p><b>Turnovers:</b> ${stats.turnovers}</p>
              <p><b>3-Pointers Made:</b> ${stats.threesMade}</p>
            </div>

            <h2>🔥 Highlights Detected</h2>
            <div class="stat-box">
              ${highlights.map(h => `<p>${h}</p>`).join("")}
            </div>

            <h2>🖼 Extracted Frames</h2>
            <p>Here are a few frames captured from your video:</p>

            ${fs.readdirSync(framesFolder)
              .slice(0, 10)
              .map(f => `<img src="/${framesFolder}/${f}" width="200" style="margin:5px;border-radius:5px;">`)
              .join("")}

          </div>
        </body>
      </html>
  `);
});

// Serve frame images
app.use(express.static("."));

// Start server
app.listen(3000, () => console.log("Hoop Vision AI running on http://localhost:3000"));
