# DOOM FC — AI Performance Intelligence Platform

A hackathon prototype coach/player dashboard that combines **real computer-vision
match tracking** (YOLOv8 + ByteTrack) and **real AI video analysis** (Gemini)
with a full demo dataset to show what a finished product would look like.

## What's real vs. what's demo data

Being upfront about this matters for judging — here's the exact split:

| Feature | Status |
|---|---|
| Video upload → YOLOv8 pose tracking of real players | **Real.** Runs on your uploaded clip. |
| Team-color clustering (K-means on jersey colors) | **Real.** |
| Gemini video analysis (tactical summary, strengths/weaknesses, drills) | **Real.** Calls the Gemini API on your actual uploaded clip. |
| Jersey-number + name identification | **Real**, with a caveat: Gemini reads jersey numbers from the footage and reports a shirt color; the app matches that shirt color against the team color you selected to decide which numbers belong to your team. This is an approximate color-name match, not a pixel-exact one. |
| Per-player & team ratings/notes, accumulated over time | **Real**, persisted to `roster_data.json`, survives server restarts. |
| 17-player squad roster, season stats, attribute ratings | **Demo data.** Fabricated for the prototype, clearly labeled as such throughout the UI. |
| Match history (except the one row added after each real analysis), Squad Combine / practice records | **Demo data.** |
| "Live match tracking" page (scripted 60-second animation) | **Demo/simulation**, labeled "SIMULATION · NOT LIVE SENSOR DATA" in the UI — illustrates the intended interface, doesn't touch uploaded footage. |

**Known limitations, stated plainly:**
- There is no real per-player movement tracking tied to identity. YOLO tracks
  anonymous blobs by position/color; Gemini separately reads jersey numbers
  from the footage. There's no shared ID between the two systems, so
  individual distance/speed numbers are the *team's* aggregate tracked
  movement applied to the players Gemini confirmed were in the clip — not an
  exact per-player GPS reading.
- Real tracked distance/speed updates are **session-only** right now — they
  reset if the server restarts. Names, AI ratings, and notes *do* persist
  (`roster_data.json`).
- A `MIN_TRACK_POINTS` filter (15 frames) drops short-lived spurious YOLO
  tracks so the "players tracked" count stays realistic. This is a heuristic —
  tune it in `main.py` if your footage's frame rate/length makes it over- or
  under-count.
- `users.json` and `roster_data.json` are flat JSON files, not a real
  database — fine for a demo, not for production.

## Architecture

- **Backend**: FastAPI (`main.py`, `auth.py`). YOLOv8-pose (via `ultralytics`)
  for player tracking, OpenCV for color clustering, Google's `google-genai`
  SDK for Gemini video analysis.
- **Frontend**: a single self-contained `index.html` (vanilla JS, no build
  step) served directly by FastAPI.
- **Storage**: `users.json` (accounts) and `roster_data.json` (AI-identified
  player names + history) — plain JSON files, created automatically.

## Setup

1. **Python 3.10+** recommended (for `list[...]` type hints used in `main.py`).
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_key_here
   GEMINI_MODEL=gemini-2.5-flash
   COACH_CODE=
   ```
   - `GEMINI_API_KEY` — required for real Gemini video analysis. Get one at
     [ai.google.dev](https://ai.google.dev/gemini-api/docs/api-key). Without
     it, every analysis falls back to the canned "safety net" result in
     `main.py` (`FALLBACK_GEMINI`) — the app still works, just without real
     AI output.
   - `GEMINI_MODEL` — optional, defaults to `gemini-2.5-flash`.
   - `COACH_CODE` — optional. If set, new coach sign-ups must enter this code.
     Leave blank for open sign-up.
4. First run downloads the YOLOv8-pose weights (`yolov8n-pose.pt`)
   automatically — needs internet the first time only.

## Running it

```bash
python main.py
```
Serves on `http://127.0.0.1:8000`. Open that URL in a browser.

**Default demo coach login** (auto-created on first use):
- Username: `DR DOOM`
- Password: `DOOMSDAY`

Or register your own coach/student account from the login screen.

## Using it

1. Log in as coach.
2. Go to **Video Analysis**, upload a short clip of your team.
3. Once processing finishes, pick the swatch(es) matching your team's shirt
   color (shift-click to also include a goalkeeper's different-colored kit).
4. Click **Merge Analysis with Team DB**. You'll see:
   - Gemini's tactical assessment, strengths/weaknesses, drills.
   - A team performance rating.
   - Players identified in the clip, matched to your roster by jersey number
     and filtered to your team's color.
5. If a jersey number doesn't match anyone in the roster, it's added as a new
   unnamed `Player #N` entry. Go to **Player directory → their profile** and
   use the "Name this player" box to give them a real name — the app
   remembers it for every future analysis.

## File structure

```
.
├── main.py           # FastAPI app: video analysis, YOLO tracking, Gemini, roster persistence
├── auth.py           # Login/register/session + role-based access (coach vs student)
├── index.html         # Entire frontend (served by main.py)
├── requirements.txt
├── .env               # You create this — not committed
├── users.json         # Auto-created on first registration — not committed
└── roster_data.json   # Auto-created on first video merge — not committed
```
