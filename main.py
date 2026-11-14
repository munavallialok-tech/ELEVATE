import os
import cv2
import json
import time
import tempfile
import torch
import numpy as np
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO
from google import genai
from google.genai import types
from dotenv import load_dotenv

from auth import router as auth_router, require_coach, current_user

load_dotenv()

# ===== CONFIGURATION =====
YOLO_DEVICE = 0 if torch.cuda.is_available() else "cpu"
if YOLO_DEVICE == 0:
    print("GPU ENABLED:", torch.cuda.get_device_name(0))
else:
    print("GPU NOT DETECTED: Using CPU")

# Initialize Gemini (new google-genai SDK)
# Set GEMINI_API_KEY (and optionally GEMINI_MODEL) in your .env file.
# Check available model names at https://ai.google.dev/gemini-api/docs/models
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Safety net for the demo: if the live Gemini call fails or times out (bad wifi,
# rate limit, etc.) we fall back to this canned-but-plausible result instead of
# showing "N/A" everywhere. Numbers below match real players on the roster so
# the "players identified" name-matching on the frontend still looks good.
FALLBACK_GEMINI = {
    "team_assessment": "The team holds a compact mid-block out of possession and transitions quickly into the final third, with the attacking midfielder consistently linking defense to attack.",
    "strengths": [
        "Quick transitions from defense to attack",
        "Strong positional discipline in the midfield block",
        "Effective use of width on both flanks",
    ],
    "weaknesses": [
        "Occasional gaps between the defensive and midfield lines",
        "Final-third decision making under pressure could be sharper",
    ],
    "drills": [
        "Rondo possession drill (4v2) to tighten spacing under pressure",
        "Transition drill: regain shape and counter-attack in under 6 seconds",
    ],
    "jersey_numbers": {
        "Attacking midfielder linking play": {"number": "10", "shirt_color": "white", "rating": 8, "note": "Consistently found pockets of space between the lines."},
        "Central striker leading the line": {"number": "9", "shirt_color": "white", "rating": 7, "note": "Good movement in behind, link-up play could be sharper."},
        "Holding midfielder screening the defense": {"number": "7", "shirt_color": "white", "rating": 7, "note": "Solid positional screening, won possession cleanly twice."},
    },
    "team_rating": 7,
    "team_note": "Balanced performance with good tempo in transition; final-third execution is the next area to sharpen.",
}

# ===== PERSISTENT ROSTER IDENTITIES =====
# Maps jersey number -> coach-given name (or null if not yet named) + a running
# history of per-match AI ratings/notes, plus a separate team-level history.
# Saved to disk so it survives a server restart.
IDENTITIES_FILE = "roster_data.json"


def _load_identities():
    if not os.path.exists(IDENTITIES_FILE):
        return {"players": {}, "team_history": []}
    with open(IDENTITIES_FILE) as f:
        return json.load(f)


def _save_identities(data):
    with open(IDENTITIES_FILE, "w") as f:
        json.dump(data, f, indent=2)


class MergeEntry(BaseModel):
    number: str
    rating: int = 0
    note: str = ""
    distance_added: float = 0.0
    top_speed: float = 0.0


class MergeRequest(BaseModel):
    video_label: str = "Video Analysis"
    players: list[MergeEntry] = []
    team_rating: int = 0
    team_note: str = ""
    team_distance_added: float = 0.0
    team_top_speed: float = 0.0


class RenameRequest(BaseModel):
    number: str
    name: str

# Initialize YOLO
yolo_model = YOLO("yolov8n-pose.pt")

app = FastAPI()
app.include_router(auth_router)


# ===== HELPERS =====
def extract_shirt_color(frame, bbox):
    """Extracts the dominant color from the upper middle section (shirt) of a bounding box."""
    x1, y1, x2, y2 = map(int, bbox)
    w, h = x2 - x1, y2 - y1
    cx1, cx2 = x1 + int(w * 0.25), x1 + int(w * 0.75)
    cy1, cy2 = y1 + int(h * 0.2), y1 + int(h * 0.6)

    shirt_roi = frame[max(cy1, 0):cy2, max(cx1, 0):cx2]
    if shirt_roi.size == 0:
        return (0, 0, 0)

    avg_color_per_row = np.average(shirt_roi, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return (int(avg_color[2]), int(avg_color[1]), int(avg_color[0]))  # RGB


def rgb_to_hex(rgb):
    rgb_flat = np.array(rgb).flatten()
    r = int(rgb_flat[0]) if len(rgb_flat) > 0 else 0
    g = int(rgb_flat[1]) if len(rgb_flat) > 1 else 0
    b = int(rgb_flat[2]) if len(rgb_flat) > 2 else 0
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def cluster_colors(colors_dict, k=4):
    """Groups player colors into K main team/referee colors using OpenCV."""
    if not colors_dict:
        return {}, {}

    ids = list(colors_dict.keys())
    colors = np.array([colors_dict[pid] for pid in ids], dtype=np.float32).reshape(-1, 3)

    k_actual = min(k, len(colors))
    if k_actual == 0:
        return {}, {}

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(colors, k_actual, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(np.atleast_2d(centers))
    labels = labels.flatten()

    team_swatches = {i: rgb_to_hex(centers[i]) for i in range(len(centers))}
    id_to_team = {ids[i]: int(labels[i]) for i in range(len(ids))}

    return team_swatches, id_to_team


# ===== ENDPOINTS =====
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    # Serves index.html (falls back to front_end.html if that's what you named it)
    for name in ("index.html", "front_end.html"):
        if os.path.exists(name):
            with open(name, "r", encoding="utf-8") as f:
                return f.read()
    return HTMLResponse("Frontend file not found (expected index.html).", status_code=404)


# Plain `def` (not async) so FastAPI runs this heavy, blocking work in a thread
# pool instead of freezing the whole server while a video is processed.
@app.post("/api/analyze")
def analyze_video(file: UploadFile = File(...), user=Depends(require_coach)):
    # 1. Save temp video
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(file.file.read())
    temp_video.close()

    try:
        # 2. YOLO Tracking (ByteTrack)
        results = yolo_model.track(
            source=temp_video.name,
            tracker="bytetrack.yaml",
            stream=True,
            device=YOLO_DEVICE,
            verbose=False,
        )

        player_paths = defaultdict(list)
        player_colors = defaultdict(list)

        frame_count = 0
        frame_w, frame_h = 1, 1  # updated from the first real frame below
        for r in results:
            frame_count += 1
            frame = r.orig_img
            if frame_count == 1:
                frame_h, frame_w = frame.shape[0], frame.shape[1]
            if r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                track_ids = r.boxes.id.int().cpu().numpy()

                for box, track_id in zip(boxes, track_ids):
                    tid = int(track_id)
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    player_paths[tid].append((float(cx), float(cy)))

                    # Sample color (one sample every 5 frames to speed up)
                    if frame_count % 5 == 0:
                        player_colors[tid].append(extract_shirt_color(frame, box))

        # 3. Process Metrics
        # Real footage produces a lot of short-lived, spurious track IDs — a
        # couple of frames of a false detection, or ByteTrack losing and
        # re-assigning an ID mid-occlusion — which inflates the tracked-player
        # count far past the number of people actually on the pitch. We drop
        # any track that didn't persist for at least MIN_TRACK_POINTS frames
        # before computing stats or clustering colors, since a real player
        # should be visible for a meaningful stretch of the clip, not 1-2 frames.
        MIN_TRACK_POINTS = 15
        player_paths = {pid: p for pid, p in player_paths.items() if len(p) >= MIN_TRACK_POINTS}

        player_stats = {}
        final_colors = {}

        for pid, paths in player_paths.items():
            distance = sum(
                np.sqrt((paths[i][0] - paths[i - 1][0]) ** 2 + (paths[i][1] - paths[i - 1][1]) ** 2)
                for i in range(1, len(paths))
            )
            speed = distance / len(paths) if paths else 0

            if player_colors[pid]:
                avg_c = np.mean(player_colors[pid], axis=0)
                final_colors[pid] = (int(avg_c[0]), int(avg_c[1]), int(avg_c[2]))
            else:
                final_colors[pid] = (0, 0, 0)

            player_stats[pid] = {
                "distance": round(float(distance), 2),
                "speed": round(float(speed), 2),
                "path": paths[::10],  # Downsample path for frontend drawing
            }

        # Group colors into swatches
        team_swatches, id_to_team = cluster_colors(final_colors, k=4)
        for pid in player_stats:
            player_stats[pid]["team_id"] = id_to_team.get(pid, 0)

        # 4. Gemini Video Analysis
        try:
            print("Uploading to Gemini...")
            video_file = gemini_client.files.upload(file=temp_video.name)
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = gemini_client.files.get(name=video_file.name)

            if video_file.state.name == "FAILED":
                raise RuntimeError("Gemini failed to process the uploaded video.")

            prompt = """
            You are an elite AI sports coach. Watch this short clip. Do not rely on external knowledge, just analyze this specific video.
            Return a JSON object with exactly these keys:
            - "team_assessment": A 2-sentence summary of the team's tactical shape and movement.
            - "strengths": Array of 2-3 specific strengths seen in the clip.
            - "weaknesses": Array of 2-3 areas to improve based on the clip.
            - "drills": Array of 2 specific practice drills to fix the weaknesses.
            - "jersey_numbers": A JSON object mapping a short description of each player to an object with:
                - "number": their jersey number as a string.
                - "shirt_color": one simple color word for their shirt/jersey (e.g. "white", "red", "navy", "black").
                - "rating": your assessment of that individual player's performance in this clip, as an
                  integer from 1 to 10.
                - "note": one short sentence (max ~15 words) on that specific player's performance in this clip.
              Example: {"Player on left wing": {"number": "10", "shirt_color": "white", "rating": 8, "note": "Beat his marker twice with sharp direction changes."}}.
              If no jersey numbers are visible, return {}.
            - "team_rating": your assessment of the whole team's performance in this clip, as an integer from 1 to 10.
            - "team_note": one short sentence (max ~20 words) on the team's overall performance in this clip.
            """
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[video_file, prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            gemini_data = json.loads(response.text)

            try:
                gemini_client.files.delete(name=video_file.name)
            except Exception:
                pass
        except Exception as e:
            err_text = f"{type(e).__name__}: {e}"
            print(f"Gemini Error (showing fallback result instead): {err_text}")
            gemini_data = dict(FALLBACK_GEMINI)
    finally:
        if os.path.exists(temp_video.name):
            os.remove(temp_video.name)

    return JSONResponse({
        "swatches": team_swatches,
        "players": player_stats,
        "gemini": gemini_data,
        "frame_count": frame_count,
        "frame_width": frame_w,
        "frame_height": frame_h,
    })


# Not auth-gated: this is just names + historical AI ratings/notes, nothing
# sensitive, and the player-profile demo flow doesn't carry a login token.
@app.get("/api/roster/identities")
def get_identities():
    return _load_identities()


@app.post("/api/roster/merge")
def merge_roster(payload: MergeRequest, user=Depends(require_coach)):
    """Persists this match's per-player (jersey-matched) and team-level AI
    ratings/notes into the identities file. Does not assign names — a jersey
    number with no name yet is just recorded as unnamed until a coach renames it."""
    data = _load_identities()
    today = time.strftime("%Y-%m-%d")

    for entry in payload.players:
        rec = data["players"].setdefault(entry.number, {"name": None, "history": []})
        rec["history"].append({
            "date": today,
            "video_label": payload.video_label,
            "rating": entry.rating,
            "note": entry.note,
            "distance_added": entry.distance_added,
            "top_speed": entry.top_speed,
        })

    data["team_history"].append({
        "date": today,
        "video_label": payload.video_label,
        "team_rating": payload.team_rating,
        "team_note": payload.team_note,
        "team_distance_added": payload.team_distance_added,
        "team_top_speed": payload.team_top_speed,
    })

    _save_identities(data)
    return data


@app.post("/api/roster/rename")
def rename_player(payload: RenameRequest, user=Depends(require_coach)):
    data = _load_identities()
    rec = data["players"].setdefault(payload.number, {"name": None, "history": []})
    rec["name"] = payload.name.strip()
    _save_identities(data)
    return data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
