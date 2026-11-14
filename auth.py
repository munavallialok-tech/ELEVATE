import os, json, hashlib, secrets
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel

router = APIRouter(prefix="/api")

USERS_FILE = "users.json"
COACH_CODE = os.getenv("COACH_CODE", "")  # optional: set in .env to restrict coach sign-ups
SESSIONS = {}   # token -> email
LATEST = {}     # last analysis shared by a coach


def _load():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE) as f:
        return json.load(f)


def _save(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000).hex()


class Creds(BaseModel):
    email: str
    password: str
    name: str = ""
    role: str = "student"
    code: str = ""


def _issue(email, user):
    token = secrets.token_hex(24)
    SESSIONS[token] = email
    return {"token": token, "user": {"name": user["name"], "email": email, "role": user["role"]}}


def current_user(authorization: str = Header(default="")):
    token = authorization.replace("Bearer ", "")
    email = SESSIONS.get(token)
    user = _load().get(email) if email else None
    if not user:
        raise HTTPException(401, "Please sign in again.")
    return {**user, "email": email}


def require_coach(user=Depends(current_user)):
    if user["role"] != "coach":
        raise HTTPException(403, "Only coaches can do this.")
    return user


@router.post("/register")
def register(c: Creds):
    email = c.email.strip().lower()
    users = _load()
    if not email or len(c.password) < 6:
        raise HTTPException(400, "Enter an email and a password of at least 6 characters.")
    if email in users:
        raise HTTPException(400, "That email already has an account. Sign in instead.")
    if c.role not in ("coach", "student"):
        raise HTTPException(400, "Choose coach or student.")
    if c.role == "coach" and COACH_CODE and c.code != COACH_CODE:
        raise HTTPException(403, "That coach code isn't right.")
    salt = secrets.token_hex(16)
    users[email] = {"name": c.name.strip() or email.split("@")[0], "role": c.role,
                    "salt": salt, "hash": _hash(c.password, salt)}
    _save(users)
    return _issue(email, users[email])


@router.post("/login")
def login(c: Creds):
    email = c.email.strip().lower()
    user = _load().get(email)
    if not user or _hash(c.password, user["salt"]) != user["hash"]:
        raise HTTPException(401, "Email or password is incorrect.")
    if user["role"] != c.role:
        raise HTTPException(403, f"This account is a {user['role']} account. Switch the role above.")
    return _issue(email, user)


@router.get("/me")
def me(user=Depends(current_user)):
    return {"name": user["name"], "email": user["email"], "role": user["role"]}


@router.post("/share")
def share(payload: dict, user=Depends(require_coach)):
    """Coach publishes the analysis students will see."""
    LATEST.clear()
    LATEST.update(payload)
    return {"ok": True}


@router.get("/latest")
def latest(user=Depends(current_user)):
    return LATEST or {}
