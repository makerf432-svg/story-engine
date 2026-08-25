import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pydantic import BaseModel

# 1. Base Directory
BASE_DIR = Path(__file__).parent

# 2. Load API keys
load_dotenv(BASE_DIR / "api.env")
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "Missing API Key! Please set GEMINI_API_KEY or GOOGLE_API_KEY in your env file."
    )

client = genai.Client(api_key=api_key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# 🛡️ IP RATE LIMITER CONFIGURATION (50 requests per 20 hours)
# -------------------------------------------------------------
RATE_LIMIT_COUNT = 50
RATE_LIMIT_WINDOW = 20 * 60 * 60  # 20 hours in seconds (72,000s)
ip_request_history = defaultdict(list)


def check_rate_limit(client_ip: str):
    """Tracks and enforces the 50 reqs / 20 hours limit per IP."""
    now = time.time()
    cutoff_time = now - RATE_LIMIT_WINDOW

    # Filter out requests older than 20 hours
    ip_request_history[client_ip] = [
        t for t in ip_request_history[client_ip] if t > cutoff_time
    ]

    # Check if user exceeded the limit
    if len(ip_request_history[client_ip]) >= RATE_LIMIT_COUNT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded! You can only make {RATE_LIMIT_COUNT} requests every 20 hours.",
        )

    # Record current request timestamp
    ip_request_history[client_ip].append(now)


class StoryRequest(BaseModel):
    summary: str


def clean_output(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"^<thought>.*", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.replace(r"$\rightarrow$", "→").replace(
        r"\rightarrow", "→"
    )
    return cleaned.strip()


def get_random_nouns(count=10) -> list[str]:
    for filename in ["noun2.txt", "nouns.txt"]:
        file_path = BASE_DIR / filename
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                nouns = [line.strip() for line in f if line.strip()]
            if nouns:
                if len(nouns) < count:
                    return nouns
                return random.sample(nouns, count)

    raise FileNotFoundError(
        "Noun file not found! Please ensure 'noun2.txt' or 'nouns.txt' exists in your project folder with valid nouns."
    )


@app.post("/api/generate")
async def generate_story(req: StoryRequest, request: Request):
    # 1. Get Client IP Address
    forwarded = request.headers.get("x-forwarded-for")
    client_ip = (
        forwarded.split(",")[0].strip()
        if forwarded
        else (request.client.host if request.client else "unknown")
    )

    # 2. Check Rate Limit
    check_rate_limit(client_ip)

    if not req.summary.strip():
        raise HTTPException(status_code=400, detail="Summary cannot be empty.")

    try:
        nouns = get_random_nouns(10)
    except FileNotFoundError as fnf_err:
        raise HTTPException(status_code=404, detail=str(fnf_err))

    # -------------------------------------------------------------
    # STEP 1: Generate 10 isolated spark sentences
    # -------------------------------------------------------------
    step1_system = """You are a surreal sensory sentence generator.
RULES:
1. Output EXACTLY 10 numbered sentences (1 to 10).
2. Do NOT write explanations, thoughts, or introductory notes.
3. Every sentence must describe a completely different, vivid scene or image.
4. Each sentence must strictly be isolated—no shared characters or storylines."""

    step1_prompt = f"Nouns: {', '.join(nouns)}\nGenerate 10 vivid, standalone sentences:"

    try:
        step1_res = client.models.generate_content(
            model="gemma-4-31b-it",
            contents=step1_prompt,
            config=types.GenerateContentConfig(
                system_instruction=step1_system,
                temperature=0.9,
                thinking_config=types.ThinkingConfig(
                    thinking_level="MINIMAL",
                ),
            ),
        )

        clean_sparks = clean_output(step1_res.text or "")
        if not clean_sparks:
            raise ValueError("Step 1 returned an empty response.")

        # -------------------------------------------------------------
        # STEP 2: Synthesize the final creative story
        # -------------------------------------------------------------
        step2_system = "You are an imaginative literary fiction author known for rich atmospheric prose. Output only the finished story with no meta-commentary."

        step2_prompt = f"""
Core Story Premise:
{req.summary}

Sensory Moodboard & Metaphors (Inspiration):
{clean_sparks}

Task:
Write a captivating short story based on the user's premise. 
Use the metaphors and imagery from the inspiration list to enrich the atmosphere and themes, but integrate them naturally into the world rather than forcing every single word verbatim.
"""

        step2_res = client.models.generate_content(
            model="gemma-4-31b-it",
            contents=step2_prompt,
            config=types.GenerateContentConfig(
                system_instruction=step2_system,
                temperature=0.75,
                thinking_config=types.ThinkingConfig(
                    thinking_level="MINIMAL",
                ),
            ),
        )

        clean_story = clean_output(step2_res.text or "")
        if not clean_story:
            raise ValueError("Step 2 returned an empty story response.")

        return {
            "nouns": nouns,
            "sparks": clean_sparks,
            "story": clean_story,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Generation Error: {str(e)}"
        )


# Mount static website files
static_path = BASE_DIR / "static"
static_path.mkdir(exist_ok=True)
app.mount(
    "/", StaticFiles(directory=str(static_path), html=True), name="static"
)