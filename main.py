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

BASE_DIR = Path(__file__).parent

load_dotenv(BASE_DIR / "api.env")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Missing API Key! Please set it in your env file.")

client = genai.Client(api_key=api_key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# 🛡️ RATE LIMITER (Check first, record only on success)
# -------------------------------------------------------------
RATE_LIMIT_COUNT = 50
RATE_LIMIT_WINDOW = 20 * 60 * 60  # 20 hours
ip_request_history = defaultdict(list)


def enforce_rate_limit(client_ip: str):
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW
    # Purge expired timestamps
    ip_request_history[client_ip] = [
        t for t in ip_request_history[client_ip] if t > cutoff
    ]

    if len(ip_request_history[client_ip]) >= RATE_LIMIT_COUNT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit reached! You have used your {RATE_LIMIT_COUNT} generations for this 20-hour window.",
        )


def record_successful_request(client_ip: str):
    ip_request_history[client_ip].append(time.time())


class StoryRequest(BaseModel):
    summary: str
    deep_edit: bool = False


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
    raise FileNotFoundError("Noun file (noun2.txt) not found!")


@app.post("/api/generate")
async def generate_story(req: StoryRequest, request: Request):
    # 1. Validate Input
    if not req.summary or not req.summary.strip():
        raise HTTPException(status_code=400, detail="Summary cannot be empty.")

    # 2. Check Rate Limit
    forwarded = request.headers.get("x-forwarded-for")
    client_ip = (
        forwarded.split(",")[0].strip()
        if forwarded
        else (request.client.host if request.client else "unknown")
    )
    enforce_rate_limit(client_ip)

    # 3. Load Nouns
    try:
        nouns = get_random_nouns(10)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # -------------------------------------------------------------
    # STEP 1: Catalyst Sparks
    # -------------------------------------------------------------
    step1_system = """You are a creative concept-blending engine.
1. Analyze the story summary to derive implied themes, atmospheres, sensory textures, and emotions.
2. Pair each of the 10 provided nouns with one of those derived concepts.
3. Craft exactly 10 distinct, standalone sentences (1 per noun).
4. STRICT ISOLATION: Do NOT write a continuous plot. Each sentence must stand on its own.
5. Output ONLY the 10 numbered sentences."""

    step1_prompt = f"Story Summary: \"{req.summary}\"\n\nProvided Nouns: {', '.join(nouns)}\n\nGenerate 10 standalone sentences:"

    try:
        step1_res = client.models.generate_content(
            model="gemma-4-31b-it",
            contents=step1_prompt,
            config=types.GenerateContentConfig(
                system_instruction=step1_system,
                temperature=0.95,
                thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        clean_sparks = clean_output(step1_res.text or "")
        if not clean_sparks:
            raise ValueError("Step 1 returned an empty spark list.")

        # -------------------------------------------------------------
        # STEP 2A: The Drafter
        # -------------------------------------------------------------
        drafter_system = "You are an imaginative fiction author. Write a bold, atmospheric first draft using the provided premise and inspiration sparks."
        drafter_prompt = f"Premise:\n{req.summary}\n\nInspiration Sparks:\n{clean_sparks}\n\nTask: Write the story draft."

        draft_res = client.models.generate_content(
            model="gemma-4-31b-it",
            contents=drafter_prompt,
            config=types.GenerateContentConfig(
                system_instruction=drafter_system,
                temperature=0.95,
                thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        raw_draft = clean_output(draft_res.text or "")
        if not raw_draft:
            raise ValueError("Step 2A returned an empty draft.")

        # -------------------------------------------------------------
        # STEP 2B: The Editor (with automatic fallback to draft)
        # -------------------------------------------------------------
        final_story = raw_draft
        editor_thinking = "HIGH" if req.deep_edit else "MINIMAL"

        editor_system = """You are a world-class literary editor. Polish the raw draft:
1. CUT FORCED ELEMENTS: Remove awkward metaphors so prose feels 100% natural.
2. TIGHTEN PACING: Ensure emotional resonance, flow, and vivid imagery.
3. Output ONLY the finished story."""

        editor_prompt = f"Original Premise:\n{req.summary}\n\nRaw Draft to Polish:\n{raw_draft}\n\nTask: Output the final polished story."

        try:
            editor_res = client.models.generate_content(
                model="gemma-4-31b-it",
                contents=editor_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=editor_system,
                    temperature=0.9,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=editor_thinking,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    ),
                ),
            )
            polished = clean_output(editor_res.text or "")
            if polished:
                final_story = polished
        except Exception:
            # Safe Fallback: keep raw_draft if editor pass times out or glitches
            pass

        # 4. Record successful generation against rate limit
        record_successful_request(client_ip)

        return {
            "nouns": nouns,
            "sparks": clean_sparks,
            "story": final_story,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Generation Error: {str(e)}"
        )


static_path = BASE_DIR / "static"
static_path.mkdir(exist_ok=True)
app.mount(
    "/", StaticFiles(directory=str(static_path), html=True), name="static"
)