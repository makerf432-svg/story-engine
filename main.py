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

# Load environment keys
load_dotenv(BASE_DIR / "api.env")
load_dotenv(BASE_DIR / "keys.env")
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
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
# 🛡️ RATE LIMITER
# -------------------------------------------------------------
RATE_LIMIT_COUNT = 50
RATE_LIMIT_WINDOW = 20 * 60 * 60
ip_request_history = defaultdict(list)


def enforce_rate_limit(client_ip: str):
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW
    ip_request_history[client_ip] = [
        t for t in ip_request_history[client_ip] if t > cutoff
    ]
    if len(ip_request_history[client_ip]) >= RATE_LIMIT_COUNT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit reached ({RATE_LIMIT_COUNT} stories/20h). Please wait for the window to refresh.",
        )


def record_successful_generation(client_ip: str):
    ip_request_history[client_ip].append(time.time())


# -------------------------------------------------------------
# PYDANTIC DATA SCHEMAS
# -------------------------------------------------------------
class SparksRequest(BaseModel):
    summary: str
    characters: str = ""


class SynthesizeRequest(BaseModel):
    summary: str
    characters: str = ""
    curated_sparks: list[str]
    deep_edit: bool = False


# -------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------
def clean_output(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"^<thought>.*", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.replace(r"$\rightarrow$", "→").replace(
        r"\rightarrow", "→"
    )
    return cleaned.strip()


def parse_sparks_to_list(raw_text: str) -> list[str]:
    """Extracts 10 clean sentences from model output into a Python list."""
    lines = raw_text.split("\n")
    sparks = []
    for line in lines:
        cleaned_line = re.sub(r"^\s*\d+[\.\)\-]\s*", "", line).strip()
        cleaned_line = re.sub(r"^\s*[\*\-]\s*", "", cleaned_line).strip()
        if len(cleaned_line) > 10:
            sparks.append(cleaned_line)
    return sparks[:10]


def get_random_nouns(count=10) -> list[str]:
    for filename in ["noun2.txt", "nouns.txt"]:
        file_path = BASE_DIR / filename
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                nouns = [line.strip() for line in f if line.strip()]
            if nouns:
                return (
                    nouns
                    if len(nouns) < count
                    else random.sample(nouns, count)
                )
    raise FileNotFoundError(
        "Noun file (noun2.txt) not found in project directory!"
    )

# -------------------------------------------------------------
# API ROUTE: Reroll a Single Spark
# -------------------------------------------------------------
class RerollRequest(BaseModel):
    summary: str
    characters: str = ""


@app.post("/api/reroll-spark")
async def reroll_single_spark_endpoint(req: RerollRequest, request: Request):
    if not req.summary.strip():
        raise HTTPException(status_code=400, detail="Summary cannot be empty.")

    forwarded = request.headers.get("x-forwarded-for")
    client_ip = (
        forwarded.split(",")[0].strip()
        if forwarded
        else (request.client.host if request.client else "unknown")
    )
    enforce_rate_limit(client_ip)

    # Pick exactly 1 fresh random noun
    noun = get_random_nouns(1)[0]
    char_context = f"\nCharacters: {req.characters}" if req.characters else ""

    prompt = f'Story Premise: "{req.summary}"{char_context}\nTarget Noun: {noun}\nGenerate exactly 1 standalone, vivid metaphor sentence fusing them:'

    try:
        res = client.models.generate_content(
            model="gemma-4-31b-it",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are a creative concept-blending engine. Output exactly 1 single sentence with no numbers, bullet points, or quotes.",
                temperature=0.95,
                thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )
        new_spark = clean_output(res.text or "")
        new_spark = re.sub(r"^\s*[\d\.\-\*]+\s*", "", new_spark).strip()
        return {"spark": new_spark}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reroll Error: {str(e)}")
# -------------------------------------------------------------
# API ROUTE 1: Generate Catalyst Sparks
# -------------------------------------------------------------
@app.post("/api/sparks")
async def generate_sparks_endpoint(req: SparksRequest, request: Request):
    if not req.summary.strip():
        raise HTTPException(status_code=400, detail="Summary cannot be empty.")

    forwarded = request.headers.get("x-forwarded-for")
    client_ip = (
        forwarded.split(",")[0].strip()
        if forwarded
        else (request.client.host if request.client else "unknown")
    )
    enforce_rate_limit(client_ip)

    try:
        nouns = get_random_nouns(10)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    char_context = (
        f"\nKey Characters / Cast: {req.characters}" if req.characters else ""
    )

    step1_system = """You are a creative concept-blending engine.
1. Extract implied themes, atmospheres, sensory textures, and emotions from the premise and characters.
2. Pair each provided noun with one of those derived concepts to craft 10 distinct, standalone sentences.
3. STRICT ISOLATION: Do NOT connect the sentences into a story.
4. Output ONLY the 10 numbered sentences."""

    step1_prompt = f"Story Premise: \"{req.summary}\"{char_context}\n\n10 Nouns: {', '.join(nouns)}\n\nGenerate 10 standalone sentences:"

    try:
        res = client.models.generate_content(
            model="gemma-4-31b-it",
            contents=step1_prompt,
            config=types.GenerateContentConfig(
                system_instruction=step1_system,
                temperature=0.95,
                thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )
        cleaned_text = clean_output(res.text or "")
        sparks_list = parse_sparks_to_list(cleaned_text)

        if not sparks_list:
            raise ValueError("Failed to parse sparks from generation.")

        return {
            "nouns": nouns,
            "sparks": sparks_list,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Spark Generation Error: {str(e)}"
        )


# -------------------------------------------------------------
# API ROUTE 2: Synthesize Story from Curated Sparks
# -------------------------------------------------------------
@app.post("/api/synthesize")
async def synthesize_story_endpoint(req: SynthesizeRequest, request: Request):
    if not req.summary.strip():
        raise HTTPException(status_code=400, detail="Summary cannot be empty.")
    if not req.curated_sparks:
        raise HTTPException(
            status_code=400,
            detail="Please select at least one inspiration spark.",
        )

    forwarded = request.headers.get("x-forwarded-for")
    client_ip = (
        forwarded.split(",")[0].strip()
        if forwarded
        else (request.client.host if request.client else "unknown")
    )
    enforce_rate_limit(client_ip)

    sparks_formatted = "\n".join(
        [f"- {s}" for s in req.curated_sparks]
    )
    char_context = (
        f"\nLocked Character Profiles:\n{req.characters}"
        if req.characters
        else ""
    )

    # 1. The Drafter (Step 2A)
    drafter_system = "You are an imaginative fiction author. Write a bold, atmospheric first draft using the premise, characters, and selected inspiration sparks."
    drafter_prompt = f"""Story Premise:
{req.summary}{char_context}

Curated Inspiration Sparks:
{sparks_formatted}

Task:
Write a captivating, atmospheric short story draft driven by the premise."""

    try:
        draft_res = client.models.generate_content(
            model="gemma-4-31b-it",
            contents=drafter_prompt,
            config=types.GenerateContentConfig(
                system_instruction=drafter_system,
                temperature=0.95,
                thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )
        raw_draft = clean_output(draft_res.text or "")
        if not raw_draft:
            raise ValueError("Draft step returned empty prose.")

        # 2. The Editor (Step 2B)
        final_story = raw_draft
        editor_thinking = "HIGH" if req.deep_edit else "MINIMAL"

        editor_system = """You are a world-class literary editor. Polish the raw draft:
1. CUT FORCED ELEMENTS: Remove any awkward or shoehorned metaphors.
2. TIGHTEN PACING: Enhance emotional resonance and prose rhythm.
3. Output ONLY the final polished story."""

        editor_prompt = f"Original Premise:\n{req.summary}{char_context}\n\nRaw Draft:\n{raw_draft}\n\nTask: Output the final polished story."

        try:
            editor_res = client.models.generate_content(
                model="gemma-4-31b-it",
                contents=editor_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=editor_system,
                    temperature=0.9,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=editor_thinking
                    ),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )
            polished = clean_output(editor_res.text or "")
            if polished:
                final_story = polished
        except Exception:
            # Fallback to draft if editor pass fails
            pass

        record_successful_generation(client_ip)

        return {"story": final_story}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Story Synthesis Error: {str(e)}"
        )


# Mount static files
static_path = BASE_DIR / "static"
static_path.mkdir(exist_ok=True)
app.mount(
    "/", StaticFiles(directory=str(static_path), html=True), name="static"
)