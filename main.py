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
load_dotenv(BASE_DIR / "keys.env")
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key: raise ValueError("Missing API Key!")

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemma-4-31b-it"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- RATE LIMITER ---
WINDOW_20_HOURS = 20 * 60 * 60
STORY_LIMIT = 100
REROLL_LIMIT = 500
ip_story_history = defaultdict(list)
ip_reroll_history = defaultdict(list)

def check_and_clean_limits(history_dict: dict, client_ip: str, max_limit: int, label: str):
    now = time.time()
    history_dict[client_ip] = [t for t in history_dict[client_ip] if t > now - WINDOW_20_HOURS]
    if len(history_dict[client_ip]) >= max_limit:
        raise HTTPException(status_code=429, detail=f"Rate limit reached! Maximum {max_limit} {label} allowed per 20-hour window.")

def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    return forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")

# --- SCHEMAS ---
class SparksRequest(BaseModel):
    summary: str
    characters: str | None = ""

class RerollRequest(BaseModel):
    summary: str
    characters: str | None = ""

class SynthesizeRequest(BaseModel):
    summary: str
    characters: str | None = ""
    curated_sparks: list[str]
    enable_editor: bool = True
    deep_edit: bool = False

class ContinueChapterRequest(BaseModel):
    overall_premise: str
    characters: str | None = ""
    rolling_recap: str
    handoff_anchor: str
    next_beat: str
    curated_sparks: list[str]
    enable_editor: bool = True
    deep_edit: bool = False
    chapter_num: int = 2

# --- HELPERS ---
def handle_api_error(e: Exception, step_name: str):
    err_str = str(e)
    if "500" in err_str or "INTERNAL" in err_str or "overload" in err_str.lower():
        raise HTTPException(status_code=503, detail="Our AI cluster is experiencing high demand. Please wait 10 seconds and try again.")
    raise HTTPException(status_code=500, detail=f"{step_name} Error: {err_str}")

def clean_output(text: str) -> str:
    if not text: return ""
    cleaned = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"^<thought>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.replace(r"$\rightarrow$", "→").replace(r"\rightarrow", "→").strip()

def parse_sparks_to_list(raw_text: str) -> list[str]:
    sparks = []
    for line in raw_text.split("\n"):
        clean = line.strip().replace("**", "").replace("*", "")
        match = re.match(r"^\d+[\.\)\-]\s*(.+)", clean)
        if match and len(match.group(1)) > 10:
            sparks.append(match.group(1))
            
    if not sparks:
        for line in raw_text.split("\n"):
            clean = line.strip().replace("**", "").replace("*", "")
            if len(clean) > 20 and not clean.lower().startswith("here are"):
                clean = re.sub(r"^\d+[\.\)\-]?\s*", "", clean)
                sparks.append(clean)
                
    if not sparks:
        return ["The sparks engine stumbled. Please click the 🎲 button to reroll this catalyst."] * 10
        
    return sparks[:10]

def generate_fallback_recap(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[-3:]).strip() if len(sentences) > 3 else text.strip()

def extract_story_and_recap(text: str, fallback_text: str) -> tuple[str, str]:
    patterns = [r"(?:---|\*\*\*|===)\s*RECAP\s*(?:---|\*\*\*|===)", r"(?:\n|^)#+\s*RECAP:?", r"(?:\n|^)\*\*RECAP:\*\*", r"(?:\n|^)RECAP:\s*", r"(?:\n|^)SUMMARY OF EVENTS:\s*"]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            story = text[: match.start()].strip()
            recap = text[match.end():].strip()
            if story and recap: return story, recap
    return text.strip(), generate_fallback_recap(fallback_text)

def get_random_nouns(count=10) -> list[str]:
    for filename in ["noun2.txt", "nouns.txt"]:
        if (BASE_DIR / filename).exists():
            with open(BASE_DIR / filename, "r", encoding="utf-8") as f:
                nouns = [line.strip() for line in f if line.strip()]
            if nouns: return nouns if len(nouns) < count else random.sample(nouns, count)
    raise FileNotFoundError("Noun file not found!")

# --- ROUTES ---
@app.post("/api/sparks")
async def generate_sparks_endpoint(req: SparksRequest, request: Request):
    clean_sum, clean_chars = (req.summary or "").strip(), (req.characters or "").strip()
    if not clean_sum: raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    
    client_ip = get_client_ip(request)
    check_and_clean_limits(ip_reroll_history, client_ip, REROLL_LIMIT, "spark generations")
    try: nouns = get_random_nouns(10)
    except FileNotFoundError as e: raise HTTPException(status_code=404, detail=str(e))

    char_context = f"\nKey Characters: {clean_chars}" if clean_chars else ""
    step1_system = "You are a creative concept-blending engine.\n1. Extract implied themes, atmospheres, sensory textures, and emotions from the premise.\n2. Pair each provided noun with one of those derived concepts to craft 10 distinct, standalone sentences.\n3. STRICT ISOLATION: Do NOT connect the sentences into a story.\n4. Output ONLY the 10 numbered sentences (1 to 10)."
    step1_prompt = f"Premise Context: \"{clean_sum}\"{char_context}\n\n10 Nouns: {', '.join(nouns)}\n\nGenerate 10 standalone sentences:"

    try:
        res = client.models.generate_content(
            model=MODEL_NAME, contents=step1_prompt,
            config=types.GenerateContentConfig(system_instruction=step1_system, temperature=0.95, thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"), automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        )
        sparks_list = parse_sparks_to_list(clean_output(res.text or ""))
        ip_reroll_history[client_ip].append(time.time())
        return {"nouns": nouns, "sparks": sparks_list}
    except Exception as e: handle_api_error(e, "Spark Generation")


@app.post("/api/reroll-spark")
async def reroll_single_spark_endpoint(req: RerollRequest, request: Request):
    clean_sum, clean_chars = (req.summary or "").strip(), (req.characters or "").strip()
    if not clean_sum: raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    
    client_ip = get_client_ip(request)
    check_and_clean_limits(ip_reroll_history, client_ip, REROLL_LIMIT, "spark rerolls")
    noun = get_random_nouns(1)[0]
    char_context = f"\nCharacters: {clean_chars}" if clean_chars else ""
    prompt = f'Premise: "{clean_sum}\"{char_context}\nTarget Noun: {noun}\nGenerate exactly 1 standalone, vivid metaphor sentence fusing them:'

    try:
        res = client.models.generate_content(
            model=MODEL_NAME, contents=prompt,
            config=types.GenerateContentConfig(system_instruction="You are a creative concept-blending engine. Output exactly 1 single sentence with no numbers, bullet points, or quotes.", temperature=0.95, thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"), automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        )
        new_spark = re.sub(r"^\s*[\d\.\-\*]+\s*", "", clean_output(res.text or "")).strip()
        ip_reroll_history[client_ip].append(time.time())
        return {"spark": new_spark}
    except Exception as e: handle_api_error(e, "Reroll")


@app.post("/api/synthesize")
async def synthesize_story_endpoint(req: SynthesizeRequest, request: Request):
    clean_sum, clean_chars = (req.summary or "").strip(), (req.characters or "").strip()
    if not clean_sum: raise HTTPException(status_code=400, detail="Story premise cannot be empty.")
    if not req.curated_sparks: raise HTTPException(status_code=400, detail="Please select at least one spark.")

    client_ip = get_client_ip(request)
    check_and_clean_limits(ip_story_history, client_ip, STORY_LIMIT, "stories")

    sparks_formatted = "\n".join([f"- {s}" for s in req.curated_sparks])
    char_context = f"\nLocked Character Profiles:\n{clean_chars}" if clean_chars else ""

    drafter_system = "You are an imaginative fiction author. Write a bold, atmospheric first chapter draft blending the premise, characters, and selected inspiration sparks."
    drafter_prompt = f"Story Premise:\n{clean_sum}{char_context}\n\nCurated Inspiration Sparks:\n{sparks_formatted}\n\nTask:\nWrite a captivating, atmospheric Chapter 1 draft driven by the premise."

    try:
        draft_res = client.models.generate_content(
            model=MODEL_NAME, contents=drafter_prompt,
            config=types.GenerateContentConfig(system_instruction=drafter_system, temperature=0.95, thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"), automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        )
        raw_draft = clean_output(draft_res.text or "")
        if not raw_draft: raise ValueError("Draft step returned empty prose.")

        final_story = raw_draft
        recap = generate_fallback_recap(raw_draft)

        if req.enable_editor:
            editor_system = "You are a world-class literary editor.\n1. Polish the raw Chapter 1 draft: remove awkward or forced metaphors, and enhance prose rhythm.\n2. At the very end, append a concise 2-3 sentence rolling plot recap for future chapter continuity.\n\nFORMAT YOUR RESPONSE:\n[Polished Chapter Prose]\n---RECAP---\n[2-3 sentence rolling plot summary]"
            editor_prompt = f"Original Premise:\n{clean_sum}{char_context}\n\nRaw Draft:\n{raw_draft}\n\nTask: Output polished Chapter 1 and the rolling recap."
            try:
                editor_res = client.models.generate_content(
                    model=MODEL_NAME, contents=editor_prompt,
                    config=types.GenerateContentConfig(system_instruction=editor_system, temperature=0.9, thinking_config=types.ThinkingConfig(thinking_level="HIGH" if req.deep_edit else "MINIMAL"), automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
                )
                extracted_story, extracted_recap = extract_story_and_recap(clean_output(editor_res.text or ""), raw_draft)
                if extracted_story:
                    final_story = extracted_story
                    recap = extracted_recap
            except Exception: pass

        ip_story_history[client_ip].append(time.time())
        return {"story": final_story, "recap": recap, "chapter_num": 1}
    except Exception as e: handle_api_error(e, "Chapter 1 Synthesis")


@app.post("/api/continue-chapter")
async def continue_chapter_endpoint(req: ContinueChapterRequest, request: Request):
    clean_premise, clean_next_beat = (req.overall_premise or "").strip(), (req.next_beat or "").strip()
    clean_recap, clean_anchor, clean_chars = (req.rolling_recap or "").strip(), (req.handoff_anchor or "").strip(), (req.characters or "").strip()

    if not clean_next_beat: raise HTTPException(status_code=400, detail="Next scene direction cannot be empty.")
    if not req.curated_sparks: raise HTTPException(status_code=400, detail="Please select at least one spark.")

    client_ip = get_client_ip(request)
    check_and_clean_limits(ip_story_history, client_ip, STORY_LIMIT, "chapters")

    sparks_formatted = "\n".join([f"- {s}" for s in req.curated_sparks])
    char_context = f"\nLocked Character Profiles:\n{clean_chars}" if clean_chars else ""

    drafter_system = f"You are a master fiction author writing Chapter {req.chapter_num} of an ongoing story.\nCRITICAL RULES:\n1. DO NOT repeat, echo, or rephrase the words in the EXACT SCENE HANDOFF.\n2. Begin Chapter {req.chapter_num} immediately with the next action, spoken dialogue, or atmospheric shift.\n3. Advance the story using the Next Scene Direction while weaving in the curated catalyst sparks.\n4. Maintain consistent character voices, tense, and perspective."
    drafter_prompt = f"OVERALL STORY PREMISE:\n{clean_premise}{char_context}\n\nSTORY RECAP SO FAR:\n{clean_recap}\n\nEXACT SCENE HANDOFF (Where previous chapter ended):\n\"{clean_anchor}\"\n\nNEXT SCENE DIRECTION (Chapter {req.chapter_num}):\n{clean_next_beat}\n\nFRESH CATALYST SPARKS:\n{sparks_formatted}\n\nTask: Write a captivating draft for Chapter {req.chapter_num}."

    try:
        draft_res = client.models.generate_content(
            model=MODEL_NAME, contents=drafter_prompt,
            config=types.GenerateContentConfig(system_instruction=drafter_system, temperature=0.95, thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"), automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        )
        raw_draft = clean_output(draft_res.text or "")
        if not raw_draft: raise ValueError(f"Draft step for Chapter {req.chapter_num} returned empty text.")

        final_story = raw_draft
        updated_recap = generate_fallback_recap(raw_draft)

        if req.enable_editor:
            editor_system = f"You are a world-class literary editor.\n1. Polish the raw Chapter {req.chapter_num} draft: remove forced metaphors, tighten pacing, and ensure seamless continuity.\n2. At the very end, append an updated 3-4 sentence plot recap summarizing ONLY the events that occurred in Chapter {req.chapter_num}.\n\nFORMAT YOUR RESPONSE:\n[Polished Chapter Prose]\n---RECAP---\n[Chapter {req.chapter_num} specific summary]"
            editor_prompt = f"Premise: {clean_premise}\nPrevious Recap: {clean_recap}\nRaw Chapter {req.chapter_num} Draft:\n{raw_draft}\n\nTask: Polish Chapter {req.chapter_num} and provide its specific recap."
            try:
                editor_res = client.models.generate_content(
                    model=MODEL_NAME, contents=editor_prompt,
                    config=types.GenerateContentConfig(system_instruction=editor_system, temperature=0.9, thinking_config=types.ThinkingConfig(thinking_level="HIGH" if req.deep_edit else "MINIMAL"), automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
                )
                extracted_story, extracted_recap = extract_story_and_recap(clean_output(editor_res.text or ""), raw_draft)
                if extracted_story:
                    final_story = extracted_story
                    updated_recap = extracted_recap
            except Exception: pass

        ip_story_history[client_ip].append(time.time())
        return {"story": final_story, "recap": updated_recap, "chapter_num": req.chapter_num}
    except Exception as e: handle_api_error(e, f"Chapter {req.chapter_num} Continuation")


static_path = BASE_DIR / "static"
static_path.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")