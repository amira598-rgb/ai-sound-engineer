from pathlib import Path
import uuid, json, base64, os, urllib.request
import soundfile as sf
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from audio_engine import analyze_audio, process_audio

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"
WORK = BASE / "work"
STATIC.mkdir(exist_ok=True)
WORK.mkdir(exist_ok=True)

app = FastAPI(title="AI SOUND ENGINEER", version="2.1.0")

# Set ALLOWED_ORIGINS to a comma-separated list in production.
origins = os.getenv(
    "ALLOWED_ORIGINS",
    "https://amira598-rgb.github.io,http://127.0.0.1:8000,http://localhost:8000"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in origins.split(",") if x.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC), name="static")

AUDIO_EXTS = {".wav", ".flac", ".aiff", ".aif", ".ogg"}
MAX_AUDIO = 150 * 1024 * 1024
MAX_IMAGE = 12 * 1024 * 1024
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:3b")


def safe_json_from_model(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Vision model did not return JSON")
    return json.loads(text[start:end + 1])


def ollama_vision(image_bytes):
    prompt = """
You are a cautious room-photo observer for a voice recording assistant.
Describe ONLY visible evidence. Do not claim exact dimensions, RT60,
hidden materials, microphone position, or measured acoustics.
Return ONLY JSON:
{
 "room_type_guess":"short string",
 "visible_surfaces":["short visible observations"],
 "soft_furnishings":"low|medium|high|unknown",
 "hard_surface_visibility":"low|medium|high|unknown",
 "visual_reflection_risk":"low|medium|high|unknown",
 "confidence":0.0,
 "natural_voice_advice":"one concise sentence",
 "limitations":"one concise sentence"
}
Use confidence <= 0.45 because a photo is not an acoustic measurement.
"""
    payload = json.dumps({
        "model": VISION_MODEL,
        "prompt": prompt,
        "images": [base64.b64encode(image_bytes).decode("ascii")],
        "stream": False,
        "format": "json"
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL.rstrip("/") + "/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=90) as response:
        result = json.loads(response.read().decode("utf-8"))
    parsed = safe_json_from_model(result.get("response", ""))
    parsed["analysis_method"] = "local_vision_model"
    parsed["model"] = VISION_MODEL
    parsed["confidence"] = min(float(parsed.get("confidence", 0.35)), 0.45)
    parsed["photo_only_warning"] = (
        "Фото показывает видимые признаки, но не измеряет акустику, эхо или RT60."
    )
    return parsed


def visual_fallback(image_bytes):
    # Honest fallback: validate and report that semantic image analysis is unavailable.
    Image.open(BytesIO(image_bytes)).verify()
    return {
        "room_type_guess": "не определён",
        "visible_surfaces": [],
        "soft_furnishings": "unknown",
        "hard_surface_visibility": "unknown",
        "visual_reflection_risk": "unknown",
        "confidence": 0.0,
        "natural_voice_advice": (
            "Фото загружено, но vision-модель недоступна. "
            "Установите и запустите Ollama с совместимой моделью."
        ),
        "analysis_method": "vision_unavailable",
        "photo_only_warning": (
            "Без vision-модели объекты и материалы не распознаются. "
            "Фото в любом случае не заменяет акустическое измерение."
        )
    }


def analyze_room_photo(image_bytes):
    try:
        im = Image.open(BytesIO(image_bytes))
        if im.format not in {"JPEG", "PNG", "WEBP"}:
            raise HTTPException(415, "Используйте JPG, PNG или WEBP.")
        im.verify()
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(422, "Не удалось прочитать изображение.")
    try:
        return ollama_vision(image_bytes)
    except Exception:
        return visual_fallback(image_bytes)


@app.get("/", response_class=HTMLResponse)
def home():
    index = STATIC / "index.html"
    if not index.is_file():
        raise HTTPException(500, "Не найден static/index.html")
    return index.read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "engine": "FastAPI",
        "vision_model": VISION_MODEL,
        "ollama_url": OLLAMA_URL,
        "version": "2.1.0"
    }


@app.post("/api/room/analyze")
async def room_analyze(file: UploadFile = File(...)):
    data = await file.read(MAX_IMAGE + 1)
    if len(data) > MAX_IMAGE:
        raise HTTPException(413, "Изображение больше 12 МБ.")
    if not data:
        raise HTTPException(400, "Получен пустой файл.")
    return analyze_room_photo(data)


async def read_audio_upload(file: UploadFile):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in AUDIO_EXTS:
        raise HTTPException(415, "Используйте WAV, FLAC, AIFF или OGG.")
    data = await file.read(MAX_AUDIO + 1)
    if len(data) > MAX_AUDIO:
        raise HTTPException(413, "Файл больше 150 МБ.")
    if not data:
        raise HTTPException(400, "Аудиофайл пуст.")
    return suffix, data


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix, data = await read_audio_upload(file)
    temp = WORK / f"{uuid.uuid4().hex}{suffix}"
    try:
        temp.write_bytes(data)
        audio, sr = sf.read(temp, dtype="float32", always_2d=True)
        return analyze_audio(audio, sr)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"Не удалось прочитать аудио: {exc}")
    finally:
        temp.unlink(missing_ok=True)


@app.post("/api/process")
async def process(
    file: UploadFile = File(...),
    profile: str = Form("voice"),
    mode: str = Form("hybrid"),
    denoise: float = Form(0.2),
    bass: float = Form(0.0),
    mid: float = Form(0.0),
    treble: float = Form(0.0),
    compression: float = Form(0.2),
    target_db: float = Form(-18.0),
    room_profile: str = Form("{}")
):
    suffix, data = await read_audio_upload(file)
    try:
        room = json.loads(room_profile or "{}")
        if not isinstance(room, dict):
            room = {}
    except json.JSONDecodeError:
        room = {}

    source = WORK / f"{uuid.uuid4().hex}{suffix}"
    output = WORK / f"{uuid.uuid4().hex}_processed.wav"
    try:
        source.write_bytes(data)
        audio, sr = sf.read(source, dtype="float32", always_2d=True)
        result, report = process_audio(
            audio, sr, profile=profile, mode=mode,
            denoise=denoise, bass=bass, mid=mid, treble=treble,
            compression=compression, target_db=target_db,
            room_profile=room
        )
        sf.write(output, result, sr, subtype="PCM_24")
        return {
            "download_url": f"/api/download/{output.name}",
            "filename": "AI_SOUND_ENGINEER_natural_voice.wav",
            "report": report
        }
    except HTTPException:
        raise
    except Exception as exc:
        output.unlink(missing_ok=True)
        raise HTTPException(422, f"Ошибка обработки: {exc}")
    finally:
        source.unlink(missing_ok=True)


@app.get("/api/download/{name}")
def download(name: str):
    if Path(name).name != name or not name.endswith("_processed.wav"):
        raise HTTPException(404, "Файл не найден.")
    path = WORK / name
    if not path.is_file():
        raise HTTPException(404, "Файл не найден.")
    return FileResponse(
        path, media_type="audio/wav",
        filename="AI_SOUND_ENGINEER_natural_voice.wav"
    )
