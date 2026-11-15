from pathlib import Path
import uuid, json, base64, os, urllib.request, urllib.error
import soundfile as sf
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from audio_engine import analyze_audio, process_audio

BASE = Path(__file__).resolve().parent
WORK = BASE / "work"
WORK.mkdir(exist_ok=True)
app = FastAPI(title="AI SOUND ENGINEER", version="2.0.0")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

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
    return json.loads(text[start:end+1])

def ollama_vision(image_bytes):
    prompt = """
You are a cautious room-photo observer for a voice recording assistant.
Describe ONLY visible evidence. Do not claim to know exact room dimensions,
RT60, hidden materials, microphone position, or measured acoustics.
Return ONLY JSON with this schema:
{
 "room_type_guess": "short string",
 "visible_surfaces": ["short visible observations"],
 "soft_furnishings": "low|medium|high|unknown",
 "hard_surface_visibility": "low|medium|high|unknown",
 "visual_reflection_risk": "low|medium|high|unknown",
 "confidence": 0.0,
 "natural_voice_advice": "one concise sentence",
 "limitations": "one concise sentence"
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
        result = json.loads(response.read().decode())
    parsed = safe_json_from_model(result.get("response", ""))
    parsed["analysis_method"] = "local_vision_model"
    parsed["model"] = VISION_MODEL
    parsed["confidence"] = min(float(parsed.get("confidence", 0.35)), 0.45)
    parsed["photo_only_warning"] = (
        "Фото показывает видимые признаки, но не измеряет акустику. "
        "Фактическое эхо и время реверберации по одному снимку определить нельзя."
    )
    return parsed

def visual_fallback(image_bytes):
    # Basic image statistics only. They cannot identify materials or room acoustics.
    im = Image.open(BytesIO(image_bytes)).convert("RGB")
    im.thumbnail((512, 512))
    import numpy as np
    a = np.asarray(im, dtype=np.float32) / 255.0
    gray = a.mean(axis=2)
    edge_x = np.abs(np.diff(gray, axis=1)).mean()
    edge_y = np.abs(np.diff(gray, axis=0)).mean()
    texture = float(min(1.0, (edge_x + edge_y) * 4))
    brightness = float(gray.mean())
    return {
        "room_type_guess": "не определён",
        "visible_surfaces": [],
        "soft_furnishings": "unknown",
        "hard_surface_visibility": "unknown",
        "visual_reflection_risk": "unknown",
        "confidence": 0.0,
        "natural_voice_advice": (
            "Фото загружено, но локальная модель распознавания изображений не подключена. "
            "Акустический профиль не будет автоматически менять звук."
        ),
        "analysis_method": "basic_image_statistics_only",
        "image_brightness": round(brightness, 3),
        "image_texture_score": round(texture, 3),
        "photo_only_warning": (
            "Статистика пикселей не распознаёт материалы и не определяет акустику. "
            "Для смыслового анализа установите Ollama и vision-модель."
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
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError, OSError):
        return visual_fallback(image_bytes)

@app.get("/", response_class=HTMLResponse)
def home():
    return (BASE / "static" / "index.html").read_text(encoding="utf-8")

@app.get("/api/health")
def health():
    return {
        "status": "ok", "engine": "local",
        "vision_model_configured": True,
        "vision_model": VISION_MODEL,
        "ollama_url": OLLAMA_URL,
        "version": "2.0.0"
    }

@app.post("/api/room/analyze")
async def room_analyze(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_IMAGE:
        raise HTTPException(413, "Изображение больше 12 МБ.")
    return analyze_room_photo(data)

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in AUDIO_EXTS:
        raise HTTPException(415, "Используйте WAV, FLAC, AIFF или OGG.")
    data = await file.read()
    if len(data) > MAX_AUDIO:
        raise HTTPException(413, "Файл больше 150 МБ.")
    temp = WORK / f"{uuid.uuid4().hex}{suffix}"
    temp.write_bytes(data)
    try:
        audio, sr = sf.read(temp, dtype="float32", always_2d=True)
        return analyze_audio(audio, sr)
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
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in AUDIO_EXTS:
        raise HTTPException(415, "Используйте WAV, FLAC, AIFF или OGG.")
    data = await file.read()
    if len(data) > MAX_AUDIO:
        raise HTTPException(413, "Файл больше 150 МБ.")
    try:
        room = json.loads(room_profile or "{}")
    except json.JSONDecodeError:
        room = {}
    source = WORK / f"{uuid.uuid4().hex}{suffix}"
    output = WORK / f"{uuid.uuid4().hex}_processed.wav"
    source.write_bytes(data)
    try:
        audio, sr = sf.read(source, dtype="float32", always_2d=True)
        result, report = process_audio(
            audio, sr, profile=profile, mode=mode,
            denoise=denoise, bass=bass, mid=mid, treble=treble,
            compression=compression, target_db=target_db,
            room_profile=room
        )
        sf.write(output, result, sr, subtype="PCM_24")
        return {"download_url": f"/api/download/{output.name}",
                "filename": "AI_SOUND_ENGINEER_natural_voice.wav",
                "report": report}
    except Exception as exc:
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
    return FileResponse(path, media_type="audio/wav",
                        filename="AI_SOUND_ENGINEER_natural_voice.wav")
