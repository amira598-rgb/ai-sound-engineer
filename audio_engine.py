"""Transparent, offline DSP chain. This is an automatic assistant, not a claim of perfect mastering."""
import numpy as np
from scipy import signal

PROFILES = {
    "voice": {"hp": 75, "bass": -1.0, "mid": -1.0, "treble": 1.5, "comp": 0.35},
    "music": {"hp": 28, "bass": 0.0, "mid": 0.0, "treble": 0.0, "comp": 0.18},
    "room": {"hp": 45, "bass": -1.5, "mid": -1.0, "treble": 0.5, "comp": 0.25},
    "outdoor": {"hp": 100, "bass": -2.0, "mid": 0.0, "treble": 0.5, "comp": 0.3},
    "podcast": {"hp": 80, "bass": -1.0, "mid": -1.5, "treble": 1.0, "comp": 0.45},
}

def _db(x):
    return 20.0 * np.log10(max(float(x), 1e-9))

def _peak(x):
    return float(np.max(np.abs(x))) if x.size else 0.0

def _rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0

def _stereo_safe(audio):
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] < 16:
        raise ValueError("Аудиофайл слишком короткий или имеет неверную структуру.")
    if not np.isfinite(x).all():
        x = np.nan_to_num(x)
    return np.clip(x, -4.0, 4.0)

def analyze_audio(audio, sr):
    x = _stereo_safe(audio)
    mono = np.mean(x, axis=1)
    peak = _peak(x)
    rms = _rms(mono)
    duration = len(x) / sr
    # Broad spectral balance estimate; informative, not a room measurement.
    freqs, psd = signal.welch(mono, sr, nperseg=min(8192, len(mono)))
    bands = [(20, 250), (250, 4000), (4000, min(20000, sr/2))]
    energy = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        energy.append(float(np.sum(psd[mask])) if np.any(mask) else 0.0)
    total = sum(energy) + 1e-12
    return {
        "duration_seconds": round(duration, 2),
        "sample_rate": int(sr),
        "channels": int(x.shape[1]),
        "peak_dbfs": round(_db(peak), 2),
        "rms_dbfs": round(_db(rms), 2),
        "clipping_samples": int(np.sum(np.abs(x) >= 0.999)),
        "band_energy_percent": {
            "low": round(100*energy[0]/total, 1),
            "mid": round(100*energy[1]/total, 1),
            "high": round(100*energy[2]/total, 1)
        },
        "note": "Частотный анализ файла не измеряет акустику комнаты. Для этого нужен тестовый сигнал и микрофон."
    }

def _sos_filter(x, sr, kind, freq, gain_db=None, q=0.8):
    nyq = sr / 2
    if kind == "highpass":
        if freq >= nyq: return x
        sos = signal.butter(2, freq, btype="highpass", fs=sr, output="sos")
    elif kind == "low_shelf":
        # Gentle first-order shelf approximation.
        sos = signal.butter(1, min(freq, nyq*0.95), btype="lowpass", fs=sr, output="sos")
        low = signal.sosfiltfilt(sos, x, axis=0)
        return x + (10**(gain_db/20)-1) * low
    elif kind == "high_shelf":
        sos = signal.butter(1, min(freq, nyq*0.95), btype="highpass", fs=sr, output="sos")
        high = signal.sosfiltfilt(sos, x, axis=0)
        return x + (10**(gain_db/20)-1) * high
    elif kind == "peak":
        # Peaking EQ via a stable biquad.
        A = 10**(gain_db/40)
        w0 = 2*np.pi*freq/sr
        alpha = np.sin(w0)/(2*q)
        b = [1+alpha*A, -2*np.cos(w0), 1-alpha*A]
        a = [1+alpha/A, -2*np.cos(w0), 1-alpha/A]
        return signal.lfilter(b, a, x, axis=0).astype(np.float32)
    else:
        return x
    return signal.sosfiltfilt(sos, x, axis=0).astype(np.float32)

def _spectral_gate(x, sr, amount):
    """Conservative spectral gate using an initial noise estimate; best when the head contains room tone."""
    if amount <= 0.001 or len(x) < 2048:
        return x
    nperseg = min(2048, max(512, 2**int(np.floor(np.log2(min(len(x), 2048))))))
    hop = nperseg // 4
    out = np.empty_like(x)
    for ch in range(x.shape[1]):
        _, _, z = signal.stft(x[:, ch], fs=sr, nperseg=nperseg,
                              noverlap=nperseg-hop, boundary="zeros")
        # Estimate noise from the quietest 20% of frame magnitudes, robust to brief opening transients.
        mag = np.abs(z)
        frame_energy = np.mean(mag, axis=0)
        count = max(1, int(len(frame_energy)*0.2))
        quiet_idx = np.argsort(frame_energy)[:count]
        noise = np.median(mag[:, quiet_idx], axis=1, keepdims=True)
        threshold = noise * (1.5 + 5.0*amount)
        floor = 1.0 - 0.88*amount
        gain = np.maximum(floor, 1.0 - amount*np.clip(threshold/(mag+1e-8), 0, 1))
        _, rec = signal.istft(z*gain, fs=sr, nperseg=nperseg,
                              noverlap=nperseg-hop, input_onesided=True,
                              boundary=True)
        if len(rec) < len(x):
            rec = np.pad(rec, (0, len(x)-len(rec)))
        out[:, ch] = rec[:len(x)]
    return out.astype(np.float32)

def _compress(x, sr, amount):
    if amount <= 0.001:
        return x
    # Smooth peak envelope, linked across channels.
    detector = np.max(np.abs(x), axis=1)
    win = max(1, int(sr*0.015))
    kernel = np.ones(win, dtype=np.float32)/win
    env = np.sqrt(np.convolve(detector**2, kernel, mode="same") + 1e-10)
    threshold = 10**(-18/20)
    ratio = 1.0 + 3.0*amount
    env_db = 20*np.log10(env + 1e-9)
    over = np.maximum(0, env_db - 20*np.log10(threshold))
    gain_db = -over*(1-1/ratio)
    # Smooth gain to reduce modulation artifacts.
    smooth = max(1, int(sr*0.04))
    gain_db = np.convolve(gain_db, np.ones(smooth)/smooth, mode="same")
    gain = (10**(gain_db/20)).astype(np.float32)
    return x * gain[:, None]

def process_audio(audio, sr, profile="voice", mode="hybrid", denoise=0.35,
                  bass=0, mid=0, treble=0, compression=0.25, target_db=-18, room_profile=None):
    x = _stereo_safe(audio)
    if sr < 8000:
        raise ValueError("Частота дискретизации ниже 8 кГц не поддерживается.")
    p = PROFILES.get(profile, PROFILES["voice"])
    if mode == "automatic":
        bass, mid, treble = p["bass"], p["mid"], p["treble"]
        compression = p["comp"]
    elif mode != "hybrid":
        raise ValueError("Неизвестный режим управления.")

    denoise = float(np.clip(denoise, 0, 0.8))
    compression = float(np.clip(compression, 0, 1))
    target_db = float(np.clip(target_db, -24, -10))
    bass, mid, treble = [float(np.clip(v, -6, 6)) for v in (bass, mid, treble)]

    # Photo-only room context may apply only a tiny, cautious tonal correction.
    # It must never be treated as measured RT60 or as proof of reverberation.
    room_note = "Фото помещения не использовано."
    photo_confidence = 0.0
    if isinstance(room_profile, dict):
        photo_confidence = float(np.clip(room_profile.get("confidence", 0.0) or 0.0, 0.0, 0.45))
        risk = str(room_profile.get("visual_reflection_risk", "unknown")).lower()
        soft = str(room_profile.get("soft_furnishings", "unknown")).lower()
        method = room_profile.get("analysis_method", "")
        if method == "local_vision_model" and photo_confidence >= 0.25:
            # Subtle changes only. The image cannot reveal the recorded voice's actual room response.
            if risk == "high" and soft in ("low", "unknown"):
                treble = float(np.clip(treble - 0.5, -6, 6))
                room_note = "На фото видны признаки потенциально отражающих поверхностей; применена очень мягкая коррекция верха. Это не удаляет эхо."
            elif soft == "high":
                room_note = "На фото заметны мягкие предметы; дополнительная коррекция помещения не применялась."
            else:
                room_note = "Фото проанализировано; данных недостаточно для уверенной коррекции, применена нейтральная обработка."
        else:
            room_note = "Нет надёжного визуального акустического профиля; фото не меняло звук."
    before_peak = _peak(x)
    before_rms = _rms(x)
    x = _spectral_gate(x, sr, denoise)
    x = _sos_filter(x, sr, "highpass", p["hp"])
    x = _sos_filter(x, sr, "low_shelf", 140, bass)
    x = _sos_filter(x, sr, "peak", 1200, mid, q=0.7)
    x = _sos_filter(x, sr, "high_shelf", 6500, treble)
    x = _compress(x, sr, compression)

    # RMS-based level matching, followed by a true sample-peak ceiling.
    current = _rms(x)
    if current > 1e-8:
        desired = 10**(target_db/20)
        x *= min(10.0, desired/current)
    peak = _peak(x)
    ceiling = 10**(-1.0/20)
    if peak > ceiling:
        x *= ceiling/peak
    x = np.clip(x, -1.0, 1.0).astype(np.float32)

    report = {
        "profile": profile,
        "mode": mode,
        "sample_rate": int(sr),
        "duration_seconds": round(len(x)/sr, 2),
        "peak_before_dbfs": round(_db(before_peak), 2),
        "rms_before_dbfs": round(_db(before_rms), 2),
        "peak_after_dbfs": round(_db(_peak(x)), 2),
        "rms_after_dbfs": round(_db(_rms(x)), 2),
        "settings": {
            "noise_reduction": round(denoise, 2),
            "bass_db": bass, "mid_db": mid, "treble_db": treble,
            "compression": compression, "target_rms_dbfs": target_db
        },
        "photo_room_context": room_note,
        "photo_confidence": round(photo_confidence, 2),
        "limitations": [
            "Фотография не измеряет акустику, размеры помещения, RT60 или фактический уровень реверберации.",
            "Для естественного голоса фото используется только как слабый контекст; сильное удаление эха без анализа записи не выполняется.",
            "Автооценка шума является приближённой; для лучшего результата начало записи должно содержать фоновый шум без речи.",
            "Эквализация по профилю не является измерением АЧХ помещения.",
            "Результат следует прослушать до публикации; алгоритм не гарантирует профессиональный мастеринг для любого материала."
        ]
    }
    return x, report
