import glob
import logging
import os
import tempfile

import numpy as np
import torch
import torchaudio
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.decorators import permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from .consumers import _RMS_THRESHOLD
from .consumers import _bandpass_radio
from .consumers import _denoise
from .consumers import _get_whisper

logger = logging.getLogger(__name__)

# Fixed upload path — Postman saves here, Record button reads from here.
_UPLOAD_DIR    = tempfile.gettempdir()
_UPLOAD_PREFIX = "vhf_radio_latest"


def _stored_upload_path() -> str | None:
    """Return path of latest uploaded file, or None if none exists."""
    matches = glob.glob(os.path.join(_UPLOAD_DIR, f"{_UPLOAD_PREFIX}.*"))
    return matches[0] if matches else None


def _transcribe_wav_pipeline(path: str, language: str | None = None) -> dict:
    """Read audio → bandpass 300-3400 Hz → DeepFilterNet denoise → Whisper.

    Supports .pcm (raw int16 mono) and any torchaudio-readable format (.wav etc.).
    Mirrors the RadioTranscribeConsumer pipeline.
    """
    from django.conf import settings as django_settings  # noqa: PLC0415

    if path.lower().endswith(".pcm"):
        samplerate = int(getattr(django_settings, "RADIO_SAMPLE_RATE", 16_000))
        raw = open(path, "rb").read()  # noqa: WPS515
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        waveform, samplerate = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        audio = waveform.squeeze(0).numpy().astype(np.float32)

    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < _RMS_THRESHOLD:
        return {"text": "", "language": None, "language_probability": 0.0, "duration": 0.0, "segments": []}

    audio = _bandpass_radio(audio, samplerate)

    try:
        audio = _denoise(audio, samplerate)
    except Exception as exc:
        logger.warning("DeepFilterNet failed, proceeding without denoise: %s", exc)

    rms_post = float(np.sqrt(np.mean(audio ** 2)))
    if rms_post < _RMS_THRESHOLD:
        return {"text": "", "language": None, "language_probability": 0.0, "duration": 0.0, "segments": []}

    target_sr = 16_000
    if samplerate != target_sr:
        t = torch.from_numpy(audio).unsqueeze(0)
        t = torchaudio.functional.resample(t, samplerate, target_sr)
        audio = t.squeeze(0).cpu().numpy()

    model = _get_whisper()
    segments, info = model.transcribe(
        audio,
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
        temperature=0.0,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
    )
    seg_list = [
        {"start": s.start, "end": s.end, "text": s.text.strip()}
        for s in segments
        if s.no_speech_prob < 0.6
    ]
    full_text = " ".join(s["text"] for s in seg_list if s["text"])
    return {
        "text": full_text,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": seg_list,
    }


def _whisper_transcribe_file(path: str, language: str | None = None) -> dict:
    """Direct Whisper transcription without radio preprocessing — used for browser chunks."""
    model = _get_whisper()
    segments, info = model.transcribe(
        path,
        language=language or None,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
        temperature=0.0,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
    )
    seg_list = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
    full_text = " ".join(s["text"] for s in seg_list if s["text"])
    return {
        "text": full_text,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": seg_list,
    }


@api_view(["POST"])
@permission_classes([AllowAny])
def transcribe_upload(request):
    """Save uploaded audio file. Does not transcribe — call /record/ to transcribe."""
    if "audio_file" not in request.FILES:
        return Response({"error": "No audio file provided"}, status=status.HTTP_400_BAD_REQUEST)

    audio_file = request.FILES["audio_file"]
    suffix = os.path.splitext(audio_file.name)[1] or ".wav"

    # Remove any previous upload before saving new one.
    for old in glob.glob(os.path.join(_UPLOAD_DIR, f"{_UPLOAD_PREFIX}.*")):
        os.remove(old)

    dest = os.path.join(_UPLOAD_DIR, f"{_UPLOAD_PREFIX}{suffix}")
    with open(dest, "wb") as f:
        for chunk in audio_file.chunks():
            f.write(chunk)

    return Response({"status": "uploaded", "filename": audio_file.name}, status=status.HTTP_200_OK)


@api_view(["POST"])
@permission_classes([AllowAny])
def transcribe_record(request):
    """Transcribe the last file saved via /upload/. Called by the Record button."""
    path = _stored_upload_path()
    if not path:
        return Response(
            {"error": "No uploaded file found. Upload a file via POST /api/transcribe/upload/ first."},
            status=status.HTTP_404_NOT_FOUND,
        )

    language = request.data.get("language") or None
    try:
        result = _transcribe_wav_pipeline(path, language)
        return Response(
            {
                "transcription": result["text"],
                "language": result["language"],
                "language_probability": result["language_probability"],
                "duration": result["duration"],
                "segments": result["segments"],
            },
            status=status.HTTP_200_OK,
        )
    except Exception as e:  # noqa: BLE001
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["POST"])
@permission_classes([AllowAny])
def transcribe_live_chunk(request):
    if "audio_chunk" not in request.FILES:
        return Response({"error": "No audio chunk provided"}, status=status.HTTP_400_BAD_REQUEST)

    chunk = request.FILES["audio_chunk"]
    language = request.data.get("language") or None

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        for data in chunk.chunks():
            tmp.write(data)
        tmp_path = tmp.name

    try:
        result = _whisper_transcribe_file(tmp_path, language)
        return Response({"text": result["text"], "language": result["language"]})
    except Exception as e:  # noqa: BLE001
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    finally:
        if os.path.exists(tmp_path):  # noqa: PTH110
            os.remove(tmp_path)  # noqa: PTH107
