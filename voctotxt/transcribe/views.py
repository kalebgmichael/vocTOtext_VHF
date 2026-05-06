import os
import tempfile

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.decorators import permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from .consumers import _get_whisper


def _whisper_transcribe_file(path: str, language: str | None = None) -> dict:
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
    if "audio_file" not in request.FILES:
        return Response({"error": "No audio file provided"}, status=status.HTTP_400_BAD_REQUEST)

    audio_file = request.FILES["audio_file"]
    language = request.data.get("language") or None

    suffix = os.path.splitext(audio_file.name)[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        for chunk in audio_file.chunks():
            tmp.write(chunk)
        temp_path = tmp.name

    try:
        result = _whisper_transcribe_file(temp_path, language)
        return Response(
            {
                "filename": audio_file.name,
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
    finally:
        if os.path.exists(temp_path):  # noqa: PTH110
            os.remove(temp_path)  # noqa: PTH107


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
