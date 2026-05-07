import asyncio
import json
import logging

import numpy as np
import torch
import torchaudio
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings
from faster_whisper import WhisperModel
from scipy.signal import butter, sosfilt

logger = logging.getLogger(__name__)

# ── Model singletons ─────────────────────────────────────────────────────────

_WHISPER_MODEL = None
_DF_MODEL = None
_DF_STATE = None


def _get_whisper():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "int8_float16" if device == "cuda" else "int8"
        logger.info("Loading faster-whisper large-v3 on %s (compute=%s)", device, compute_type)
        _WHISPER_MODEL = WhisperModel(
            "large-v3",
            device=device,
            compute_type=compute_type,
            num_workers=1,
        )
        logger.info("faster-whisper ready")
    return _WHISPER_MODEL


def _get_df():
    global _DF_MODEL, _DF_STATE
    if _DF_MODEL is None:
        from df.enhance import init_df  # noqa: PLC0415

        _DF_MODEL, _DF_STATE, _ = init_df()
        if torch.cuda.is_available():
            _DF_MODEL = _DF_MODEL.to("cuda")
        logger.info("DeepFilterNet loaded on %s", "cuda" if torch.cuda.is_available() else "cpu")
    return _DF_MODEL, _DF_STATE


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _bandpass_radio(audio: np.ndarray, samplerate: int) -> np.ndarray:
    """4th-order Butterworth bandpass 300–3400 Hz — isolates radio voice band."""
    nyq = samplerate / 2.0
    sos = butter(4, [300.0 / nyq, min(3400.0 / nyq, 0.99)], btype="band", output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def _denoise(audio: np.ndarray, samplerate: int) -> np.ndarray:
    """Run DeepFilterNet noise suppression. Resamples to 48 kHz and back."""
    from df.enhance import enhance  # noqa: PLC0415

    model, df_state = _get_df()
    df_sr = df_state.sr()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # copy() prevents torch from sharing memory with the numpy array
    t = torch.from_numpy(audio.copy()).float().unsqueeze(0)  # [1, T]
    if samplerate != df_sr:
        t = torchaudio.functional.resample(t, samplerate, df_sr)
    t = t.to(device)

    enhanced = enhance(model, df_state, t)

    # detach from autograd graph, move to CPU before any further ops
    enhanced = enhanced.detach().cpu()
    if samplerate != df_sr:
        enhanced = torchaudio.functional.resample(enhanced, df_sr, samplerate)
    return enhanced.squeeze(0).cpu().numpy()


def _amplify_pcm(raw: bytes, db: float = 6.0) -> bytes:
    gain = 10 ** (db / 20)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    samples = np.clip(samples * gain, -32768, 32767)
    return samples.astype(np.int16).tobytes()


def _rms_bytes(raw: bytes) -> float:
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(samples ** 2)))


# ── UDP protocol ──────────────────────────────────────────────────────────────

class _UDPProtocol(asyncio.DatagramProtocol):
    def __init__(self, queue: asyncio.Queue) -> None:
        self._queue = queue

    def datagram_received(self, data: bytes, addr: tuple) -> None:
        logger.debug("UDP datagram from %s: %d bytes", addr, len(data))
        self._queue.put_nowait(data)

    def error_received(self, exc: Exception) -> None:
        logger.warning("UDP transport error: %s", exc)


# ── Consumers ─────────────────────────────────────────────────────────────────

class LiveTranscribeConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, bytes_data=None, text_data=None):
        pass  # mic consumer removed — kept as stub


_RMS_THRESHOLD = 0.003


class RadioTranscribeConsumer(AsyncWebsocketConsumer):
    """Receives UDP PCM from Marine VHF radio, streams audio to browser,
    denoises via DeepFilterNet, transcribes via faster-whisper on GPU."""

    async def connect(self):
        await self.accept()
        self._active = True
        self._task = asyncio.ensure_future(self._loop())

    async def disconnect(self, close_code):
        self._active = False
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    async def receive(self, bytes_data=None, text_data=None):
        pass  # read-only stream

    async def _loop(self):
        udp_ip     = getattr(settings, "RADIO_UDP_IP",      "0.0.0.0")
        udp_port   = int(getattr(settings, "RADIO_UDP_PORT",    5005))
        samplerate = int(getattr(settings, "RADIO_SAMPLE_RATE", 16_000))

        bytes_per_sec = samplerate * 2  # int16 mono

        # 30 s = Whisper's native context window; aligning to it gives best accuracy.
        # 2 s minimum guarantees we have at least a short complete phrase.
        # 1 s silence gate: PTT drop causes near-instant squelch, so 1 s reliably
        # signals end-of-transmission while not cutting natural mid-sentence pauses
        # (which are typically 0.2–0.4 s on radio).
        max_buf_bytes = bytes_per_sec * 30
        min_buf_bytes = bytes_per_sec * 2
        silence_drain = bytes_per_sec * 1

        queue: asyncio.Queue[bytes] = asyncio.Queue()
        loop = asyncio.get_event_loop()
        transport, _ = await loop.create_datagram_endpoint(
            lambda: _UDPProtocol(queue),
            local_addr=(udp_ip, udp_port),
        )

        logger.info("RadioTranscribeConsumer: UDP socket bound on %s:%s", udp_ip, udp_port)
        await self.send(text_data=json.dumps({"status": "connected", "source": "udp"}))

        audio_buffer: list[bytes] = []
        silence_accum = 0

        try:
            while self._active:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Browser gets amplified audio for real-time monitoring.
                # Transcription buffer stores the original (non-amplified) bytes so
                # DeepFilterNet sees correct signal levels for its noise estimation.
                amplified = _amplify_pcm(data)
                try:
                    await self.send(bytes_data=amplified)
                except Exception as exc:
                    logger.warning("WebSocket send failed, closing loop: %s", exc)
                    break
                audio_buffer.append(data)

                # Silence gate is checked on amplified signal for better sensitivity.
                if _rms_bytes(amplified) < _RMS_THRESHOLD:
                    silence_accum += len(amplified)
                else:
                    silence_accum = 0

                total = sum(len(b) for b in audio_buffer)
                end_of_tx = silence_accum >= silence_drain and total >= min_buf_bytes
                if total >= max_buf_bytes or end_of_tx:
                    raw = b"".join(audio_buffer)
                    audio_buffer = []
                    silence_accum = 0
                    try:
                        text = await asyncio.to_thread(_transcribe_bytes, raw, samplerate)
                    except Exception as exc:
                        logger.error("Transcription error: %s", exc)
                        text = ""
                    if text and self._active:
                        try:
                            await self.send(text_data=json.dumps({"text": text, "source": "udp"}))
                        except Exception as exc:
                            logger.warning("WebSocket text send failed, closing loop: %s", exc)
                            break
        except asyncio.CancelledError:
            raise
        finally:
            transport.close()


# ── Transcription ─────────────────────────────────────────────────────────────

def _transcribe_bytes(raw: bytes, samplerate: int) -> str:
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < _RMS_THRESHOLD:
        logger.debug("Skipping silent chunk (RMS %.5f)", rms)
        return ""

    # 1. Bandpass 300–3400 Hz: discards RF artefacts and sub-bass rumble
    #    outside the radio voice band before denoising.
    audio = _bandpass_radio(audio, samplerate)

    # 2. Denoise at natural (non-amplified) signal levels so DeepFilterNet
    #    can correctly estimate the noise floor.
    try:
        audio = _denoise(audio, samplerate)
    except Exception as exc:
        logger.warning("DeepFilterNet failed, proceeding without denoise: %s", exc)

    rms_post = float(np.sqrt(np.mean(audio ** 2)))
    if rms_post < _RMS_THRESHOLD:
        logger.debug("Post-denoise silence (RMS %.5f), skipping", rms_post)
        return ""

    # 3. Resample to 16 kHz if needed — faster-whisper always expects 16 kHz float32.
    target_sr = 16_000
    if samplerate != target_sr:
        t = torch.from_numpy(audio).unsqueeze(0)
        t = torchaudio.functional.resample(t, samplerate, target_sr)
        audio = t.squeeze(0).numpy()

    model = _get_whisper()
    segments, _ = model.transcribe(
        audio,
        language=None,                    # auto-detect EN / IT per chunk
        beam_size=5,
        vad_filter=True,                  # Silero VAD: skip non-speech frames within chunk
        vad_parameters=dict(
            min_silence_duration_ms=500,  # 500 ms gap required to split segments
            speech_pad_ms=200,            # keep 200 ms of context around each segment
        ),
        temperature=0.0,                  # greedy decoding — deterministic and fastest
        condition_on_previous_text=False, # no cross-chunk context → prevents hallucination
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
    )

    parts = [s.text.strip() for s in segments if s.no_speech_prob < 0.6]
    return " ".join(p for p in parts if p)
