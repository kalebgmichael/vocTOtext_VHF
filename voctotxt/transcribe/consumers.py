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

# Computed once at import time — avoids repeated driver queries in hot paths.
_CUDA = torch.cuda.is_available()

# ── Model singletons ─────────────────────────────────────────────────────────

_WHISPER_MODEL = None
_DF_MODEL = None
_DF_STATE = None


def _get_whisper():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        device = "cuda" if _CUDA else "cpu"
        compute_type = "int8_float16" if _CUDA else "int8"
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
        if _CUDA:
            _DF_MODEL = _DF_MODEL.to("cuda")
        logger.info("DeepFilterNet loaded on %s", "cuda" if _CUDA else "cpu")
    return _DF_MODEL, _DF_STATE


# ── Audio helpers ─────────────────────────────────────────────────────────────

# Cached per samplerate — butter() is pure-math but non-trivial; no reason to
# recompute on every transcription call when samplerate never changes at runtime.
_BANDPASS_SOS: dict[int, np.ndarray] = {}


def _bandpass_radio(audio: np.ndarray, samplerate: int) -> np.ndarray:
    """4th-order Butterworth bandpass 300–3400 Hz — isolates radio voice band."""
    if samplerate not in _BANDPASS_SOS:
        nyq = samplerate / 2.0
        _BANDPASS_SOS[samplerate] = butter(
            4, [300.0 / nyq, min(3400.0 / nyq, 0.99)], btype="band", output="sos"
        )
    return sosfilt(_BANDPASS_SOS[samplerate], audio).astype(np.float32)


def _denoise(audio: np.ndarray, samplerate: int) -> np.ndarray:
    """Run DeepFilterNet noise suppression. Resamples to 48 kHz and back."""
    from df.enhance import enhance  # noqa: PLC0415

    model, df_state = _get_df()
    df_sr = df_state.sr()
    device = "cuda" if _CUDA else "cpu"

    # copy() prevents torch from sharing memory with the numpy array
    t = torch.from_numpy(audio.copy()).float().unsqueeze(0)  # [1, T]
    if samplerate != df_sr:
        t = torchaudio.functional.resample(t, samplerate, df_sr)
    t = t.to(device)

    enhanced = enhance(model, df_state, t).detach().cpu()
    if samplerate != df_sr:
        enhanced = torchaudio.functional.resample(enhanced, df_sr, samplerate)
    return enhanced.squeeze(0).cpu().numpy()


def _amplify_and_rms(raw: bytes, db: float = 6.0) -> tuple[bytes, float]:
    """Amplify PCM in one pass, return (amplified_bytes, rms_of_amplified).

    Merges the former _amplify_pcm + _rms_bytes to avoid two frombuffer calls
    per UDP packet on the same data.
    """
    gain = 10 ** (db / 20)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    amplified = np.clip(samples * gain, -32768, 32767)
    rms = float(np.sqrt(np.mean((amplified / 32768.0) ** 2)))
    return amplified.astype(np.int16).tobytes(), rms


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

        # Silence gate is the primary flush path — fires at PTT release for clean
        # transmissions and always yields a semantically complete chunk.
        # max_buf_bytes is the fallback for noisy conditions where RMS never drops
        # below threshold.  20 s captures virtually all complete marine VHF messages
        # while cutting worst-case latency from ~35 s (30 s + GPU) to ~23 s.
        # Cutting to 8 s would slice active transmissions mid-sentence and give
        # Whisper broken context — worse accuracy for no benefit in the common case.
        # 1 s minimum avoids flushing on a noise burst shorter than a real phrase.
        # 500 ms silence gate — PTT squelch drops hard; natural mid-sentence pauses
        # are < 300 ms so 500 ms reliably marks end-of-transmission (was 1 s).
        max_buf_bytes  = bytes_per_sec * 20
        min_buf_bytes  = bytes_per_sec * 1
        silence_drain  = bytes_per_sec // 2

        # 80 ms audio batch before each WebSocket send.  Bundles ~4 UDP packets
        # into one message, cutting send frequency ~4x and giving the browser
        # larger, jitter-stable chunks for gapless Web Audio scheduling.
        audio_batch_threshold = bytes_per_sec * 80 // 1000

        queue: asyncio.Queue[bytes] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(
            lambda: _UDPProtocol(queue),
            local_addr=(udp_ip, udp_port),
        )

        logger.info("RadioTranscribeConsumer: UDP socket bound on %s:%s", udp_ip, udp_port)
        await self.send(text_data=json.dumps({"status": "connected", "source": "udp"}))

        audio_buffer: list[bytes] = []
        total_bytes   = 0
        silence_accum = 0
        audio_batch:  list[bytes] = []
        audio_batch_bytes = 0

        async def _flush_audio() -> bool:
            """Send accumulated audio batch. Returns False on WebSocket error."""
            nonlocal audio_batch, audio_batch_bytes
            if not audio_batch:
                return True
            try:
                await self.send(bytes_data=b"".join(audio_batch))
                audio_batch = []
                audio_batch_bytes = 0
                return True
            except Exception as exc:
                logger.warning("WebSocket audio send failed: %s", exc)
                return False

        try:
            while self._active:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Browser gets amplified audio for real-time monitoring.
                # Transcription buffer stores the original (non-amplified) bytes so
                # DeepFilterNet sees correct signal levels for its noise estimation.
                amplified, rms = _amplify_and_rms(data)

                audio_batch.append(amplified)
                audio_batch_bytes += len(amplified)
                if audio_batch_bytes >= audio_batch_threshold:
                    if not await _flush_audio():
                        break

                audio_buffer.append(data)
                total_bytes += len(data)

                # Silence gate checked on amplified signal for better sensitivity.
                if rms < _RMS_THRESHOLD:
                    silence_accum += len(amplified)
                else:
                    silence_accum = 0

                end_of_tx = silence_accum >= silence_drain and total_bytes >= min_buf_bytes
                if total_bytes >= max_buf_bytes or end_of_tx:
                    # Flush remaining audio so browser hears the full transmission
                    # before the transcription text arrives.
                    if not await _flush_audio():
                        break

                    raw = b"".join(audio_buffer)
                    audio_buffer = []
                    total_bytes   = 0
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
                            logger.warning("WebSocket text send failed: %s", exc)
                            break
        except asyncio.CancelledError:
            raise
        finally:
            transport.close()


# ── File playback ─────────────────────────────────────────────────────────────

class PlaybackConsumer(AsyncWebsocketConsumer):
    """Streams the last uploaded WAV file as raw int16 PCM chunks.

    Mirrors the binary audio format sent by RadioTranscribeConsumer so the
    frontend can reuse scheduleAudio() without any changes.
    """

    async def connect(self):
        await self.accept()
        self._active = True
        self._task = asyncio.ensure_future(self._stream())

    async def disconnect(self, close_code):
        self._active = False
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    async def receive(self, bytes_data=None, text_data=None):
        pass

    async def _stream(self):
        import os
        import wave as _wave

        media_root = settings.MEDIA_ROOT
        last_upload = os.path.join(media_root, "uploads", ".last_upload")
        try:
            path = open(last_upload).read().strip()
            if not os.path.isfile(path):
                path = None
        except FileNotFoundError:
            path = None

        if not path:
            await self.send(text_data=json.dumps({"error": "No uploaded file"}))
            await self.close()
            return

        samplerate = int(getattr(settings, "RADIO_SAMPLE_RATE", 16_000))
        frames_per_chunk = samplerate * 80 // 1000  # 80 ms per chunk

        try:
            with _wave.open(path, "rb") as wf:
                await self.send(text_data=json.dumps({"status": "playing"}))
                while self._active:
                    data = wf.readframes(frames_per_chunk)
                    if not data:
                        break
                    await self.send(bytes_data=data)
                    await asyncio.sleep(0.08)
            if self._active:
                await self.send(text_data=json.dumps({"status": "done"}))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("PlaybackConsumer stream error: %s", exc)
        finally:
            try:
                await self.close()
            except Exception:
                pass


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
