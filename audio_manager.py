"""
Audio Manager
==============
Handles PyAudio microphone capture and speaker playback.
Routes audio frames simultaneously to:
  - VoicePipeline (for STT via Deepgram WebSocket)
  - BehavioralAnalyzer (for local MFCC analysis)
  - Playback stream (for TTS audio chunks from ElevenLabs/Deepgram)
"""

from __future__ import annotations

import io
import logging
import threading
from typing import Optional, Callable

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_SIZE = 512
CHANNELS = 1
FORMAT_INT16 = 8   # pyaudio.paInt16


class AudioManager:
    """
    Manages all audio I/O for the interviewer session.
    Thread-safe: input callbacks run on PyAudio's callback thread.
    """

    def __init__(self):
        self._pa = None
        self._input_stream = None
        self._output_stream = None
        self._is_running = False
        self._playback_lock = threading.Lock()

        # Callbacks registered by other modules
        self._on_frame_callbacks: list[Callable[[bytes], None]] = []

        # Playback buffer (FIFO of MP3/PCM chunks)
        self._playback_queue: list[bytes] = []
        self._mp3_buffer = b""
        self._pydub = None
        self._ffmpeg_available = self._check_ffmpeg()

    # ── Initialization ────────────────────────────────────────────────────────

    def initialize(self):
        """Initialize PyAudio and open streams."""
        import pyaudio
        self._pa = pyaudio.PyAudio()
        self._open_input_stream()
        self._open_output_stream()
        self._is_running = True
        logger.info("AudioManager initialized")

    def _open_input_stream(self):
        import pyaudio
        self._input_stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self._input_callback,
        )
        self._input_stream.start_stream()

    def _open_output_stream(self):
        import pyaudio
        self._output_stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=22050,            # ElevenLabs default output rate
            output=True,
            frames_per_buffer=1024,
        )

    def _input_callback(self, in_data, frame_count, time_info, status):
        """PyAudio input callback — runs on audio thread."""
        import pyaudio
        if self._is_running and in_data:
            for cb in self._on_frame_callbacks:
                try:
                    cb(in_data)
                except Exception as e:
                    logger.error(f"Audio frame callback error: {e}")
        return (None, pyaudio.paContinue)

    # ── Registration API ──────────────────────────────────────────────────────

    def register_frame_callback(self, callback: Callable[[bytes], None]):
        """Register a callback to receive raw PCM frames (16kHz, 16-bit, mono)."""
        self._on_frame_callbacks.append(callback)

    # ── Playback API ──────────────────────────────────────────────────────────

    def play_audio_chunk(self, mp3_bytes: bytes):
        """
        Accepts MP3 chunks from TTS stream and plays them.
        Decodes MP3 → PCM using pydub/ffmpeg if available,
        otherwise buffers and attempts to play raw.
        """
        self._mp3_buffer += mp3_bytes
        # Attempt decode when we have enough data
        if len(self._mp3_buffer) >= 8192:
            self._flush_mp3_buffer()

    def flush_playback(self):
        """Flush remaining MP3 buffer at end of TTS stream."""
        if self._mp3_buffer:
            self._flush_mp3_buffer()

    def _flush_mp3_buffer(self):
        try:
            pcm_data = self._decode_mp3(self._mp3_buffer)
            if pcm_data and self._output_stream:
                with self._playback_lock:
                    self._output_stream.write(pcm_data)
        except Exception as e:
            logger.warning(f"MP3 decode/playback error: {e}")
        finally:
            self._mp3_buffer = b""

    def _decode_mp3(self, mp3_bytes: bytes) -> Optional[bytes]:
        """Decode MP3 to raw PCM. Uses pydub if available."""
        try:
            from pydub import AudioSegment
            segment = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
            segment = segment.set_frame_rate(22050).set_channels(1).set_sample_width(2)
            return segment.raw_data
        except ImportError:
            # pydub not available — try soundfile
            try:
                import soundfile as sf
                data, sr = sf.read(io.BytesIO(mp3_bytes), dtype="int16")
                return data.tobytes()
            except Exception:
                return None
        except Exception as e:
            logger.debug(f"MP3 decode failed: {e}")
            return None

    def _check_ffmpeg(self) -> bool:
        import subprocess
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def get_input_device_list(self) -> list[dict]:
        """Return list of available input devices."""
        if not self._pa:
            return []
        devices = []
        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                devices.append({"index": i, "name": info["name"]})
        return devices

    def shutdown(self):
        self._is_running = False
        if self._input_stream:
            self._input_stream.stop_stream()
            self._input_stream.close()
        if self._output_stream:
            self._output_stream.stop_stream()
            self._output_stream.close()
        if self._pa:
            self._pa.terminate()
        logger.info("AudioManager shut down")
