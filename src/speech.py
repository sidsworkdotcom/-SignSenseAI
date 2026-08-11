"""Non-blocking text-to-speech for SignSense AI.

pyttsx3 uses the OS's built-in voices (SAPI5 on Windows, NSSpeech on macOS,
eSpeak on Linux) — fully offline, no API keys. Speaking is blocking, so we run
it in a daemon worker thread fed by a queue; the video loop never stalls.

If pyttsx3 isn't installed or fails to initialize, Speaker degrades to a
silent no-op so the app still runs.
"""
from __future__ import annotations

import queue
import threading


class Speaker:
    def __init__(self, rate: int = 165, enabled: bool = True):
        self.enabled = enabled
        self.available = False
        self._q: queue.Queue[str | None] = queue.Queue()
        try:
            import pyttsx3  # noqa: F401

            self._rate = rate
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
            self.available = True
        except Exception as e:  # missing package / no audio device
            print(f"[speech] voice disabled: {e}")

    def _worker(self) -> None:
        import pyttsx3

        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)
        while True:
            text = self._q.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[speech] error: {e}")

    def say(self, text: str) -> None:
        """Queue text for speech. Returns immediately."""
        text = text.strip()
        if self.enabled and self.available and text:
            # drop stale backlog so speech never lags far behind the signing
            while self._q.qsize() > 2:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
            self._q.put(text)

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled

    def close(self) -> None:
        if self.available:
            self._q.put(None)
