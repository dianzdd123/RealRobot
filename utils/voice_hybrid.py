import os, json, re, threading, queue, time, sys
import numpy as np
import sounddevice as sd
import soundfile as sf
from pynput import keyboard
from openai import OpenAI
from dotenv import load_dotenv

# Initialize OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Audio hardware configuration
try:
    SR = int(sd.query_devices(kind='input')['default_samplerate'])
except Exception:
    SR = 48000  # Fallback sample rate

CHANNELS = 1
DTYPE = "float32"

class PushToTalkRecorder:
    """Manages audio stream recording with a push-to-talk mechanism."""
    def __init__(self, samplerate=SR, channels=CHANNELS, device=None, dtype=DTYPE):
        self.samplerate = samplerate
        self.channels = channels
        self.device = device
        self.dtype = dtype
        self._audio_q = queue.Queue()
        self._stream = None
        self._recording = False
        self._frames = 0

    def _callback(self, indata, frames, time_info, status):
        if self._recording:
            self._audio_q.put(indata.copy())
            self._frames += frames

    def start(self):
        """Starts the audio input stream and enables recording."""
        if self._stream is None:
            self._stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                dtype=self.dtype,
                callback=self._callback,
                device=self.device,
            )
            self._stream.start()
        self._recording = True
        self._frames = 0
        while not self._audio_q.empty():
            try: self._audio_q.get_nowait()
            except queue.Empty: break

    def stop(self, out_path="input.wav"):
        """Disables recording and saves the captured buffer to a WAV file."""
        self._recording = False
        samples = []
        while not self._audio_q.empty():
            samples.append(self._audio_q.get())
        if not samples:
            print("[Audio] No samples captured.")
            return None
        arr = np.concatenate(samples, axis=0)
        sf.write(out_path, arr, self.samplerate)
        print(f"[Audio] Saved: {out_path} ({len(arr)/self.samplerate:.2f}s)")
        return out_path

    def close(self):
        """Properly closes the audio stream hardware."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

def transcribe_speech(wav_path="input.wav", prefer_model="gpt-4o-mini-transcribe", fallback_model="whisper-1"):
    """Converts speech audio to text using OpenAI STT models."""
    def _do(model_name):
        with open(wav_path, "rb") as f:
            return client.audio.transcriptions.create(model=model_name, file=f)
    try:
        print(f"[STT] Transcribing with: {prefer_model}")
        resp = _do(prefer_model)
    except Exception:
        print(f"[STT] Primary model failed. Falling back to: {fallback_model}")
        resp = _do(fallback_model)
    
    text = (getattr(resp, "text", None) or getattr(resp, "transcript", "")).strip()
    print(f"[STT] Result: {text}")
    return text

def wait_for_command(timeout_sec=10) -> str | None:
    """Waits for user input through either PTT voice (SPACE) or keyboard text (Any other key)."""
    rec = PushToTalkRecorder()
    out_path = "input.wav"
    decision_made = threading.Event()
    timed_out = threading.Event()
    mode = {"value": None}
    listener_ref = {"obj": None}
    text_result = {"value": None}

    def timeout_func():
        if not decision_made.is_set():
            timed_out.set()
            print("\n[Error] Input timeout. No command detected.")
            if listener_ref["obj"]: listener_ref["obj"].stop()

    def on_press(key):
        if not decision_made.is_set():
            if key == keyboard.Key.esc:
                print("[Exit] User cancelled.")
                if listener_ref["obj"]: listener_ref["obj"].stop()
                return False
            if key == keyboard.Key.space:
                mode["value"] = "voice"
                decision_made.set()
                print("[Mode] Voice Input: Hold SPACE to record, release to stop.")
                if not rec._recording: rec.start()
                return
            else:
                mode["value"] = "text"
                decision_made.set()
                print("[Mode] Keyboard Input: Please type your command in the console.")
                if listener_ref["obj"]: listener_ref["obj"].stop()
                return False
        
        if mode["value"] == "voice" and key == keyboard.Key.space:
            if not rec._recording: rec.start()

    def on_release(key):
        if mode["value"] != "voice": return
        if key == keyboard.Key.space:
            if rec._recording:
                print("[PTT] Processing audio...")
                wav = rec.stop(out_path=out_path)
                if wav:
                    try:
                        text = transcribe_speech(wav_path=wav)
                        if text:
                            text_result["value"] = text
                            rec.close()
                            return False # Stop listener and return to main
                    except Exception as e:
                        print(f"[STT] Error: {e}")
                print("[Hint] Hold SPACE to record again, or press ESC to exit.")
        elif key == keyboard.Key.esc:
            rec.close()
            return False

    # Start keyboard listener with a timeout timer
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener_ref["obj"] = listener
        timer = threading.Timer(timeout_sec, timeout_func)
        timer.start()
        listener.join()
        timer.cancel()

    if timed_out.is_set(): return None

    if mode["value"] == "text":
        try:
            user_line = input(">> ").strip()
            return user_line if user_line else None
        except (EOFError, KeyboardInterrupt):
            print("\n[Exit] Text input cancelled.")
            return None

    return text_result["value"]

def ptt_loop(on_text, on_exit=None):
    """Continuous PTT loop that triggers a callback on each successful transcription."""
    rec = PushToTalkRecorder()
    out_path = "input.wav"

    def on_press(key):
        if key == keyboard.Key.esc:
            return False
        if key == keyboard.Key.space and not rec._recording:
            print("[PTT] Recording...")
            rec.start()

    def on_release(key):
        if key == keyboard.Key.space and rec._recording:
            print("[PTT] Transcribing...")
            wav = rec.stop(out_path=out_path)
            if wav:
                try:
                    text = transcribe_speech(wav_path=wav)
                    if text: on_text(text)
                except Exception as e:
                    print(f"[STT] Error: {e}")
            print("[Hint] Hold SPACE to record, ESC to exit.")

    try:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            print("[Mode] Voice Loop active. ESC to quit.")
            listener.join()
    finally:
        if on_exit: on_exit()
        rec.close()