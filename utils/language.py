import os, json, re, threading, queue, time, sys
from dotenv import load_dotenv
import sounddevice as sd
import soundfile as sf
from pynput import keyboard
from openai import OpenAI

# ============ 基础 ============
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SR = 16000
CHANNELS = 1
DTYPE = "float32"

# ============ Push-to-Talk 录音器 ============
class PushToTalkRecorder:
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
        if status:
            pass
        if self._recording:
            self._audio_q.put(indata.copy())
            self._frames += frames
        return None

    def start(self):
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
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                break

    def stop(self, out_path="input.wav"):
        self._recording = False
        samples = []
        while not self._audio_q.empty():
            samples.append(self._audio_q.get())
        if not samples:
            print("[Audio] No samples captured.")
            return None
        data = b"".join([s.tobytes() for s in samples])
        import numpy as np
        arr = np.frombuffer(data, dtype=self.dtype).reshape(-1, self.channels)
        sf.write(out_path, arr, self.samplerate)
        print(f"[Audio] Saved: {out_path} ({len(arr)/self.samplerate:.2f}s)")
        return out_path

    def close(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

# ============ 语音转写 ============
def transcribe_speech(
    wav_path="input.wav",
    prefer_model="gpt-4o-mini-transcribe",
    fallback_model="whisper-1",
    lang_hint=None,
):
    def _do(model_name):
        with open(wav_path, "rb") as f:
            return client.audio.transcriptions.create(
                model=model_name,
                file=f,
                # language=lang_hint,
            )
    try:
        print(f"[STT] Using: {prefer_model}")
        resp = _do(prefer_model)
    except Exception as e:
        print(f"[STT] Prefer failed: {e}\n[STT] Fallback: {fallback_model}")
        resp = _do(fallback_model)
    text = getattr(resp, "text", None) or getattr(resp, "transcript", "")
    text = (text or "").strip()
    print(f"[STT] => {text}")
    return text

# ============ 你的原有函数 ============
def roborefer_prompts(user_text: str) -> list:
    system_instructions = (
        "You need to understand the user's instruction and extract the parts of the command "
        "that describe specific objects or areas of interest.\n"
        "Return ONLY a JSON array of strings in the same order."
    )
    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": user_text}
    ]
    resp = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0)
    parsed_raw = resp.choices[0].message.content or ""

    def _extract_json_block(text: str) -> str:
        text = text.strip()
        if text.startswith('[') and text.endswith(']'):
            return text
        match = re.search(r'\[[\s\S]*\]', text)
        if match:
            return match.group(0)
        raise ValueError(f"Could not find JSON array in: {text[:200]}")
    slots = json.loads(_extract_json_block(parsed_raw))
    prompts = [f'Find the object or area "{s.strip()}" in the image.' for s in slots]
    print("\n[Extracted Targets]"); [print("-", s) for s in slots]
    print("\n[Generated Prompts]"); [print("-", p) for p in prompts]
    return prompts

# ============ 主程序：混合输入（首键决定） ============
def main():
    print("----- Hybrid Input (10s timeout) -----")
    print("在 10 秒内按下首个按键来决定输入方式：")
    print(" - 空格：按住说话、松手停止（语音输入）")
    print(" - 其他任意键：切换为文字输入（在控制台输入一行并回车）\n")

    rec = PushToTalkRecorder()
    out_path = "input.wav"

    decision_made = threading.Event()
    timed_out = threading.Event()
    mode = {"value": None}  # 'voice' or 'text'

    listener_ref = {"obj": None}  # 供超时回调里 stop()

    def timeout_func():
        if not decision_made.is_set():
            timed_out.set()
            print("[Error] 未检测到输入（10 秒超时）。程序退出。")
            # 停止监听器使主线程继续
            if listener_ref["obj"] is not None:
                listener_ref["obj"].stop()

    # 键盘回调
    def on_press(key):
        # 若尚未决定模式，由“首个按键”决定
        if not decision_made.is_set():
            if key == keyboard.Key.esc:
                print("[Exit] 用户取消。")
                if listener_ref["obj"] is not None:
                    listener_ref["obj"].stop()
                return False

            if key == keyboard.Key.space:
                mode["value"] = "voice"
                decision_made.set()
                print("[Mode] 语音输入：按住空格说话，松手停止。")
                # 首按空格也要开始录音（PTT 体验更自然）
                if not rec._recording:
                    print("[PTT] Recording… (hold SPACE)")
                    rec.start()
                return  # 继续监听后续的空格/ESC
            else:
                mode["value"] = "text"
                decision_made.set()
                print("[Mode] 文字输入：请在控制台输入一行并回车。")
                # 切到文字模式就停止监听，转入 input()
                if listener_ref["obj"] is not None:
                    listener_ref["obj"].stop()
                return False  # 终止监听器

        # 若已是语音模式，正常处理 PTT
        if mode["value"] == "voice" and key == keyboard.Key.space:
            if not rec._recording:
                print("[PTT] Recording… (hold SPACE)")
                rec.start()

    def on_release(key):
        if mode["value"] != "voice":
            return  # 仅语音模式处理释放事件
        if key == keyboard.Key.space:
            if rec._recording:
                print("[PTT] Stopping…")
                wav = rec.stop(out_path=out_path)
                if wav:
                    text = transcribe_speech(wav_path=wav, lang_hint=None)
                    if text:
                        roborefer_prompts(text)
                print("\n[Hint] 继续按住空格可再次录音，或按 ESC 退出。")
        elif key == keyboard.Key.esc:
            print("[PTT] Exit.")
            rec.close()
            return False

    # 启动监听 + 超时计时器
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener_ref["obj"] = listener
        timer = threading.Timer(10.0, timeout_func)
        timer.start()
        listener.join()
        timer.cancel()

    # 若超时，直接退出
    if timed_out.is_set():
        sys.exit(1)

    # 文字模式：阻塞读取一行文本 -> 调你的流水线
    if mode["value"] == "text":
        try:
            user_line = input(">> ")
        except (EOFError, KeyboardInterrupt):
            print("\n[Exit] 取消输入。")
            sys.exit(1)
        user_line = (user_line or "").strip()
        if not user_line:
            print("[Error] 未检测到输入文本。程序退出。")
            sys.exit(1)
        roborefer_prompts(user_line)
        print("[Done] 文字输入处理完成。")
        return

    # 语音模式：到这里说明在监听循环中被 ESC 退出或进程结束
    rec.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Exit] 用户中断。")