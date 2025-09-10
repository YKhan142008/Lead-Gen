import os, sys, traceback, time, queue, tempfile, subprocess
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from pydub.playback import play
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

# --- New: attempt to import VibeVoice demo code as a module
VIBEVOICE_OK = False
try:
    # The community fork installs a package 'vibevoice' with demo utils
    import torch
    from vibevoice.demo.inference_from_file import main as vv_infer_main
    VIBEVOICE_OK = True
except Exception:
    VIBEVOICE_OK = False

SAMPLERATE = 16000
BLOCKSIZE = 1024
DTYPE = "int16"
SILENCE_THRESHOLD = 100
SILENCE_DURATION = 1.2
WHISPER_SIZE = "base"
OLLAMA_CMD = "ollama"

audio_q = queue.Queue()

def debug(msg): print(f"[debug] {msg}", flush=True)

def check_ollama():
    try:
        subprocess.run([OLLAMA_CMD, "--version"], capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise RuntimeError("'ollama' not found on PATH.")
    except subprocess.CalledProcessError as e:
        debug(f"ollama --version stderr:\n{e.stderr}")

def check_audio():
    devices = sd.query_devices()
    if not any(d.get("max_input_channels", 0) > 0 for d in devices):
        raise RuntimeError("No audio input devices found.")

def audio_callback(indata, frames, time_info, status):
    if status:
        debug(f"PortAudio status: {status}")
    audio_q.put(indata.copy())

def record_loop():
    buffer = []
    speaking = False
    silence_start = None
    with sd.InputStream(callback=audio_callback, channels=1,
                        samplerate=SAMPLERATE, blocksize=BLOCKSIZE, dtype=DTYPE):
        print("🎙️ Listening... (say 'quit' to exit)")
        while True:
            try:
                chunk = audio_q.get(timeout=1.0)
            except queue.Empty:
                continue
            volume = float(np.abs(chunk).mean())
            if volume > SILENCE_THRESHOLD:
                buffer.append(chunk)
                speaking = True
                silence_start = None
            else:
                if speaking:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_DURATION:
                        return np.concatenate(buffer, axis=0).reshape(-1)

def transcribe(audio):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        write(tmpfile.name, SAMPLERATE, audio)
        model = WhisperModel(WHISPER_SIZE)
        segments, _ = model.transcribe(tmpfile.name)
        text = " ".join(seg.text for seg in segments)
    return text.strip()

def ask_llama3(prompt):
    res = subprocess.run([OLLAMA_CMD, "run", "llama3", prompt],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ollama error (code {res.returncode}):\n{res.stderr.strip()}")
    return res.stdout.strip()

# --- New: VibeVoice (community fork) synth on CPU
def speak_with_vibevoice(text, speaker="Alice"):
    if not VIBEVOICE_OK:
        raise RuntimeError("VibeVoice module not available.")

    # Prepare a minimal multi-speaker text file format expected by demo script
    # One line with: <Speaker>: <Text>
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as txtf:
        txtf.write(f"{speaker}: {text}\n")
        txt_path = txtf.name

    # Build arg list for demo.inference_from_file.main
    # NOTE: The demo defaults to CUDA if present; we force CPU by torch.set_default_device when CUDA is unavailable.
    # The model repo names are from the Hugging Face card.
    # 1.5B is lighter and better for CPU (still slow).
    model_path = "microsoft/VibeVoice-1.5B"  # or "microsoft/VibeVoice-Large"
    sys_argv_backup = sys.argv[:]  # safeguard
    try:
        # Force CPU if no CUDA
        if not torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            torch.set_num_threads(max(1, os.cpu_count() // 2))

        # Simulate CLI: demo/inference_from_file.py --model_path ... --txt_path ... --speaker_names Alice
        sys.argv = [
            "inference_from_file.py",
            "--model_path", model_path,
            "--txt_path", txt_path,
            "--speaker_names", speaker,
            # You can experiment adding flags if the fork exposes them, e.g.:
            # "--output_dir", some_tmp_dir,
            # "--device", "cpu"
        ]
        # The demo script typically writes a WAV to an outputs directory and prints the path.
        vv_infer_main()

        # Find most recent WAV produced (simplest portable approach)
        # Look for *.wav in cwd and demo/outputs
        candidate_dirs = [os.getcwd(), os.path.join(os.getcwd(), "outputs"), os.path.join(os.getcwd(), "demo", "outputs")]
        wav_candidates = []
        for d in candidate_dirs:
            if os.path.isdir(d):
                for name in os.listdir(d):
                    if name.lower().endswith(".wav"):
                        wav_candidates.append(os.path.join(d, name))
        if not wav_candidates:
            raise RuntimeError("VibeVoice generated no WAV file (couldn’t locate output).")

        wav_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        wav_path = wav_candidates[0]
        sound = AudioSegment.from_wav(wav_path)
        play(sound)
    finally:
        sys.argv = sys_argv_backup

# --- Fallback TTS (fast CPU, just to keep the loop usable)
def speak_with_fallback_tts(text):
    import pyttsx3
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    check_audio()
    check_ollama()
    print("✅ Audio OK. Using CPU for TTS unless CUDA is available.")
    if VIBEVOICE_OK:
        print("✅ VibeVoice module found (community fork).")
    else:
        print("⚠️ VibeVoice not importable; will use fallback TTS.")

    while True:
        audio = record_loop()
        if audio is None or len(audio) == 0:
            continue

        user_text = transcribe(audio)
        print(f"👤 You said: {user_text}")

        if user_text.lower() in {"quit", "exit"}:
            print("👋 Bye!")
            break

        reply = ask_llama3(user_text)
        print(f"🤖 Bot: {reply}")

        try:
            if VIBEVOICE_OK:
                speak_with_vibevoice(reply)
            else:
                speak_with_fallback_tts(reply)
        except Exception as e:
            print(f"⚠️ TTS failed ({e}). Falling back to basic TTS.")
            speak_with_fallback_tts(reply)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
    except Exception:
        print("\n❌ Unhandled error:\n")
        traceback.print_exc()
        try:
            input("\nPress Enter to close…")
        except Exception:
            pass

