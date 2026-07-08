import asyncio, subprocess
from pathlib import Path
import edge_tts, wave

async def test():
    text = "你好，这是一段测试语音，边缘语音合成正常工作。"
    voice = "zh-CN-YunyangNeural"
    tmp_mp3 = Path("/tmp/test_edge.mp3")
    out_wav = Path("/tmp/test_edge.wav")

    communicate = edge_tts.Communicate(text, voice, rate="+0%")
    await communicate.save(str(tmp_mp3))
    print(f"MP3 size: {tmp_mp3.stat().st_size} bytes")

    subprocess.run([
        "ffmpeg", "-y", "-i", str(tmp_mp3),
        "-ar", "44100", "-ac", "1", "-sample_fmt", "s16",
        str(out_wav)
    ], capture_output=True, check=True)

    with wave.open(str(out_wav), "rb") as w:
        dur = w.getnframes() / w.getframerate()
        print(f"WAV OK: {dur:.2f}s, {w.getframerate()}Hz, {w.getnchannels()}ch")

    print("✅ edge-tts is working correctly")

asyncio.run(test())
