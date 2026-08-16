"""
LoRA 数学动画配音脚本

使用 Edge TTS 生成中文配音，然后用 ffmpeg 合并到视频中。

运行: python add_voiceover.py
输出: lora_math_with_voice.mp4
"""

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path

# 配音文本及对应时间点（秒）
# 每段旁白精简到能在对应动画幕内播完（各幕约 8-12 秒）
NARRATIONS = [
    {
        "start": 0.0,
        "text": "LoRA，低秩适配。为什么只用百分之零点四的参数就够了？",
    },
    {
        "start": 6.0,
        "text": "全量微调要修改整个权重矩阵，一千六百万个参数。",
    },
    {
        "start": 14.0,
        "text": "但微调的变化是低秩的。奇异值集中在前几个，大部分方向几乎没有变化。",
    },
    {
        "start": 24.0,
        "text": "所以我们把变化分解为 B 乘 A，两个小矩阵。参数量降低两百多倍。",
    },
    {
        "start": 34.0,
        "text": "几何上，A 把高维输入压缩到瓶颈，B 再展开回去。只保留最关键的方向。",
    },
    {
        "start": 46.0,
        "text": "多任务切换？换一对矩阵指针就行。base 模型不动，零拷贝。",
    },
    {
        "start": 57.0,
        "text": "总结：低秩分解，极少参数，即时切换。",
    },
]

# Edge TTS 配置
VOICE = "zh-CN-YunxiNeural"  # 男声，清晰自然
RATE = "+5%"  # 稍快一点
VIDEO_PATH = Path(__file__).parent / "media/videos/lora_math_animation/480p15/LoRAMathScene.mp4"
OUTPUT_PATH = Path(__file__).parent / "lora_math_with_voice.mp4"


async def generate_audio_segment(text: str, output_file: str):
    """用 edge-tts 生成单段音频"""
    import edge_tts

    communicate = edge_tts.Communicate(text, VOICE, rate=RATE)
    await communicate.save(output_file)


async def generate_all_audio(tmpdir: str) -> list[dict]:
    """生成所有音频片段，返回带文件路径的列表"""
    segments = []
    tasks = []

    for i, narration in enumerate(NARRATIONS):
        audio_file = os.path.join(tmpdir, f"segment_{i:02d}.mp3")
        segments.append({**narration, "file": audio_file})
        tasks.append(generate_audio_segment(narration["text"], audio_file))

    await asyncio.gather(*tasks)
    return segments


def get_audio_duration(file_path: str) -> float:
    """获取音频文件时长"""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", file_path],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def merge_audio_segments(segments: list[dict], video_duration: float, tmpdir: str) -> str:
    """将所有音频片段按时间点拼接为一个完整音轨（无重叠）。

    策略：在每段音频前插入静音，使其对齐到指定的 start 时间点。
    如果上一段还没播完就到了下一段的 start，则等上一段结束后再开始下一段。
    """
    merged_audio = os.path.join(tmpdir, "merged_narration.mp3")

    # 计算每段实际开始时间（避免重叠）
    actual_starts = []
    current_end = 0.0
    for seg in segments:
        actual_start = max(seg["start"], current_end)
        actual_starts.append(actual_start)
        dur = get_audio_duration(seg["file"])
        current_end = actual_start + dur

    # 生成 ffmpeg concat 列表：静音 + 音频交替
    concat_list_file = os.path.join(tmpdir, "concat_list.txt")
    parts = []

    for i, seg in enumerate(segments):
        # 计算需要在此段前插入多长的静音
        if i == 0:
            silence_dur = actual_starts[0]
        else:
            prev_end = actual_starts[i - 1] + get_audio_duration(segments[i - 1]["file"])
            silence_dur = actual_starts[i] - prev_end

        # 生成静音文件（如果需要）
        if silence_dur > 0.01:
            silence_file = os.path.join(tmpdir, f"silence_{i:02d}.mp3")
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i",
                 f"anullsrc=r=24000:cl=mono", "-t", str(silence_dur),
                 "-c:a", "libmp3lame", "-b:a", "128k", silence_file],
                capture_output=True, check=True,
            )
            parts.append(silence_file)

        parts.append(seg["file"])

    # 末尾补静音到视频总时长
    tail_silence_dur = video_duration - current_end
    if tail_silence_dur > 0.1:
        tail_file = os.path.join(tmpdir, "silence_tail.mp3")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i",
             f"anullsrc=r=24000:cl=mono", "-t", str(tail_silence_dur),
             "-c:a", "libmp3lame", "-b:a", "128k", tail_file],
            capture_output=True, check=True,
        )
        parts.append(tail_file)

    # 写 concat 列表
    with open(concat_list_file, "w") as f:
        for p in parts:
            f.write(f"file '{p}'\n")

    # 用 ffmpeg concat 拼接（无重叠）
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_list_file,
        "-c:a", "libmp3lame", "-b:a", "128k",
        merged_audio,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return merged_audio


def combine_video_audio(video_path: str, audio_path: str, output_path: str):
    """将音频合并到视频中"""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


async def main():
    print("=" * 60)
    print("  LoRA 数学动画配音工具")
    print("=" * 60)

    if not VIDEO_PATH.exists():
        print(f"\n❌ 视频文件不存在: {VIDEO_PATH}")
        print("   请先运行: manim -ql lora_math_animation.py LoRAMathScene")
        return

    video_duration = get_audio_duration(str(VIDEO_PATH))
    print(f"\n📹 视频时长: {video_duration:.1f}s")
    print(f"🗣️  配音语音: {VOICE}")
    print(f"📝 旁白段数: {len(NARRATIONS)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: 生成所有 TTS 音频
        print("\n[1/3] 生成 TTS 音频...")
        segments = await generate_all_audio(tmpdir)

        for seg in segments:
            dur = get_audio_duration(seg["file"])
            print(f"  t={seg['start']:5.1f}s | {dur:.1f}s | {seg['text'][:30]}...")

        # Step 2: 合并音频到统一音轨
        print("\n[2/3] 合并音频轨道...")
        merged_audio = merge_audio_segments(segments, video_duration, tmpdir)
        print(f"  合并音轨时长: {get_audio_duration(merged_audio):.1f}s")

        # Step 3: 视频 + 音频合成
        print("\n[3/3] 合成最终视频...")
        combine_video_audio(str(VIDEO_PATH), merged_audio, str(OUTPUT_PATH))

    print(f"\n✅ 完成！输出文件: {OUTPUT_PATH}")
    print(f"   文件大小: {OUTPUT_PATH.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    asyncio.run(main())
