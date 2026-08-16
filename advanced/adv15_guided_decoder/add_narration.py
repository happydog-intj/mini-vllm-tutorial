"""
adv15_guided_decoder/add_narration.py

为 JSON Guided Decoding 动画添加 AI 配音（Edge TTS + ffmpeg 合成）

使用方法:
    python add_narration.py

依赖:
    pip install edge-tts moviepy

输出:
    media/JSONGuidedScene_narrated.mp4
"""

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path

import edge_tts

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
VOICE = "zh-CN-XiaoxiaoNeural"  # 微软晓晓，中文女声，清晰教学风格
RATE = "+0%"                     # 语速调节: -20% 慢 / +0% 正常 / +20% 快
VOLUME = "+0%"

VIDEO_PATH = Path("media/videos/json_guided_animation/480p15/JSONGuidedScene.mp4")
OUTPUT_PATH = Path("media/JSONGuidedScene_narrated.mp4")

# ---------------------------------------------------------------------------
# 旁白脚本：每段对应动画的一个幕/场景
# (start_sec, text) — start_sec 是该段旁白的起始时间点
# ---------------------------------------------------------------------------
NARRATION_SEGMENTS = [
    # 标题页
    (0.5, "模型如何输出合法的Jason？今天我们来拆解引导解码的核心机制。"),

    # 第一幕：自由生成 vs 约束生成
    (8.0, "当模型自由生成时，经常输出格式错误的Jason。"),
    (13.0, "但如果我们在每一步只允许语法合法的托肯，"
           "输出就一定是合法的。这就是引导解码。"),

    # 第二幕：JSON FSM
    (21.0, "Jason的语法可以表示为一个有限状态机。"),
    (25.0, "每个状态定义了下一步允许什么字符。"),
    (29.5, "注意这个黄色高亮，它代表当前状态。"),
    (33.5, "每个状态只允许特定字符，不合法的直接屏蔽。"),

    # 第三幕：逐 token 掩码
    (38.0, "现在让我们看具体的生成过程。"),
    (41.5, "第一步，开头只有左花括号是合法的。"),
    (45.0, "第七步，key之后只能是冒号，模型别无选择。"),
    (49.5, "第八步有多个合法选择，模型在它们之间自由竞争。"),
    (54.5, "最终输出完整的Jason，百分之百合法。"),

    # 第四幕：logits masking
    (58.0, "核心机制是logits masking。"),
    (61.0, "模型对每个托肯都输出一个分数，叫logit。"),
    (65.5, "当前状态只允许左花括号，其他设为负无穷。"),
    (70.0, "经过soft max之后，负无穷变成零概率。模型被迫选择合法选项。"),

    # 总结
    (76.0, "这就是引导解码，保证百分之百合法输出。"),
]


# ---------------------------------------------------------------------------
# 核心逻辑
# ---------------------------------------------------------------------------

async def generate_audio_segment(text: str, output_path: str) -> float:
    """用 Edge TTS 生成单段音频，返回时长(秒)"""
    communicate = edge_tts.Communicate(text, VOICE, rate=RATE, volume=VOLUME)
    await communicate.save(output_path)

    # 获取时长
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", output_path],
        capture_output=True, text=True
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


async def generate_all_narration(tmp_dir: str) -> list[tuple[float, str, float]]:
    """生成所有旁白音频片段，返回 [(start, path, duration), ...]"""
    segments = []
    tasks = []

    for i, (start_sec, text) in enumerate(NARRATION_SEGMENTS):
        out_path = os.path.join(tmp_dir, f"seg_{i:03d}.mp3")
        tasks.append((i, start_sec, out_path, text))

    results = []
    for i, start_sec, out_path, text in tasks:
        print(f"  生成第 {i+1}/{len(tasks)} 段: {text[:20]}...")
        duration = await generate_audio_segment(text, out_path)
        results.append((start_sec, out_path, duration))

    # 检测并警告重叠
    for i in range(len(results) - 1):
        start_i, _, dur_i = results[i]
        start_next, _, _ = results[i + 1]
        end_i = start_i + dur_i
        if end_i > start_next:
            overlap = end_i - start_next
            print(f"  ⚠️  第{i+1}段结束于{end_i:.1f}s，第{i+2}段开始于{start_next:.1f}s，重叠{overlap:.1f}s")

    return results


def merge_audio_segments(segments: list[tuple[float, str, float]],
                         video_duration: float,
                         output_path: str):
    """
    将多段音频按时间点拼接为一条完整音轨（无重叠）。

    策略：如果某段音频还没播完下一段就开始了，用 atrim 截断前一段，
    避免重音问题。最终用 concat 而不是 amix 来拼接。
    """
    if not segments:
        return

    # 构建 ffmpeg 复杂滤镜：每段 adelay 到正确位置，然后用 amix
    # 但设置 normalize=0 避免音量被平均化
    inputs = []
    filter_parts = []

    for i, (start_sec, audio_path, duration) in enumerate(segments):
        inputs.extend(["-i", audio_path])
        delay_ms = int(start_sec * 1000)

        # 如果有下一段，截断当前段避免重叠
        if i < len(segments) - 1:
            next_start = segments[i + 1][0]
            max_dur = next_start - start_sec
            if duration > max_dur and max_dur > 0:
                # 截断这段音频
                filter_parts.append(
                    f"[{i}:a]atrim=0:{max_dur},asetpts=PTS-STARTPTS,"
                    f"adelay={delay_ms}|{delay_ms},aformat=sample_rates=44100[a{i}]"
                )
                continue

        filter_parts.append(
            f"[{i}:a]adelay={delay_ms}|{delay_ms},aformat=sample_rates=44100[a{i}]"
        )

    # 混合所有音轨（concat 模式：不重叠时等价于叠加静音段）
    mix_inputs = "".join(f"[a{i}]" for i in range(len(segments)))
    filter_parts.append(
        f"{mix_inputs}amix=inputs={len(segments)}:duration=longest"
        f":dropout_transition=0:normalize=0[aout]"
    )

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[aout]",
        "-t", str(video_duration),
        "-ac", "1",
        "-ar", "44100",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def combine_video_audio(video_path: str, audio_path: str, output_path: str):
    """将视频和混合音频合并为最终带配音视频"""
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
    subprocess.run(cmd, check=True, capture_output=True)


def get_video_duration(video_path: str) -> float:
    """获取视频时长"""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", video_path],
        capture_output=True, text=True
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


async def main():
    print("=" * 55)
    print("adv15 — 为 JSON Guided Decoding 动画添加 AI 配音")
    print("=" * 55)

    # 检查视频存在
    if not VIDEO_PATH.exists():
        print(f"❌ 视频文件不存在: {VIDEO_PATH}")
        print("   请先运行: manim -ql json_guided_animation.py JSONGuidedScene")
        return

    video_duration = get_video_duration(str(VIDEO_PATH))
    print(f"视频时长: {video_duration:.1f}s")
    print(f"旁白段数: {len(NARRATION_SEGMENTS)}")
    print(f"TTS 声音: {VOICE}")
    print()

    # 生成音频片段
    with tempfile.TemporaryDirectory() as tmp_dir:
        print("[1/3] 生成旁白音频...")
        segments = await generate_all_narration(tmp_dir)
        print(f"  ✓ 共生成 {len(segments)} 段音频\n")

        # 混合为一条音轨
        print("[2/3] 混合音频轨道...")
        mixed_audio = os.path.join(tmp_dir, "mixed_narration.wav")
        merge_audio_segments(segments, video_duration, mixed_audio)
        print("  ✓ 音轨混合完成\n")

        # 合并视频 + 音频
        print("[3/3] 合并视频与配音...")
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        combine_video_audio(str(VIDEO_PATH), mixed_audio, str(OUTPUT_PATH))
        print(f"  ✓ 输出: {OUTPUT_PATH}\n")

    print("=" * 55)
    print(f"✅ 配音视频已生成: {OUTPUT_PATH}")
    print(f"   播放: open {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
