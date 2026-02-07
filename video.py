import os
from moviepy import ImageClip, TextClip, CompositeVideoClip, concatenate_videoclips, AudioFileClip
import numpy as np

def apply_ken_burns_effect(clip, zoom_ratio=0.08):
    """
    Apply Ken Burns zoom effect using frame-by-frame transformation
    Gradually zooms in on the image for visual interest
    """
    w, h = clip.size

    def make_frame(get_frame, t):
        # Get original frame using the get_frame function
        frame = get_frame(t)

        # Calculate zoom progression (0 to zoom_ratio over duration)
        progress = t / clip.duration
        zoom = 1.0 + (zoom_ratio * progress)

        # Calculate new dimensions
        new_w = int(w * zoom)
        new_h = int(h * zoom)

        # Resize using PIL
        from PIL import Image
        img = Image.fromarray(frame)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Crop to center to maintain original dimensions
        x_offset = (new_w - w) // 2
        y_offset = (new_h - h) // 2
        cropped = np.array(img)[y_offset:y_offset+h, x_offset:x_offset+w]

        return cropped

    return clip.transform(make_frame)

def create_crossfade(clip1, clip2, duration=0.5):
    """
    Create a crossfade transition between two clips
    """
    # Overlap the clips by crossfade duration
    clip1_end = clip1.duration - duration

    def make_frame(t):
        if t < clip1_end:
            return clip1.get_frame(t)
        elif t < clip1.duration:
            # Crossfade period
            alpha = (t - clip1_end) / duration
            frame1 = clip1.get_frame(t)
            frame2 = clip2.get_frame(t - clip1_end)
            return (frame1 * (1 - alpha) + frame2 * alpha).astype(np.uint8)
        else:
            return clip2.get_frame(t - clip1_end)

    total_duration = clip1.duration + clip2.duration - duration
    return clip1.transform(make_frame).with_duration(total_duration)

def create_text_with_fade(text, duration=3.5, fade_duration=0.4):
    """Create text clip with manual fade in/out effect"""
    txt = TextClip(
        text=text,
        font_size=52,
        color="white",
        stroke_color="black",
        stroke_width=2
    )

    def make_frame(get_frame, t):
        frame = get_frame(0)  # Get the text frame (static, so t=0)
        # Fade in
        if t < fade_duration:
            alpha = t / fade_duration
            return (frame * alpha).astype(np.uint8)
        # Fade out
        elif t > duration - fade_duration:
            alpha = (duration - t) / fade_duration
            return (frame * alpha).astype(np.uint8)
        # Full opacity
        else:
            return frame

    return txt.transform(make_frame).with_duration(duration)

def assemble_video(scene_paths, scene_texts, audio_path, out_path, storyboard=None):
    """
    Assemble video with:
    - Ken Burns zoom effect on each image for visual interest
    - Text with fade in/out for smooth appearance
    - Black stroke on text for better readability
    - Scene durations synced to TTS narration timing
    """
    clips = []

    # Get actual scene durations from storyboard (set by audio generation)
    scene_durations = []
    if storyboard and "scenes" in storyboard:
        scene_durations = [scene.get("duration_s", 3.5) for scene in storyboard["scenes"]]
    else:
        # Fallback to text-length-based durations
        for t in scene_texts:
            base_duration = 3.5
            text_length_factor = min(len(t) / 100, 1.5)
            scene_durations.append(base_duration + text_length_factor)

    for i, (p, t) in enumerate(zip(scene_paths, scene_texts)):
        # Use actual duration from narration timing
        scene_duration = scene_durations[i] if i < len(scene_durations) else 3.5

        # Create image clip
        img = ImageClip(p).with_duration(scene_duration)

        # Apply Ken Burns zoom effect for visual interest
        try:
            img = apply_ken_burns_effect(img, zoom_ratio=0.08)
            print(f"  ✓ Applied Ken Burns effect to scene {i+1}")
        except Exception as e:
            print(f"  ⚠️  Ken Burns effect failed for scene {i+1}, using static image: {e}")

        # Create text with fade effect
        try:
            txt = create_text_with_fade(t, duration=scene_duration, fade_duration=0.5)
            txt = txt.with_position(("center", 0.82), relative=True)
            print(f"  ✓ Added text with fade to scene {i+1}")
        except Exception as e:
            # Fallback to basic text if fade fails
            print(f"  ⚠️  Text fade failed for scene {i+1}, using basic text: {e}")
            txt = TextClip(text=t, font_size=52, color="white", stroke_color="black", stroke_width=2)
            txt = txt.with_position(("center", 0.82), relative=True).with_duration(scene_duration)

        # Composite and resize
        composite = CompositeVideoClip([img, txt]).resized((1280, 720))
        clips.append(composite)

    print(f"🎬 Concatenating {len(clips)} scenes...")
    # Concatenate all clips
    video = concatenate_videoclips(clips, method="compose")

    # Add audio if file exists
    if os.path.exists(audio_path):
        audio = AudioFileClip(audio_path)
        # Trim audio to match video duration if needed
        if audio.duration > video.duration:
            audio = audio.subclipped(0, video.duration)
        final_video = video.with_audio(audio)
    else:
        print(f"Warning: Audio file {audio_path} not found. Creating video without audio.")
        final_video = video

    final_video.write_videofile(out_path, fps=24, codec="libx264", audio_codec="aac")
