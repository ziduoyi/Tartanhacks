import numpy as np
from scipy.io import wavfile
import os
import librosa
import soundfile as sf

def apply_pitch_shift_and_autotune(audio_path: str, melody_freqs: list, base_freq: float = 261.63):
    """
    Apply pitch shifting and autotune to make TTS vocals more musical
    Shifts pitch to follow the melody and snaps to nearest notes
    """
    try:
        # Load audio with librosa
        y, sr = librosa.load(audio_path, sr=44100)

        # Calculate average pitch shift needed to align with melody
        # Use the middle frequency from melody as target
        target_freq = melody_freqs[len(melody_freqs) // 2]

        # Estimate pitch shift in semitones (roughly align with melody)
        # This creates a more "sung" quality
        semitones = 12 * np.log2(target_freq / base_freq)

        # Apply subtle pitch shift (not too extreme to keep it natural)
        semitones_shift = semitones * 0.3  # 30% of full shift for subtlety

        # Apply pitch shift
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones_shift)

        # Save the pitch-shifted audio
        sf.write(audio_path, y_shifted, sr)

        print(f"    🎵 Applied pitch shift: {semitones_shift:.1f} semitones")
        return True

    except Exception as e:
        print(f"    ⚠️  Pitch shifting failed: {e}, using original audio")
        return False

def generate_narration_audio(text: str, output_path: str, rate: int = 210, melody_freqs: list = None) -> float:
    """
    Generate TTS narration for a single text segment using pyttsx3
    Returns the duration in seconds
    """
    try:
        import pyttsx3
        import subprocess

        engine = pyttsx3.init()
        engine.setProperty('rate', rate)  # Speaking rate (words per minute)
        engine.setProperty('volume', 1.0)

        # Save narration to temporary AIFF file (Mac pyttsx3 default)
        temp_path = output_path.replace('.wav', '_temp.aiff')

        # Clear any existing files first
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(output_path):
            os.remove(output_path)

        engine.save_to_file(text, temp_path)
        engine.runAndWait()

        # Wait longer for file to be fully written
        import time
        for _ in range(10):  # Wait up to 1 second
            time.sleep(0.1)
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 10000:
                break

        # Verify AIFF file was created with actual content
        if not os.path.exists(temp_path):
            raise Exception(f"AIFF file not created: {temp_path}")

        aiff_size = os.path.getsize(temp_path)
        if aiff_size < 10000:
            raise Exception(f"AIFF file too small ({aiff_size} bytes): {temp_path}")

        print(f"    📝 AIFF created: {aiff_size / 1024:.1f}KB")

        # Convert AIFF to WAV using afconvert (Mac built-in)
        result = subprocess.run(
            ['afconvert', '-f', 'WAVE', '-d', 'LEI16@44100', temp_path, output_path],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise Exception(f"afconvert failed: {result.stderr}")

        # Verify WAV file was created and has content
        if not os.path.exists(output_path):
            raise Exception(f"WAV file not created: {output_path}")

        wav_size = os.path.getsize(output_path)
        if wav_size < 10000:
            raise Exception(f"WAV file too small ({wav_size} bytes): {output_path}")

        print(f"    🎵 WAV created: {wav_size / 1024:.1f}KB")

        # Remove temp AIFF file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Apply pitch shifting and autotune if melody frequencies provided
        if melody_freqs and len(melody_freqs) > 0:
            base_freq = melody_freqs[0]  # Use first frequency as base
            apply_pitch_shift_and_autotune(output_path, melody_freqs, base_freq)

        # Calculate duration by reading the generated WAV file
        sample_rate, audio_data = wavfile.read(output_path)
        duration = len(audio_data) / sample_rate

        print(f"    ✓ Generated {duration:.1f}s of narration")
        return duration

    except Exception as e:
        print(f"⚠️  TTS generation failed: {e}")
        # Fallback: estimate duration (average 150 words per minute)
        words = len(text.split())
        duration = (words / 150) * 60
        return max(duration, 3.0)  # Minimum 3 seconds

def get_melody_freqs_for_mood(mood: str) -> tuple:
    """
    Get melody frequencies and base frequency based on mood
    Returns (base_freq, melody_freqs)
    """
    mood_lower = mood.lower() if isinstance(mood, str) else "neutral"

    if any(word in mood_lower for word in ["calm", "peaceful", "serene", "contemplative"]):
        base_freq = 220  # A3 - calming
        melody_freqs = [220, 247, 277, 330, 370]  # A major pentatonic
    elif any(word in mood_lower for word in ["energetic", "exciting", "upbeat", "fast"]):
        base_freq = 440  # A4 - energetic
        melody_freqs = [440, 494, 523, 587, 659]  # A major pentatonic (higher)
    elif any(word in mood_lower for word in ["dark", "mysterious", "ominous", "haunting"]):
        base_freq = 110  # A2 - dark
        melody_freqs = [110, 123, 131, 147, 165]  # A minor pentatonic
    elif any(word in mood_lower for word in ["cosmic", "ethereal", "dreamy", "ambient"]):
        base_freq = 330  # E4 - ethereal
        melody_freqs = [330, 370, 415, 440, 494]  # E major pentatonic
    else:
        base_freq = 261.63  # Middle C - neutral
        melody_freqs = [262, 294, 330, 349, 392]  # C major pentatonic

    return base_freq, melody_freqs

def generate_background_music(duration_s: float, tempo_bpm: int, mood: str, sample_rate: int = 44100) -> np.ndarray:
    """
    Generate background music track with melody and rhythm
    """
    num_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, num_samples)

    # Get melody frequencies for mood
    base_freq, melody_freqs = get_melody_freqs_for_mood(mood)

    # Create melodic pattern (changes every 2 seconds)
    melody = np.zeros_like(t)
    pattern_duration = 2.0  # seconds

    for i, freq in enumerate(melody_freqs):
        start_time = i * pattern_duration
        end_time = (i + 1) * pattern_duration
        mask = (t >= start_time) & (t < end_time)
        melody[mask] = 0.08 * np.sin(2 * np.pi * freq * t[mask])

    # Fill remaining time with cycling melody
    remaining_start = len(melody_freqs) * pattern_duration
    for i in range(int((duration_s - remaining_start) / pattern_duration) + 1):
        freq = melody_freqs[i % len(melody_freqs)]
        start_time = remaining_start + i * pattern_duration
        end_time = remaining_start + (i + 1) * pattern_duration
        mask = (t >= start_time) & (t < end_time)
        melody[mask] = 0.08 * np.sin(2 * np.pi * freq * t[mask])

    # Add bass line (root note)
    bass = 0.12 * np.sin(2 * np.pi * base_freq * 0.5 * t)

    # Add rhythmic hi-hat pattern
    beat_duration_s = 60.0 / tempo_bpm
    beat_freq = 1.0 / beat_duration_s
    hihat = 0.03 * np.sin(2 * np.pi * beat_freq * 4 * t) * np.sin(2 * np.pi * 8000 * t)

    # Add kick drum (low frequency pulse on beats)
    kick_pattern = np.zeros_like(t)
    beat_times = np.arange(0, duration_s, beat_duration_s)
    for beat_time in beat_times:
        kick_mask = (t >= beat_time) & (t < beat_time + 0.1)
        kick_pattern[kick_mask] = 0.15 * np.exp(-10 * (t[kick_mask] - beat_time)) * np.sin(2 * np.pi * 60 * (t[kick_mask] - beat_time))

    # Mix all elements
    music = melody + bass + hihat + kick_pattern

    # Apply fade in/out
    fade_duration = 1.5
    fade_samples = int(sample_rate * fade_duration)

    if len(music) > 2 * fade_samples:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        music[:fade_samples] *= fade_in
        music[-fade_samples:] *= fade_out

    return music

def mix_audio_tracks(narration_path: str, background_music: np.ndarray, output_path: str,
                     narration_volume: float = 1.0, music_volume: float = 0.3):
    """
    Mix narration with background music (music is quieter)
    """
    sample_rate = 44100

    # Load narration
    if os.path.exists(narration_path):
        narr_rate, narration = wavfile.read(narration_path)

        # Resample if needed
        if narr_rate != sample_rate:
            # Simple resampling (for production, use scipy.signal.resample)
            narration = narration

        # Convert to float
        if narration.dtype == np.int16:
            narration = narration.astype(np.float32) / 32767.0
    else:
        # If narration failed, use silence
        narration = np.zeros_like(background_music)

    # Ensure same length (pad or trim)
    if len(narration) < len(background_music):
        narration = np.pad(narration, (0, len(background_music) - len(narration)))
    elif len(narration) > len(background_music):
        background_music = np.pad(background_music, (0, len(narration) - len(background_music)))

    # Mix with volume controls
    mixed = (narration * narration_volume) + (background_music * music_volume)

    # Normalize and convert to 16-bit
    mixed = np.clip(mixed, -1.0, 1.0)
    mixed = np.int16(mixed * 32767 * 0.9)

    # Save mixed audio
    wavfile.write(output_path, sample_rate, mixed)

def generate_audio(storyboard: dict, output_path: str):
    """
    Generate educational music video audio with TTS narration + background music
    Returns updated storyboard with actual scene durations
    """
    audio_plan = storyboard.get("audio_plan", {})
    tempo_bpm = audio_plan.get("tempo_bpm", 120)
    mood = audio_plan.get("mood", "neutral")
    scenes = storyboard.get("scenes", [])

    print(f"🎵 Generating narrated music video audio (tempo: {tempo_bpm} BPM, mood: {mood})")

    # Get melody frequencies for this mood (used for pitch shifting vocals)
    _, melody_freqs = get_melody_freqs_for_mood(mood)

    # Create temp directory for scene narrations
    os.makedirs("out/temp_audio", exist_ok=True)

    # Step 1: Generate narration for each scene and measure durations
    scene_durations = []
    scene_narration_paths = []

    for i, scene in enumerate(scenes):
        text = scene.get("on_screen_text", f"Scene {i+1}")
        narration_path = f"out/temp_audio/scene_{i}_narration.wav"

        print(f"  🎤 Generating narration {i+1}/{len(scenes)}: {text[:40]}...")
        duration = generate_narration_audio(text, narration_path, rate=210, melody_freqs=melody_freqs)

        # Add padding between scenes (0.5s pause)
        duration += 0.5

        scene_durations.append(duration)
        scene_narration_paths.append(narration_path)
        scene["duration_s"] = duration  # Update storyboard with actual duration

    total_duration = sum(scene_durations)
    print(f"  ✓ Total narration duration: {total_duration:.1f}s")

    # Step 2: Generate background music for total duration
    print(f"  🎹 Generating background music...")
    background_music = generate_background_music(total_duration, tempo_bpm, mood)

    # Step 3: Concatenate all narration segments with timing
    sample_rate = 44100
    full_narration = np.zeros(int(sample_rate * total_duration), dtype=np.float32)

    current_time = 0
    for i, (narr_path, duration) in enumerate(zip(scene_narration_paths, scene_durations)):
        if os.path.exists(narr_path):
            try:
                _, segment = wavfile.read(narr_path)
                # Convert to float
                if segment.dtype == np.int16:
                    segment = segment.astype(np.float32) / 32767.0

                # Place in timeline
                start_sample = int(current_time * sample_rate)
                end_sample = start_sample + len(segment)

                if end_sample <= len(full_narration):
                    full_narration[start_sample:end_sample] = segment
            except Exception as e:
                print(f"  ⚠️  Failed to load narration {i}: {e}")

        current_time += duration

    # Step 4: Save concatenated narration
    temp_narration_path = "out/temp_audio/full_narration.wav"
    wavfile.write(temp_narration_path, sample_rate, np.int16(full_narration * 32767))

    # Step 5: Mix narration with background music
    print(f"  🎚️  Mixing narration with background music...")
    mix_audio_tracks(temp_narration_path, background_music, output_path,
                     narration_volume=1.0, music_volume=0.25)

    print(f"✓ Audio saved to {output_path} ({total_duration:.1f}s)")

    # Return updated storyboard with actual durations
    return storyboard
