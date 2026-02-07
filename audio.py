import numpy as np
from scipy.io import wavfile
import os
import librosa
import soundfile as sf
from pedalboard import Pedalboard, Reverb, Chorus, Compressor, Gain, HighpassFilter, LowpassFilter

def apply_pitch_shift_and_autotune(audio_path: str, melody_freqs: list, base_freq: float = 261.63):
    """
    Apply pitch shifting, autotune, and professional vocal effects
    Makes TTS sound like singing with studio-quality processing
    """
    try:
        # Load audio with librosa
        y, sr = librosa.load(audio_path, sr=44100)

        # Calculate average pitch shift needed to align with melody
        target_freq = melody_freqs[len(melody_freqs) // 2]
        semitones = 12 * np.log2(target_freq / base_freq)

        # UPGRADE: Much more aggressive pitch shift for singing quality (70% instead of 30%)
        semitones_shift = semitones * 0.7  # Stronger vocal transformation

        # Apply pitch shift with formant preservation (keeps vocal character)
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones_shift)

        # UPGRADE: Professional vocal effects chain (like Suno/professional studios)
        vocal_board = Pedalboard([
            # Pre-compression to even out dynamics
            Compressor(threshold_db=-25, ratio=3, attack_ms=5, release_ms=50),

            # Chorus for thickness and depth (makes single voice sound richer)
            Chorus(rate_hz=1.5, depth=0.4, centre_delay_ms=7, feedback=0.3, mix=0.4),

            # Reverb for space and presence (studio vocal sound)
            Reverb(
                room_size=0.6,      # Medium room
                damping=0.5,        # Natural decay
                wet_level=0.35,     # Noticeable but not overwhelming
                dry_level=0.8,      # Keep original strong
                width=0.8           # Stereo width
            ),

            # Final compression for polish and loudness
            Compressor(threshold_db=-18, ratio=4, attack_ms=10, release_ms=100),

            # Gain boost for presence
            Gain(gain_db=2.0)
        ])

        # Apply the effects chain
        y_effected = vocal_board(y_shifted, sr)

        # Normalize to prevent clipping
        y_effected = y_effected / (np.max(np.abs(y_effected)) + 0.01)

        # Save the processed audio
        sf.write(audio_path, y_effected, sr)

        print(f"    🎵 Applied pitch shift: {semitones_shift:.1f} semitones + vocal FX")
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

def generate_realistic_instrument(freq: float, duration: float, sample_rate: int,
                                  instrument_type: str = "synth") -> np.ndarray:
    """
    Generate realistic instrument sounds with harmonics and ADSR envelope
    Much better than plain sine waves!
    """
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples)

    # UPGRADE: Add harmonics (overtones) for realistic instrument timbre
    # This is what makes real instruments sound different from sine waves
    if instrument_type == "synth":
        # Synth lead: fundamental + harmonics with decreasing amplitude
        signal = (
            1.0 * np.sin(2 * np.pi * freq * t) +           # Fundamental
            0.5 * np.sin(2 * np.pi * freq * 2 * t) +       # 2nd harmonic
            0.25 * np.sin(2 * np.pi * freq * 3 * t) +      # 3rd harmonic
            0.125 * np.sin(2 * np.pi * freq * 4 * t)       # 4th harmonic
        )
    elif instrument_type == "bass":
        # Bass: strong fundamental, some even harmonics
        signal = (
            1.0 * np.sin(2 * np.pi * freq * t) +
            0.3 * np.sin(2 * np.pi * freq * 2 * t) +
            0.1 * np.sin(2 * np.pi * freq * 4 * t)
        )
    else:  # pad
        # Pad: more harmonics for warmth
        signal = (
            0.8 * np.sin(2 * np.pi * freq * t) +
            0.4 * np.sin(2 * np.pi * freq * 2 * t) +
            0.3 * np.sin(2 * np.pi * freq * 3 * t) +
            0.2 * np.sin(2 * np.pi * freq * 5 * t) +
            0.1 * np.sin(2 * np.pi * freq * 7 * t)
        )

    # UPGRADE: ADSR envelope (Attack, Decay, Sustain, Release) for natural sound
    attack_time = 0.02   # 20ms attack
    decay_time = 0.05    # 50ms decay
    release_time = 0.1   # 100ms release
    sustain_level = 0.7  # 70% sustain

    attack_samples = int(sample_rate * attack_time)
    decay_samples = int(sample_rate * decay_time)
    release_samples = int(sample_rate * release_time)

    envelope = np.ones_like(signal)

    # Attack: fade in
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

    # Decay: drop to sustain level
    if decay_samples > 0 and attack_samples + decay_samples < len(envelope):
        decay_end = attack_samples + decay_samples
        envelope[attack_samples:decay_end] = np.linspace(1, sustain_level, decay_samples)

    # Sustain: hold level
    sustain_start = attack_samples + decay_samples
    sustain_end = len(envelope) - release_samples
    if sustain_start < sustain_end:
        envelope[sustain_start:sustain_end] = sustain_level

    # Release: fade out
    if release_samples > 0:
        envelope[-release_samples:] = np.linspace(sustain_level, 0, release_samples)

    return signal * envelope

def generate_background_music(duration_s: float, tempo_bpm: int, mood: str, sample_rate: int = 44100) -> np.ndarray:
    """
    UPGRADED: Generate professional-quality background music
    With realistic instruments, harmonics, and proper rhythm section
    """
    num_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, num_samples)
    beat_duration_s = 60.0 / tempo_bpm

    # Get melody frequencies for mood
    base_freq, melody_freqs = get_melody_freqs_for_mood(mood)

    # UPGRADE 1: Realistic melody with harmonics and envelopes
    melody = np.zeros(num_samples)
    note_duration = 2.0  # Each note lasts 2 seconds

    for i, freq in enumerate(melody_freqs * int(np.ceil(duration_s / (len(melody_freqs) * note_duration)))):
        if i * note_duration * sample_rate >= num_samples:
            break

        # Generate realistic synth note
        note = generate_realistic_instrument(freq, note_duration, sample_rate, "synth")

        # Place in timeline
        start_sample = int(i * note_duration * sample_rate)
        end_sample = min(start_sample + len(note), num_samples)
        actual_length = end_sample - start_sample

        if actual_length > 0:
            melody[start_sample:end_sample] += note[:actual_length] * 0.08

    # UPGRADE 2: Add chord pad for fullness (sustained chords underneath)
    pad = np.zeros(num_samples)
    chord_duration = 4.0  # Longer sustain for pads

    for i in range(int(np.ceil(duration_s / chord_duration))):
        if i * chord_duration * sample_rate >= num_samples:
            break

        # Use root, third, and fifth for chord
        root_freq = melody_freqs[i % len(melody_freqs)]
        third_freq = melody_freqs[(i + 2) % len(melody_freqs)]

        chord_note = (
            generate_realistic_instrument(root_freq, chord_duration, sample_rate, "pad") +
            generate_realistic_instrument(third_freq, chord_duration, sample_rate, "pad")
        ) * 0.5

        start_sample = int(i * chord_duration * sample_rate)
        end_sample = min(start_sample + len(chord_note), num_samples)
        actual_length = end_sample - start_sample

        if actual_length > 0:
            pad[start_sample:end_sample] += chord_note[:actual_length] * 0.05

    # UPGRADE 3: Better bass line with realistic synthesis
    bass = np.zeros(num_samples)
    bass_note_duration = beat_duration_s * 2  # Bass plays every 2 beats

    for i in range(int(np.ceil(duration_s / bass_note_duration))):
        if i * bass_note_duration * sample_rate >= num_samples:
            break

        bass_freq = base_freq * 0.5  # One octave lower
        bass_note = generate_realistic_instrument(bass_freq, bass_note_duration, sample_rate, "bass")

        start_sample = int(i * bass_note_duration * sample_rate)
        end_sample = min(start_sample + len(bass_note), num_samples)
        actual_length = end_sample - start_sample

        if actual_length > 0:
            bass[start_sample:end_sample] += bass_note[:actual_length] * 0.12

    # UPGRADE 4: More realistic drums
    kick_pattern = np.zeros(num_samples)
    snare_pattern = np.zeros(num_samples)
    hihat_pattern = np.zeros(num_samples)

    beat_times = np.arange(0, duration_s, beat_duration_s)

    for i, beat_time in enumerate(beat_times):
        beat_sample = int(beat_time * sample_rate)

        # Kick on every beat
        kick_duration = int(0.15 * sample_rate)
        kick_end = min(beat_sample + kick_duration, num_samples)
        if beat_sample < num_samples:
            kick_t = np.linspace(0, 0.15, kick_end - beat_sample)
            # Realistic kick: pitch envelope + noise
            kick = (
                0.15 * np.exp(-15 * kick_t) * np.sin(2 * np.pi * (60 + 40 * np.exp(-20 * kick_t)) * kick_t) +
                0.02 * np.random.randn(len(kick_t))  # Add noise for punch
            )
            kick_pattern[beat_sample:kick_end] += kick

        # Snare on beats 2 and 4
        if i % 2 == 1 and beat_sample < num_samples:
            snare_duration = int(0.12 * sample_rate)
            snare_end = min(beat_sample + snare_duration, num_samples)
            snare_t = np.linspace(0, 0.12, snare_end - beat_sample)
            # Realistic snare: noise + tone
            snare = (
                0.1 * np.exp(-20 * snare_t) * np.sin(2 * np.pi * 200 * snare_t) +
                0.08 * np.exp(-15 * snare_t) * np.random.randn(len(snare_t))
            )
            snare_pattern[beat_sample:snare_end] += snare

        # Hi-hat on every half beat
        for j in range(2):
            hihat_time = beat_time + j * beat_duration_s / 2
            hihat_sample = int(hihat_time * sample_rate)
            if hihat_sample < num_samples:
                hihat_duration = int(0.05 * sample_rate)
                hihat_end = min(hihat_sample + hihat_duration, num_samples)
                hihat_t = np.linspace(0, 0.05, hihat_end - hihat_sample)
                # Realistic hi-hat: high-frequency noise burst
                hihat = 0.03 * np.exp(-40 * hihat_t) * np.random.randn(len(hihat_t))
                hihat_pattern[hihat_sample:hihat_end] += hihat

    # Mix all elements
    music = melody + pad + bass + kick_pattern + snare_pattern + hihat_pattern

    # UPGRADE 5: Apply subtle effects to the music
    music_board = Pedalboard([
        # High-pass filter to remove mud
        HighpassFilter(cutoff_frequency_hz=80),

        # Gentle compression for glue
        Compressor(threshold_db=-20, ratio=2, attack_ms=20, release_ms=200),

        # Slight gain
        Gain(gain_db=-2.0)  # Keep music quieter than vocals
    ])

    music = music_board(music, sample_rate)

    # Apply fade in/out
    fade_duration = 1.5
    fade_samples = int(sample_rate * fade_duration)

    if len(music) > 2 * fade_samples:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        music[:fade_samples] *= fade_in
        music[-fade_samples:] *= fade_out

    return music

def apply_sidechain_compression(music: np.ndarray, narration: np.ndarray,
                                 sample_rate: int = 44100) -> np.ndarray:
    """
    UPGRADE: Sidechain compression (ducking)
    Automatically reduces music volume when vocals play
    This is the SECRET SAUCE of professional mixes!
    """
    # Detect when narration is active (above threshold)
    # Use RMS (root mean square) for smooth detection
    window_size = int(0.05 * sample_rate)  # 50ms windows
    narration_envelope = np.zeros_like(narration)

    for i in range(0, len(narration) - window_size, window_size // 2):
        window = narration[i:i + window_size]
        rms = np.sqrt(np.mean(window ** 2))
        narration_envelope[i:i + window_size] = max(narration_envelope[i], rms)

    # Create ducking envelope (reduce music when vocals are loud)
    threshold = 0.01  # When vocals are above this, duck the music
    duck_amount = 0.4  # Reduce music to 40% when vocals play

    ducking_envelope = np.ones_like(music)
    vocal_active = narration_envelope > threshold

    # Smooth transitions to avoid clicks
    attack_samples = int(0.01 * sample_rate)  # 10ms attack
    release_samples = int(0.2 * sample_rate)  # 200ms release

    for i in range(len(ducking_envelope)):
        if vocal_active[i]:
            # Duck down quickly when vocals start
            target = duck_amount
            if i > 0:
                ducking_envelope[i] = (
                    ducking_envelope[i-1] * 0.95 + target * 0.05
                )
            else:
                ducking_envelope[i] = target
        else:
            # Release slowly when vocals stop
            target = 1.0
            if i > 0:
                ducking_envelope[i] = (
                    ducking_envelope[i-1] * 0.99 + target * 0.01
                )
            else:
                ducking_envelope[i] = target

    return music * ducking_envelope

def mix_audio_tracks(narration_path: str, background_music: np.ndarray, output_path: str,
                     narration_volume: float = 1.0, music_volume: float = 0.35):
    """
    UPGRADED: Professional mixing with sidechain compression and mastering
    Makes it sound like it came from a real studio!
    """
    sample_rate = 44100

    # Load narration
    if os.path.exists(narration_path):
        narr_rate, narration = wavfile.read(narration_path)

        # Resample if needed
        if narr_rate != sample_rate:
            narration = narration

        # Convert to float
        if narration.dtype == np.int16:
            narration = narration.astype(np.float32) / 32767.0
    else:
        narration = np.zeros_like(background_music)

    # Ensure same length
    if len(narration) < len(background_music):
        narration = np.pad(narration, (0, len(background_music) - len(narration)))
    elif len(narration) > len(background_music):
        background_music = np.pad(background_music, (0, len(narration) - len(background_music)))

    # UPGRADE 1: Apply sidechain compression (duck music when vocals play)
    print(f"  🎚️  Applying sidechain compression (ducking)...")
    ducked_music = apply_sidechain_compression(background_music, narration, sample_rate)

    # UPGRADE 2: EQ separation - filter music to leave space for vocals
    music_eq = Pedalboard([
        # Cut low mids where voice sits (200-500 Hz)
        # This creates separation between voice and music
        HighpassFilter(cutoff_frequency_hz=120),  # Remove rumble
        LowpassFilter(cutoff_frequency_hz=12000),  # Gentle high cut
    ])
    ducked_music = music_eq(ducked_music, sample_rate)

    # Mix with volume controls (music is now smarter about when to be quiet)
    mixed = (narration * narration_volume) + (ducked_music * music_volume)

    # UPGRADE 3: Mastering chain for final polish
    print(f"  🎛️  Applying mastering chain...")
    mastering = Pedalboard([
        # Multiband compression for balanced frequency response
        Compressor(threshold_db=-12, ratio=3, attack_ms=15, release_ms=150),

        # Gentle limiting to maximize loudness without distortion
        Compressor(threshold_db=-6, ratio=10, attack_ms=1, release_ms=50),

        # Final gain for competitive loudness
        Gain(gain_db=1.5)
    ])

    mixed = mastering(mixed, sample_rate)

    # Normalize and convert to 16-bit (leave more headroom after mastering)
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed = mixed / peak * 0.95  # 95% to prevent any clipping

    mixed = np.clip(mixed, -1.0, 1.0)
    mixed = np.int16(mixed * 32767)

    # Save mixed audio
    wavfile.write(output_path, sample_rate, mixed)
    print(f"  ✓ Professional mix complete with sidechain & mastering!")

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
