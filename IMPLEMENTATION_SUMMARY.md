# Educational Music Video Implementation

## What Changed

Your AI agent now generates **educational music videos** that narrate webpage content with text-to-speech + background music, instead of just ambient drones with static images.

## Architecture

### 1. **Audio Generation** ([audio.py](audio.py))

**Before:** Boring ambient sine wave drones
**Now:** Professional narrated educational video with music

**New Components:**
- **TTS Narration** (`generate_narration_audio`): Uses pyttsx3 to convert scene text to speech
- **Background Music** (`generate_background_music`): Generates melodic music with:
  - Pentatonic melody that changes every 2 seconds
  - Bass line for depth
  - Hi-hat rhythm pattern
  - Kick drum on beats
  - Mood-based frequencies (calm=220Hz, energetic=440Hz, dark=110Hz, cosmic=330Hz)
- **Audio Mixing** (`mix_audio_tracks`): Narration at 100% volume, music at 25% (background)
- **Dynamic Timing**: Measures actual narration duration, adjusts scene timings automatically

### 2. **Storyboard Generation** ([llm.py](llm.py))

**Updated Prompt:**
- Now generates **narration scripts** instead of captions
- Content structured like an educational video essay (Intro → Key Points → Conclusion)
- Text written to be **spoken aloud** naturally
- Each scene explains one concept from webpage

**Mock Storyboard:**
- Educational narration-style text (e.g., "Welcome! Today we're exploring...")
- Varied durations (3.5-4.5 seconds per scene)

### 3. **Video Assembly** ([video.py](video.py))

**Updated:**
- Now accepts `storyboard` parameter with actual narration durations
- Scene durations sync to TTS timing (not fixed 3.5s)
- Ken Burns zoom effect
- Text fade in/out
- Black stroke on text for readability

### 4. **Agent Orchestration** ([agent.py](agent.py))

**Updated Flow:**
1. Fetch webpage
2. Extract text
3. **Generate storyboard** (educational narration script)
4. Generate images
5. **Generate audio** (TTS + music) → Returns updated storyboard with actual durations
6. **Assemble video** using actual narration timing

## How It Works

```
1. Webpage → Extract key concepts
2. Gemini → Generate educational narration script (8-12 scenes)
3. pyttsx3 → Convert each scene's text to speech
4. NumPy → Generate background music with melody, bass, rhythm
5. Mix → Narration (100%) + Music (25%)
6. Stable Diffusion → Generate visuals for each scene
7. MoviePy → Sync visuals to narration timing + Ken Burns effects
8. Output → Educational music video MP4
```

## Example Scene

**Old:**
- Text: "Scene 1 - cosmic vibe" (instant appearance)
- Audio: Ambient drone (no meaning)
- Duration: Fixed 3.5s

**New:**
- Text: "Welcome! Today we're exploring the fascinating world of cosmic concepts. Let's dive in and discover what makes this topic so interesting."
- Audio: Narrator speaks this text + cosmic background music (330Hz ethereal tones)
- Duration: ~4.2s (measured from actual narration length)
- Visuals: Fade in text with Ken Burns zoom

## Dependencies Added

- `pyttsx3` - Offline text-to-speech (Mac native)

## Testing

Run your agent normally:

```bash
# Start server
python3 main.py

# Generate video
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"url": "YOUR_URL", "vibe": "cosmic"}'
```

You'll now hear actual narration explaining the webpage content!

## Files Modified

- ✅ [audio.py](audio.py) - Complete rewrite with TTS + music generation
- ✅ [llm.py](llm.py) - Updated prompt for narration-style educational scripts
- ✅ [video.py](video.py) - Sync to actual narration timing
- ✅ [agent.py](agent.py) - Pass storyboard for timing sync

## What You'll Notice

✅ Actual voice narrating the webpage content
✅ Background music with melody, not just drones
✅ Scene timing syncs to narration (not fixed intervals)
✅ Educational video essay structure (intro → explanation → conclusion)
✅ Professional polish with text fade, zoom effects, readable text
