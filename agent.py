import os
import hashlib
import json
from tools import fetch_html, extract_main_text
from llm import make_storyboard
from images import generate_image_stub
from video import assemble_video
from audio import generate_audio

def run_agent(url: str, vibe: str, max_retries: int = 3) -> dict:
    """
    Advanced agent with error recovery and retry logic
    """
    # Create output directory
    os.makedirs("out", exist_ok=True)

    # Create cache key from URL + vibe
    cache_key = hashlib.md5(f"{url}_{vibe}".encode()).hexdigest()
    result_cache = f"out/result_{cache_key}.json"
    video_output = f"out/video_{cache_key}.mp4"

    print(f"⚡ Processing request (will reuse cached storyboard if available)...")
    trace = []
    errors = []

    # Step 1: Fetch webpage with retry
    for attempt in range(max_retries):
        try:
            html = fetch_html(url)
            trace.append("fetch_html")
            break
        except Exception as e:
            errors.append(f"fetch_html_attempt_{attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                return {"error": "Failed to fetch webpage", "trace": trace, "errors": errors}
            print(f"⚠️  Fetch failed (attempt {attempt + 1}/{max_retries}), retrying...")

    # Step 2: Extract text with fallback
    try:
        text = extract_main_text(html)
        trace.append("extract_main_text")
    except Exception as e:
        errors.append(f"extract_text: {str(e)}")
        print(f"⚠️  Text extraction failed, using raw HTML...")
        text = html[:5000]  # Fallback: use first 5000 chars of HTML
        trace.append("extract_text_fallback")

    # Step 3: Generate storyboard with retry
    board = None
    for attempt in range(max_retries):
        try:
            board = make_storyboard(text, vibe)
            trace.append("make_storyboard")
            break
        except Exception as e:
            errors.append(f"storyboard_attempt_{attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                return {"error": "Failed to generate storyboard", "trace": trace, "errors": errors}
            print(f"⚠️  Storyboard generation failed (attempt {attempt + 1}/{max_retries}), retrying...")

    # Step 4: Generate images with error recovery
    scene_paths = []
    scene_texts = []
    image_errors = 0

    for i, s in enumerate(board["scenes"]):
        p = f"out/scene_{cache_key}_{i}.png"

        # Skip if image already exists
        if os.path.exists(p):
            print(f"✓ Using existing image: {p}")
            scene_paths.append(p)
            scene_texts.append(s["on_screen_text"])
            continue

        # Try to generate image
        image_prompt = s.get("image_prompt", f"Scene {i}")
        try:
            generate_image_stub(p, image_prompt)
            scene_paths.append(p)
            scene_texts.append(s["on_screen_text"])
        except Exception as e:
            image_errors += 1
            errors.append(f"image_{i}: {str(e)}")
            print(f"⚠️  Image generation failed for scene {i}, using placeholder")
            # Fallback still handled by generate_image_stub
            scene_paths.append(p)
            scene_texts.append(s["on_screen_text"])

    trace.append(f"generate_images (errors: {image_errors})")

    # Step 5: Generate audio (returns updated storyboard with actual durations)
    audio_path = "audio.wav"
    try:
        board = generate_audio(board, audio_path)  # Returns updated storyboard!
        trace.append("generate_audio")
    except Exception as e:
        errors.append(f"audio_generation: {str(e)}")
        print(f"⚠️  Audio generation failed: {e}")
        # Continue without audio - video.py will handle missing audio

    # Step 6: Assemble video with retry (pass storyboard for timing info)
    for attempt in range(max_retries):
        try:
            assemble_video(scene_paths, scene_texts, audio_path, video_output, board)
            trace.append("assemble_video")
            break
        except Exception as e:
            errors.append(f"video_attempt_{attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                return {"error": "Failed to assemble video", "trace": trace, "errors": errors}
            print(f"⚠️  Video assembly failed (attempt {attempt + 1}/{max_retries}), retrying...")

    result = {
        "title": board.get("title"),
        "out_mp4": video_output,
        "trace": trace,
        "storyboard": board,
        "errors": errors if errors else []
    }

    # Cache the result
    with open(result_cache, 'w') as f:
        json.dump(result, f, indent=2)

    if errors:
        print(f"⚠️  Completed with {len(errors)} non-fatal errors")
    else:
        print(f"✓ Completed successfully with no errors!")

    return result
