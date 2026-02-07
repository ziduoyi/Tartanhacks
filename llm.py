import os, json, hashlib
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def critique_storyboard(storyboard: dict, text: str, vibe: str) -> dict:
    """Agent critiques its own storyboard"""
    print("🔍 Agent is critiquing its own work...")

    critique_prompt = f"""
You are a creative director reviewing a music video storyboard.

Storyboard:
{json.dumps(storyboard, indent=2)}

Original content: {text[:500]}...
Intended vibe: {vibe}

Evaluate the storyboard on:
1. Narrative coherence - Does it tell a story?
2. Scene variety - Are scenes visually diverse?
3. Text quality - Are captions engaging and concise?
4. Vibe alignment - Does it match the '{vibe}' aesthetic?
5. Visual consistency - Is the style unified?

Return JSON with:
{{"score": 0-10, "issues": ["issue1", ...], "strengths": ["strength1", ...], "needs_improvement": true/false}}
"""

    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(critique_prompt)

    # Extract JSON
    text_response = response.text
    if '```json' in text_response:
        text_response = text_response.split('```json')[1].split('```')[0].strip()
    elif '```' in text_response:
        text_response = text_response.split('```')[1].split('```')[0].strip()

    critique = json.loads(text_response)
    print(f"✓ Critique complete - Score: {critique.get('score', 0)}/10")

    return critique

def improve_storyboard(original: dict, critique: dict, text: str, vibe: str) -> dict:
    """Agent improves storyboard based on critique"""
    print("✨ Agent is improving the storyboard...")

    improve_prompt = f"""
Original storyboard:
{json.dumps(original, indent=2)}

Critique identified these issues:
{json.dumps(critique.get('issues', []), indent=2)}

Create an IMPROVED version that:
- Addresses all identified issues
- Maintains the strengths: {critique.get('strengths', [])}
- Better captures the '{vibe}' vibe
- Uses content from: {text[:500]}...

Return the same JSON structure as before:
- title (string)
- scenes (array of 8-12 objects), each with:
  - duration_s (number)
  - on_screen_text (string, <= 12 words, original)
  - image_prompt (string, describes a visual scene, consistent style)
- audio_plan with tempo_bpm (number) and mood (string)
"""

    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(improve_prompt)

    # Extract JSON
    text_response = response.text
    if '```json' in text_response:
        text_response = text_response.split('```json')[1].split('```')[0].strip()
    elif '```' in text_response:
        text_response = text_response.split('```')[1].split('```')[0].strip()

    improved = json.loads(text_response)
    print("✓ Improved storyboard created")

    return improved

def make_storyboard(text: str, vibe: str) -> dict:
    # Create cache directory if it doesn't exist
    os.makedirs("cache", exist_ok=True)

    # Create cache key from text + vibe
    cache_key = hashlib.md5(f"{text[:1000]}_{vibe}".encode()).hexdigest()
    cache_file = f"cache/{cache_key}.json"

    # Check if cached result exists
    if os.path.exists(cache_file):
        print(f"✓ Using cached storyboard (cache/{cache_key[:8]}...)")
        with open(cache_file, 'r') as f:
            return json.load(f)

    # Check if API calls are enabled
    enable_api = os.getenv("ENABLE_API", "true").lower() == "true"

    if not enable_api:
        print("⏭️  API calls disabled (ENABLE_API=false) - Creating storyboard from webpage content")

        # Extract title from first line or beginning of text
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        title = lines[0][:100] if lines else "Educational Overview"

        # Extract meaningful content chunks
        # Try paragraphs first (better for article content)
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]

        # If not enough paragraphs, split by sentences
        if len(paragraphs) < 5:
            # Split by period, question mark, or exclamation
            import re
            sentences = re.split(r'[.!?]+', text)
            paragraphs = [s.strip() for s in sentences if len(s.strip()) > 50]

        # Clean and prepare content chunks (2-3 sentences each for good narration)
        content_chunks = []
        for para in paragraphs[:15]:  # Take up to 15 chunks
            # Clean whitespace and limit length
            cleaned = ' '.join(para.split())
            if len(cleaned) > 50:  # Skip very short chunks
                # Aim for 200-500 characters per narration (comfortable speaking length)
                content_chunks.append(cleaned[:500])

        # Ensure we have at least 8 scenes
        while len(content_chunks) < 8:
            # If not enough content, split the text into chunks
            chunk_size = len(text) // (8 - len(content_chunks))
            for i in range(len(content_chunks), 8):
                start = i * chunk_size
                chunk = text[start:start + chunk_size].strip()
                if chunk:
                    content_chunks.append(' '.join(chunk.split())[:500])

        # Create educational narration scenes
        scenes = []

        # Scene 1: Introduction
        intro = content_chunks[0] if content_chunks else text[:400]
        scenes.append({
            "duration_s": 5.0,
            "on_screen_text": f"Welcome! Today we're diving into an fascinating topic. {intro}",
            "image_prompt": f"A {vibe} opening scene with introduction visual elements and welcoming atmosphere"
        })

        # Scenes 2-7: Main content (actual webpage information)
        for i in range(1, min(7, len(content_chunks))):
            scenes.append({
                "duration_s": 5.0 if i % 2 == 1 else 5.5,
                "on_screen_text": content_chunks[i],
                "image_prompt": f"A {vibe} scene illustrating: {content_chunks[i][:80]}"
            })

        # Scene 8: Conclusion
        if len(content_chunks) > 7:
            conclusion = content_chunks[7]
        elif len(content_chunks) > 1:
            conclusion = content_chunks[-1]
        else:
            conclusion = "That wraps up our overview of this topic."

        scenes.append({
            "duration_s": 5.0,
            "on_screen_text": f"To sum it all up: {conclusion}",
            "image_prompt": f"A {vibe} conclusion scene with summary elements and closing atmosphere"
        })

        result = {
            "title": f"{title} - Explained",
            "scenes": scenes,
            "audio_plan": {
                "tempo_bpm": 110,
                "mood": vibe
            }
        }

        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"✓ Mock storyboard cached to {cache_file}")

        return result

    print(f"⚡ Generating new storyboard with Gemini...")

    prompt = f"""
Create a storyboard for a MUSIC VIDEO with SONG LYRICS that summarize the webpage content.

The video will use TEXT-TO-SPEECH VOCALS with background music (like a real song), so:
- Each scene's "on_screen_text" will be SUNG/NARRATED ALOUD with music
- Write it like SONG LYRICS - poetic, rhythmic, flowing
- Use rhyme, metaphor, and poetic language (not dry educational text)
- Make it catchy and memorable like a real song
- Structure like a song: Verse 1 → Chorus → Verse 2 → Bridge → Chorus/Outro

Vibe: {vibe}

LYRICS STYLE GUIDE:
- Use rhythm and flow (like rap, pop, or poetry)
- Add rhymes where natural (but don't force it)
- Keep lines concise (5-12 words each)
- Use emotional, engaging language
- Make it sound like something you'd actually sing/perform

Example structure:
Scene 1 (Intro/Hook): Catchy opening line that grabs attention
Scenes 2-4 (Verse 1): First key points in lyrical form
Scene 5 (Chorus): Catchy, memorable summary
Scenes 6-8 (Verse 2): More key points
Scene 9 (Bridge): Unique perspective or transition
Scenes 10-12 (Outro/Chorus): Memorable ending

Return valid JSON with:
- title (string) - A catchy song title (not "explained" or "overview")
- scenes (array of 8-12 objects), each with:
  - duration_s (number) - Initial estimate, will be adjusted based on vocal timing
  - on_screen_text (string) - SONG LYRICS (1-2 lines that will be sung, like actual song lyrics)
  - image_prompt (string) - Visual that matches the vibe and lyrics, consistent {vibe} style
- audio_plan with tempo_bpm (number, 90-140) and mood (string matching {vibe})

IMPORTANT - Write it like you're a songwriter:
- Make it flow like actual song lyrics (rhythm, cadence, emotion)
- Use poetic devices (metaphor, imagery, rhyme)
- Keep each line singable (not too wordy)
- Capture the essence emotionally, not just facts
- Think: "Would this sound good with music?"

Webpage content to transform into song:
{text}
"""
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)

    # Extract JSON from response (save original text first)
    webpage_text = text  # Save the original text parameter
    response_text = response.text
    # Sometimes the response wraps JSON in markdown code blocks
    if '```json' in response_text:
        response_text = response_text.split('```json')[1].split('```')[0].strip()
    elif '```' in response_text:
        response_text = response_text.split('```')[1].split('```')[0].strip()

    result = json.loads(response_text)

    # Self-critique and improvement (agentic behavior!)
    # Can be disabled via ENABLE_CRITIQUE=false in .env
    enable_critique = os.getenv("ENABLE_CRITIQUE", "true").lower() == "true"

    if enable_critique:
        critique = critique_storyboard(result, webpage_text, vibe)

        # If score is low or has issues, improve it
        if critique.get("score", 10) < 7 or critique.get("needs_improvement", False):
            print(f"⚠️  Initial score: {critique.get('score')}/10 - Improving...")
            result = improve_storyboard(result, critique, webpage_text, vibe)

            # Re-critique to verify improvement
            final_critique = critique_storyboard(result, webpage_text, vibe)
            print(f"✓ Final score: {final_critique.get('score')}/10")
        else:
            print(f"✓ Initial storyboard approved (score: {critique.get('score')}/10)")
    else:
        print("⏭️  Self-critique disabled (ENABLE_CRITIQUE=false)")

    # Save to cache
    with open(cache_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"✓ Storyboard cached to {cache_file}")

    return result
