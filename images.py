import os
import warnings
from PIL import Image

# Suppress diffusers/transformers warnings
warnings.filterwarnings('ignore', category=UserWarning, module='diffusers')

# Global pipeline variable (loaded once)
_pipeline = None
_pipeline_load_attempted = False

def _get_pipeline():
    """Lazy load the Stable Diffusion pipeline"""
    global _pipeline, _pipeline_load_attempted

    if _pipeline is None and not _pipeline_load_attempted:
        _pipeline_load_attempted = True
        try:
            # Import directly to avoid AutoPipeline loading all pipelines
            from diffusers import StableDiffusionXLPipeline
            import torch

            print("🎨 Loading Stable Diffusion SDXL-Turbo model...")

            # Use direct pipeline instead of Auto to avoid importing problematic pipelines
            _pipeline = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            )

            # Use Mac GPU if available, otherwise CPU
            if torch.backends.mps.is_available():
                _pipeline.to("mps")
                print("✓ Using Mac GPU (MPS)")
            elif torch.cuda.is_available():
                _pipeline.to("cuda")
                print("✓ Using NVIDIA GPU")
            else:
                _pipeline.to("cpu")
                print("✓ Using CPU")

            print("✓ Stable Diffusion loaded successfully!")

        except Exception as e:
            print(f"❌ Stable Diffusion failed to load: {str(e)[:100]}")
            print("   This is likely a Python 3.13 compatibility issue.")
            _pipeline = False  # Mark as failed

    return _pipeline if _pipeline is not False else None

def generate_image_stub(path: str, prompt: str):
    """Generate an image using local Stable Diffusion"""
    try:
        pipeline = _get_pipeline()

        if pipeline is None:
            raise Exception("Stable Diffusion not available")

        print(f"Generating image: {prompt[:50]}...")

        # Generate image with SDXL-Turbo (only 1 step needed!)
        image = pipeline(
            prompt=prompt,
            num_inference_steps=1,  # Very fast!
            guidance_scale=0.0
        ).images[0]

        # Resize to 1280x720 (16:9)
        image = image.resize((1280, 720), Image.Resampling.LANCZOS)
        image.save(path)

        print(f"✓ Saved to {path}")

    except Exception:
        # Fallback to visually interesting gradient placeholder
        import hashlib
        import numpy as np

        # Generate two colors from prompt hash for gradient
        hash_digest = hashlib.md5(prompt.encode()).hexdigest()
        color_seed_1 = int(hash_digest[:6], 16)
        color_seed_2 = int(hash_digest[6:12], 16)

        # Extract RGB for first color
        r1 = (color_seed_1 >> 16) & 0xFF
        g1 = (color_seed_1 >> 8) & 0xFF
        b1 = color_seed_1 & 0xFF

        # Extract RGB for second color
        r2 = (color_seed_2 >> 16) & 0xFF
        g2 = (color_seed_2 >> 8) & 0xFF
        b2 = color_seed_2 & 0xFF

        # Boost brightness for visibility
        def boost_brightness(r, g, b):
            brightness = (r + g + b) / 3
            if brightness < 100:
                r, g, b = min(r + 100, 255), min(g + 100, 255), min(b + 100, 255)
            return r, g, b

        r1, g1, b1 = boost_brightness(r1, g1, b1)
        r2, g2, b2 = boost_brightness(r2, g2, b2)

        # Create diagonal gradient
        width, height = 1280, 720
        gradient = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                # Diagonal gradient from top-left to bottom-right
                ratio = (x + y) / (width + height)
                gradient[y, x] = [
                    int(r1 + (r2 - r1) * ratio),
                    int(g1 + (g2 - g1) * ratio),
                    int(b1 + (b2 - b1) * ratio)
                ]

        img = Image.fromarray(gradient, 'RGB')
        img.save(path)
