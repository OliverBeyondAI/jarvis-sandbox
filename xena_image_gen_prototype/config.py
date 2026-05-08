"""
Configuration module for Xena Image Generation Prototype.

Defines:
- SMB profile (hypothetical business context for Xena)
- Marketing image prompts for common use cases
- Provider settings for OpenAI and Google GenAI
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY", "")

OPENAI_IMAGE_MODEL = "gpt-image-1"
GOOGLE_IMAGE_MODEL = "imagen-3.0-generate-002"

OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "outputs")
OPENAI_OUTPUT_DIR = os.path.join(OUTPUT_BASE, "openai")
GEMINI_OUTPUT_DIR = os.path.join(OUTPUT_BASE, "gemini")

# ---------------------------------------------------------------------------
# SMB profile — hypothetical business that Xena serves
# ---------------------------------------------------------------------------

@dataclass
class SMBProfile:
    """Represents a small/medium business using Xena for marketing."""
    business_name: str
    industry: str
    tagline: str
    brand_colors: list[str]
    target_audience: str
    tone: str
    logo_description: str
    products_services: list[str]


XENA_SMB_PROFILE = SMBProfile(
    business_name="Greenleaf Wellness Studio",
    industry="Health & Wellness",
    tagline="Nourish your body, calm your mind",
    brand_colors=["#4A7C59", "#F5E6CC", "#2C3E50", "#FFFFFF"],
    target_audience="Health-conscious professionals aged 28-45 in urban areas",
    tone="Warm, inviting, professional, nature-inspired",
    logo_description="A minimalist green leaf with a subtle wellness circle motif",
    products_services=[
        "Yoga and meditation classes",
        "Nutritional coaching",
        "Wellness retreats",
        "Online mindfulness courses",
        "Organic smoothie bar",
    ],
)

# ---------------------------------------------------------------------------
# Marketing image prompts — organized by use case
# ---------------------------------------------------------------------------

@dataclass
class ImagePrompt:
    """A single marketing image prompt with metadata."""
    use_case: str
    title: str
    prompt: str
    aspect_ratio: str = "1:1"
    style_notes: str = ""


MARKETING_PROMPTS: list[ImagePrompt] = [
    # --- Social media posts ---
    ImagePrompt(
        use_case="social_media",
        title="Instagram — Morning Yoga Session",
        prompt=(
            "A serene morning yoga session in a sunlit studio with large windows, "
            "natural wood floors, and lush green plants. A diverse group of people "
            "practicing tree pose. Soft golden-hour light streaming in. Warm, inviting "
            "atmosphere with earth tones and green accents. Professional lifestyle "
            "photography style, high resolution."
        ),
        aspect_ratio="1:1",
        style_notes="Lifestyle photography, warm tones, aspirational",
    ),
    ImagePrompt(
        use_case="social_media",
        title="Instagram — Organic Smoothie Bowl",
        prompt=(
            "A beautifully arranged acai smoothie bowl topped with fresh berries, "
            "granola, coconut flakes, and edible flowers, placed on a natural wood "
            "table with a sprig of mint. Overhead flat-lay shot. Soft natural lighting. "
            "Clean, minimal background with a linen napkin and small potted succulent. "
            "Food photography style, vibrant colors."
        ),
        aspect_ratio="1:1",
        style_notes="Flat-lay food photography, vibrant, clean",
    ),
    ImagePrompt(
        use_case="social_media",
        title="Facebook — Community Event Announcement",
        prompt=(
            "A welcoming community wellness event in an outdoor garden setting. "
            "String lights, yoga mats arranged in a circle, a wooden stage with "
            "a small podium. Sunset colors in the sky. People chatting and smiling. "
            "Warm and inclusive atmosphere. Professional event photography."
        ),
        aspect_ratio="16:9",
        style_notes="Event photography, warm sunset palette, community feel",
    ),
    ImagePrompt(
        use_case="social_media",
        title="LinkedIn — Team Spotlight",
        prompt=(
            "A portrait of a wellness coach in a modern studio, smiling warmly at "
            "the camera. Clean, bright background with soft bokeh of green plants. "
            "Professional attire with a natural, approachable look. Corporate headshot "
            "style with lifestyle warmth. Studio lighting."
        ),
        aspect_ratio="1:1",
        style_notes="Professional headshot, approachable, bright",
    ),

    # --- Blog headers ---
    ImagePrompt(
        use_case="blog_header",
        title="Blog — Benefits of Morning Meditation",
        prompt=(
            "A peaceful zen garden scene at dawn with smooth stones, raked sand "
            "patterns, a single bonsai tree, and soft morning mist. Pastel colors "
            "with hints of green and gold. Wide cinematic composition. Calm, "
            "contemplative mood. Digital art with photorealistic textures."
        ),
        aspect_ratio="16:9",
        style_notes="Zen aesthetic, cinematic wide, contemplative",
    ),
    ImagePrompt(
        use_case="blog_header",
        title="Blog — Nutrition Tips for Busy Professionals",
        prompt=(
            "A modern kitchen countertop with a neatly arranged meal-prep scene: "
            "glass containers with colorful healthy meals, fresh vegetables, a cutting "
            "board with herbs, and a recipe notebook. Bright, clean, organized. "
            "Natural daylight from a side window. Editorial food photography."
        ),
        aspect_ratio="16:9",
        style_notes="Editorial, organized, aspirational lifestyle",
    ),
    ImagePrompt(
        use_case="blog_header",
        title="Blog — Finding Balance in a Digital World",
        prompt=(
            "A split-composition image: the left half shows a cluttered desk with "
            "glowing screens and notifications, the right half shows a calm nature "
            "scene with a person sitting peacefully by a lake. The transition between "
            "halves is a smooth gradient. Conceptual illustration style."
        ),
        aspect_ratio="16:9",
        style_notes="Conceptual, split composition, contrast theme",
    ),

    # --- Product announcements ---
    ImagePrompt(
        use_case="product_announcement",
        title="New Online Course Launch",
        prompt=(
            "A modern laptop on a clean desk displaying a beautiful wellness course "
            "landing page. Surrounding the laptop are a cup of herbal tea, a journal, "
            "and a small plant. Soft gradient background blending from forest green to "
            "cream. Clean product-shot aesthetic with gentle shadows. Marketing mockup."
        ),
        aspect_ratio="16:9",
        style_notes="Product mockup, clean, premium feel",
    ),
    ImagePrompt(
        use_case="product_announcement",
        title="Wellness Retreat Weekend",
        prompt=(
            "An idyllic countryside retreat setting: a rustic-modern cabin surrounded "
            "by rolling green hills, a yoga deck overlooking a misty valley at sunrise. "
            "Warm golden light. A few people in comfortable athleisure walking along "
            "a nature path. Travel editorial photography, aspirational."
        ),
        aspect_ratio="16:9",
        style_notes="Travel editorial, aspirational, golden hour",
    ),
    ImagePrompt(
        use_case="product_announcement",
        title="Seasonal Smoothie Menu",
        prompt=(
            "Three artfully arranged smoothies in glass bottles with natural "
            "ingredient garnishes — one green (spinach, kiwi), one pink (strawberry, "
            "beet), one golden (mango, turmeric). Set against a marble countertop "
            "with scattered fresh ingredients. Bright studio lighting. Premium "
            "beverage photography with condensation details."
        ),
        aspect_ratio="1:1",
        style_notes="Beverage photography, studio-lit, premium",
    ),

    # --- Email marketing ---
    ImagePrompt(
        use_case="email_marketing",
        title="Monthly Newsletter Hero",
        prompt=(
            "An abstract wellness-themed background with soft flowing shapes in "
            "forest green, warm cream, and gentle gold. Organic curves suggesting "
            "leaves and gentle waves. Plenty of negative space for text overlay. "
            "Modern, minimal graphic design."
        ),
        aspect_ratio="3:1",
        style_notes="Abstract background, text-overlay-friendly, brand colors",
    ),
    ImagePrompt(
        use_case="email_marketing",
        title="Seasonal Promotion Banner",
        prompt=(
            "A cozy autumn wellness scene: a warm-toned studio with candles, "
            "a soft blanket draped over a meditation cushion, dried eucalyptus "
            "arrangement, and warm amber lighting. Left third of image has open "
            "space for promotional text. Lifestyle photography, editorial."
        ),
        aspect_ratio="2:1",
        style_notes="Seasonal, cozy, editorial with text space",
    ),
]


def sanitize_filename(title: str) -> str:
    """Convert a prompt title into a safe filename."""
    return (
        title.lower()
        .replace(" — ", "_")
        .replace("—", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "")
        .replace(",", "")
        .strip("_")
    )


def build_enhanced_prompt(prompt_obj: ImagePrompt) -> str:
    """Enhance the base prompt with brand context for richer results."""
    brand_context = (
        f'For the brand "{XENA_SMB_PROFILE.business_name}" — '
        f"{XENA_SMB_PROFILE.tagline}. {XENA_SMB_PROFILE.tone} aesthetic."
    )
    return f"{prompt_obj.prompt}\n\nBrand context: {brand_context}"


def get_prompts_by_use_case(use_case: str) -> list[ImagePrompt]:
    """Filter prompts by use case category."""
    return [p for p in MARKETING_PROMPTS if p.use_case == use_case]


def get_all_use_cases() -> list[str]:
    """Return unique list of use case categories."""
    return sorted(set(p.use_case for p in MARKETING_PROMPTS))
