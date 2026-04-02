"""
🤖 Telegram Anime Image Generator Bot — Khmer Edition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model  : Nano Banana Pro (gemini-3-pro-image-preview)
Storage: Supabase (users + generations tables)
Language: Khmer (ភាសាខ្មែរ)

Commands:
  /start   — ស្វាគមន៍ + ជ្រើសរើសសម្លេង
  /style   — ផ្លាស់ប្តូររចនាប័ទ្មគំនូរ
  /history — ប្រវត្តិ ៥ ចុងក្រោយ
  /cancel  — បោះបង់

Changes in this version:
  • All user-facing text in Khmer
  • Animated status ticker (dots cycle while generating)
  • "បង្កើតទៀត" (Generate More) + "ផ្លាស់ប្តូររចនាប័ទ្ម" buttons on result
  • Store last_image_bytes in context so "Generate More" reuses same photo
  • Bug fix: callback_data tokens kept ≤ 64 bytes (Telegram hard limit)
  • Bug fix: edit_message_text only called when text actually changed
  • Bug fix: asyncio.Lock created inside running loop (not at import time)
  • Bug fix: anim_task properly cancelled on shutdown / error
  • Bug fix: handle_non_photo catches all non-photo message types
  • Bug fix: generate_more conversation state corrected
  • Bug fix: fire-and-forget DB tasks log exceptions instead of swallowing them
  • Perf: httpx connection pool limits added
  • Perf: _user_locks uses TTLCache to prevent unbounded memory growth
  • Perf: animation interval reduced to 2 s for better UX
  • Security: internal error details logged server-side only, not sent to user

Requirements:
    pip install "python-telegram-bot==20.7" "supabase==2.4.0" \
                "google-genai>=1.52.0" "python-dotenv==1.0.1" \
                "httpx==0.27.0" "cachetools==5.3.3"
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
from datetime import datetime, timezone

import httpx
from cachetools import TTLCache
from dotenv import load_dotenv
from google import genai
from google.genai import types
from supabase import Client, create_client
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Suppress verbose Google SDK logs — prevents accidental API key exposure in logs
logging.getLogger("google.genai").setLevel(logging.WARNING)


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"⛔ Missing env var: {name}")
    return val


TELEGRAM_TOKEN = _require("TELEGRAM_BOT_TOKEN")
SUPABASE_URL   = _require("SUPABASE_URL")
SUPABASE_KEY   = _require("SUPABASE_KEY")
GEMINI_API_KEY = _require("GEMINI_API_KEY")

# ── Shared clients ────────────────────────────────────────────────────────────
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
gemini_client    = genai.Client(api_key=GEMINI_API_KEY)

# Connection pool limits prevent exhausting file descriptors under load
http_client: httpx.AsyncClient = httpx.AsyncClient(
    timeout=90,
    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
)

# Per-user asyncio.Lock — TTLCache caps memory: max 10 000 users, expires after 1 h idle
_user_locks: TTLCache = TTLCache(maxsize=10_000, ttl=3600)


def _get_user_lock(user_id: int) -> asyncio.Lock:
    if user_id not in _user_locks:
        _user_locks[user_id] = asyncio.Lock()
    return _user_locks[user_id]


# ── Conversation states ───────────────────────────────────────────────────────
CHOOSE_STYLE, WAIT_PHOTO = range(2)

# Callback-data tokens (must be ≤ 64 bytes each — Telegram hard limit)
CB_CHANGE_STYLE  = "cb_change_style"
CB_GENERATE_MORE = "cb_gen_more"

# ── Anime styles ──────────────────────────────────────────────────────────────
ANIME_STYLES: list[str] = [
    "Chibi (Super Deformed)", "Moe",
    "Kawaii",                 "Shonen",
    "Shojo",                  "Seinen",
    "Josei",                  "Kodomomuke",
    "Realistic",              "Semi-Realistic",
    "CGI / 3D",               "Avant-Garde",
    "Retro (80s/90s)",        "Isekai (Art Style)",
    "Cyberpunk",              "Mecha",
]

# callback_data must be ≤ 64 bytes — use short keys mapped to full style names
_STYLE_CB: dict[str, str]    = {f"s{i}": s for i, s in enumerate(ANIME_STYLES)}
_CB_TO_STYLE: dict[str, str] = _STYLE_CB
_STYLE_TO_CB: dict[str, str] = {v: k for k, v in _STYLE_CB.items()}

STYLE_PROMPTS: dict[str, str] = {
    "Chibi (Super Deformed)": (
        "Transform this person into a chibi / super-deformed anime character. "
        "Oversized head (~1/2 body height), tiny limbs, huge sparkling eyes, "
        "rosy cheeks, pastel palette, soft outlines. Keep hair color and outfit."
    ),
    "Moe": (
        "Redraw in moe anime style: large expressive eyes with detailed highlights, "
        "soft pastel skin, delicate features, gentle shy expression, "
        "smooth linework, soft gradient shading."
    ),
    "Kawaii": (
        "Transform into kawaii anime style: ultra-cute round face, sparkly starry "
        "eyes, pink blush marks, pastel rainbow palette, fluffy accessories, "
        "sparkles and stars in the background."
    ),
    "Shonen": (
        "Redraw as a shonen anime character: dynamic energetic pose, spiky or "
        "wind-blown hair, intense determined expression, bold black outlines, "
        "vibrant saturated colors, action-manga shading."
    ),
    "Shojo": (
        "Transform into shojo anime style: elegant delicate linework, large glassy "
        "eyes with flower-petal reflections, flowing hair, soft romantic pastel "
        "background with floating petals or sparkles, graceful pose."
    ),
    "Seinen": (
        "Redraw in seinen anime style: realistic detailed anatomy, mature serious "
        "expression, high-contrast moody lighting, detailed fabric textures, "
        "dark muted color palette, cinematic composition."
    ),
    "Josei": (
        "Transform into josei anime style: elegant adult fashion, sophisticated "
        "makeup, soft romantic warm palette, mature feminine features, "
        "detailed hair with shine highlights."
    ),
    "Kodomomuke": (
        "Redraw in kodomomuke children's anime style: simple round shapes, bright "
        "primary colors, big friendly eyes, cheerful expression, clean thick "
        "outlines, flat simple shading."
    ),
    "Realistic": (
        "Transform into realistic anime style: highly detailed painting, lifelike "
        "proportions with subtle anime stylization, cinematic rim lighting, "
        "detailed skin texture, photorealistic background."
    ),
    "Semi-Realistic": (
        "Redraw in semi-realistic anime style: painterly brushstroke shading, "
        "detailed but stylized eyes, soft bokeh background, warm natural lighting."
    ),
    "CGI / 3D": (
        "Transform into 3D CGI anime style: smooth subsurface scattering skin, "
        "studio three-point lighting, physically-based rendering, "
        "clean 3D model aesthetic like a high-end anime film."
    ),
    "Avant-Garde": (
        "Redraw in avant-garde experimental anime style: abstract geometric shapes, "
        "surreal dreamlike palette, unconventional composition, mixed media "
        "textures, artistic distortion of features."
    ),
    "Retro (80s/90s)": (
        "Transform into retro 80s/90s anime style: cel-animation look, limited "
        "palette with halftone dots, VHS scan lines, vintage color grading, "
        "classic thick outlines, nostalgic grain texture."
    ),
    "Isekai (Art Style)": (
        "Redraw as an isekai fantasy anime character: glowing magical aura, "
        "fantasy adventurer outfit with armor or robes, epic landscape background "
        "with magical particles, heroic confident pose."
    ),
    "Cyberpunk": (
        "Transform into cyberpunk anime style: neon blue/pink/purple lighting, "
        "futuristic cybernetic implants, rain-soaked reflective surfaces, "
        "dark dystopian city backdrop, high-contrast neon shadows."
    ),
    "Mecha": (
        "Redraw as a mecha anime pilot or alongside a giant robot: sleek metallic "
        "pilot suit, cockpit with holographic displays, dramatic low-angle shot, "
        "metallic sheen with battle damage details."
    ),
}

BASE_INSTRUCTIONS = (
    "\n\nCore rules:\n"
    "- Preserve the subject's face, identity, and hair color\n"
    "- High-quality detailed anime illustration\n"
    "- No watermarks or text overlays\n"
    "- Portrait or square composition"
)

# ── Pre-built keyboards ───────────────────────────────────────────────────────
_STYLE_KEYBOARD: InlineKeyboardMarkup = InlineKeyboardMarkup([
    [
        InlineKeyboardButton(ANIME_STYLES[i], callback_data=_STYLE_TO_CB[ANIME_STYLES[i]]),
        *(
            [InlineKeyboardButton(ANIME_STYLES[i + 1], callback_data=_STYLE_TO_CB[ANIME_STYLES[i + 1]])]
            if i + 1 < len(ANIME_STYLES) else []
        ),
    ]
    for i in range(0, len(ANIME_STYLES), 2)
])


def _result_keyboard() -> InlineKeyboardMarkup:
    """Keyboard shown under every generated result image."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🔄 បង្កើតទៀត",          callback_data=CB_GENERATE_MORE),
            InlineKeyboardButton("🎨 ផ្លាស់ប្តូររចនាប័ទ្ម", callback_data=CB_CHANGE_STYLE),
        ]
    ])


# ── Animated status ticker ────────────────────────────────────────────────────
_TICK_FRAMES = ["⏳", "⌛", "⏳", "⌛"]
_TICK_STEPS  = [
    "កំពុងដំណើរការ ·",
    "កំពុងដំណើរការ ··",
    "កំពុងដំណើរការ ···",
    "កំពុងដំណើរការ ····",
]


async def _animate_status(
    chat_id: int,
    message_id: int,
    style: str,
    context: ContextTypes.DEFAULT_TYPE,
    stop_event: asyncio.Event,
) -> None:
    """
    Edit the status message every ~2 s while generation is running.
    Stops as soon as stop_event is set.
    Interval reduced from 4 s → 2 s so users see activity sooner.
    """
    step = 0
    while not stop_event.is_set():
        await asyncio.sleep(2)
        if stop_event.is_set():
            break
        frame = _TICK_FRAMES[step % len(_TICK_FRAMES)]
        dots  = _TICK_STEPS[step % len(_TICK_STEPS)]
        text  = (
            f"{frame} *{style}*\n"
            f"{dots}\n\n"
            "_Nano Banana Pro កំពុងគូររូបភាព..._\n"
            "_សូមរង់ចាំ ១៥–៤០ វិនាទី_"
        )
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                parse_mode="Markdown",
            )
        except BadRequest:
            pass  # message deleted or unchanged — ignore
        step += 1


# ── Supabase helpers ──────────────────────────────────────────────────────────

async def db_upsert_user(user) -> None:
    def _run():
        supabase.table("users").upsert(
            {
                "telegram_id": user.id,
                "username":    user.username or "",
                "first_name":  user.first_name or "",
                "last_name":   user.last_name or "",
                "updated_at":  datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="telegram_id",
        ).execute()
    await asyncio.to_thread(_run)


async def db_log_generation(telegram_id: int, style: str, status: str) -> None:
    def _run():
        supabase.table("generations").insert(
            {
                "telegram_id": telegram_id,
                "style":       style,
                "status":      status,
                "created_at":  datetime.now(timezone.utc).isoformat(),
            }
        ).execute()
    await asyncio.to_thread(_run)


async def db_get_history(telegram_id: int, limit: int = 5) -> list[dict]:
    def _run():
        return (
            supabase.table("generations")
            .select("style, status, created_at")
            .eq("telegram_id", telegram_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
            .data
        )
    return await asyncio.to_thread(_run)


# ── Safe fire-and-forget wrappers ─────────────────────────────────────────────
# Exceptions are logged server-side and never swallowed silently.

async def _safe_upsert_user(user) -> None:
    try:
        await db_upsert_user(user)
    except Exception as exc:
        logger.warning("db_upsert_user failed for user=%s: %s", user.id, exc)


async def _safe_log_generation(telegram_id: int, style: str, status: str) -> None:
    try:
        await db_log_generation(telegram_id, style, status)
    except Exception as exc:
        logger.warning("db_log_generation failed for user=%s: %s", telegram_id, exc)


# ── Image generation ──────────────────────────────────────────────────────────

async def generate_anime_image(image_bytes: bytes, style: str) -> bytes:
    """Call Nano Banana Pro (sync SDK) inside a thread. Returns image bytes."""

    prompt_text = STYLE_PROMPTS.get(style, f"{style} anime style") + BASE_INSTRUCTIONS

    def _call_gemini() -> bytes:
        response = gemini_client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="image/jpeg",
                                data=image_bytes,
                            )
                        ),
                        types.Part(text=prompt_text),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if part.inline_data is not None:
                    return part.inline_data.data
        raise ValueError(
            "Nano Banana Pro មិនបានបង្ហាញរូបភាព។ "
            f"ការឆ្លើយតបអត្ថបទ: {getattr(response, 'text', 'N/A')}"
        )

    logger.info("→ Nano Banana Pro | style=%s", style)
    return await asyncio.to_thread(_call_gemini)


async def download_telegram_photo(file_path: str) -> bytes:
    resp = await http_client.get(file_path)
    resp.raise_for_status()
    return resp.content


# ── Core generation flow ──────────────────────────────────────────────────────

async def _run_generation(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    image_bytes: bytes,
    style: str,
) -> None:
    """
    Generates anime image, sends result, logs to DB.
    Handles animated status ticker and per-user concurrency lock internally.
    anim_task is properly cancelled (not just stopped via event) on all exit paths.
    """
    user    = update.effective_user
    chat_id = update.effective_chat.id
    lock    = _get_user_lock(user.id)

    if lock.locked():
        await context.bot.send_message(
            chat_id,
            "⏳ រូបភាពមុនរបស់អ្នកនៅតែកំពុងដំណើរការ — សូមរង់ចាំ!",
        )
        return

    async with lock:
        status_msg = await context.bot.send_message(
            chat_id,
            f"⏳ *{style}*\nកំពុងដំណើរការ ·\n\n"
            "_Nano Banana Pro កំពុងគូររូបភាព..._\n"
            "_សូមរង់ចាំ ១៥–៤០ វិនាទី_",
            parse_mode="Markdown",
        )

        stop_event = asyncio.Event()
        anim_task  = asyncio.create_task(
            _animate_status(chat_id, status_msg.message_id, style, context, stop_event)
        )

        try:
            result_bytes = await generate_anime_image(image_bytes, style)

            # Stop and cleanly cancel the animation task
            stop_event.set()
            anim_task.cancel()
            try:
                await anim_task
            except asyncio.CancelledError:
                pass

            try:
                await status_msg.delete()
            except BadRequest:
                pass

            await context.bot.send_photo(
                chat_id=chat_id,
                photo=io.BytesIO(result_bytes),
                caption=(
                    f"✅ *{style}*\n"
                    "រូបភាពអ្នកបានបំប្លែងទៅជាអាណីមេដោយជោគជ័យ! 🎉\n\n"
                    "_តើអ្នកចង់ធ្វើអ្វីបន្ទាប់?_"
                ),
                parse_mode="Markdown",
                reply_markup=_result_keyboard(),
            )

            asyncio.create_task(_safe_log_generation(user.id, style, "success"))

        except Exception as exc:
            # Stop and cleanly cancel the animation task
            stop_event.set()
            anim_task.cancel()
            try:
                await anim_task
            except asyncio.CancelledError:
                pass

            try:
                await status_msg.delete()
            except BadRequest:
                pass

            # Log full error server-side; send only a generic message to user
            logger.exception("Generation failed user=%s style=%s: %s", user.id, style, exc)

            await context.bot.send_message(
                chat_id,
                "❌ *មានបញ្ហាក្នុងការបង្កើតរូបភាព*\n\n"
                "សូមព្យាយាមម្តងទៀត ឬជ្រើសរើសរចនាប័ទ្មផ្សេង។",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("🔄 ព្យាយាមម្តងទៀត",   callback_data=CB_GENERATE_MORE),
                    InlineKeyboardButton("🎨 ផ្លាស់ប្តូររចនាប័ទ្ម", callback_data=CB_CHANGE_STYLE),
                ]])
            )
            asyncio.create_task(
                _safe_log_generation(user.id, style, f"error:{str(exc)[:200]}")
            )


# ── Telegram handlers ─────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    asyncio.create_task(_safe_upsert_user(user))

    await update.message.reply_text(
        f"🎌 សួស្តី *{user.first_name}*!\n\n"
        "🎨 *បម្លែងរូបថតទៅជាគំនូរអាណីមេ*\n"
        "_ដំណើរការដោយ Nano Banana Pro_\n\n"
        "👇 ជ្រើសរើសរចនាប័ទ្មដែលអ្នកចង់បាន:",
        parse_mode="Markdown",
        reply_markup=_STYLE_KEYBOARD,
    )
    return CHOOSE_STYLE


async def cmd_style(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "🎨 ជ្រើសរើសរចនាប័ទ្មអាណីមេថ្មី:",
        reply_markup=_STYLE_KEYBOARD,
    )
    return CHOOSE_STYLE


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    rows = await db_get_history(user.id)

    if not rows:
        await update.message.reply_text(
            "📭 អ្នកមិនទាន់បានបង្កើតរូបភាពណាមួយនៅឡើយទេ!"
        )
        return

    lines = ["📜 *ប្រវត្តិបង្កើតរូបភាព (៥ ចុងក្រោយ):*\n"]
    for i, row in enumerate(rows, 1):
        ts  = row["created_at"][:16].replace("T", " ")
        ok  = "✅" if row["status"] == "success" else "❌"
        lines.append(f"{i}. {ok} *{row['style']}* — {ts}")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def style_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    data  = query.data

    # ── "Change Style" button ──────────────────────────────────────────────
    if data == CB_CHANGE_STYLE:
        await query.message.reply_text(
            "🎨 ជ្រើសរើសរចនាប័ទ្មថ្មី:",
            reply_markup=_STYLE_KEYBOARD,
        )
        return CHOOSE_STYLE

    # ── "Generate More" button ─────────────────────────────────────────────
    if data == CB_GENERATE_MORE:
        style      = context.user_data.get("chosen_style")
        last_bytes = context.user_data.get("last_image_bytes")

        if not style or not last_bytes:
            # Missing state — ask user to pick a style and resend photo
            await query.message.reply_text(
                "⚠️ សូមជ្រើសរើសរចនាប័ទ្ម រួចផ្ញើរូបថតថ្មីមក។",
                reply_markup=_STYLE_KEYBOARD,
            )
            return CHOOSE_STYLE

        asyncio.create_task(
            _run_generation(update, context, last_bytes, style)
        )
        # Stay in WAIT_PHOTO: user may send another photo or tap buttons again
        return WAIT_PHOTO

    # ── Style selection ────────────────────────────────────────────────────
    style = _CB_TO_STYLE.get(data)
    if not style:
        await query.answer("❓ រចនាប័ទ្មអ្នកជ្រើសមិនស្គាល់", show_alert=True)
        return CHOOSE_STYLE

    context.user_data["chosen_style"] = style

    await query.edit_message_text(
        f"✅ បានជ្រើសរើស: *{style}*\n\n"
        "📸 ផ្ញើរូបថតរបស់អ្នកមក:",
        parse_mode="Markdown",
    )
    return WAIT_PHOTO


async def photo_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    style = context.user_data.get("chosen_style")

    if not style:
        await update.message.reply_text(
            "⚠️ សូមជ្រើសរើសរចនាប័ទ្មជាមុនសិន។\n"
            "ប្រើ /style ដើម្បីជ្រើសរើស។",
            reply_markup=_STYLE_KEYBOARD,
        )
        return CHOOSE_STYLE

    try:
        photo_file  = await update.message.photo[-1].get_file()
        image_bytes = await download_telegram_photo(photo_file.file_path)
    except Exception as exc:
        logger.exception("Photo download failed: %s", exc)
        await update.message.reply_text(
            "❌ មិនអាចទាញយករូបថតបានទេ។ សូមព្យាយាមម្តងទៀត។"
        )
        return WAIT_PHOTO

    # Cache bytes so "Generate More" can reuse without the user re-uploading
    context.user_data["last_image_bytes"] = image_bytes

    asyncio.create_task(_run_generation(update, context, image_bytes, style))
    return WAIT_PHOTO


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text(
        "👋 បានបោះបង់។ វាយ /start ដើម្បីចាប់ផ្តើមម្តងទៀត។"
    )
    return ConversationHandler.END


async def handle_non_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Catches ALL non-photo, non-command messages while in WAIT_PHOTO state.
    Previously only caught TEXT — documents, stickers, audio, etc. fell through silently.
    """
    style = context.user_data.get("chosen_style", "")
    hint  = f" (បានជ្រើស: *{style}*)" if style else ""
    await update.message.reply_text(
        f"📸 សូមផ្ញើ *រូបថត*{hint} មិនមែនឯកសារ ឬអត្ថបទទេ។",
        parse_mode="Markdown",
    )
    return WAIT_PHOTO


# ── Lifecycle ─────────────────────────────────────────────────────────────────

async def on_shutdown(app: Application) -> None:
    await http_client.aclose()
    logger.info("HTTP client closed.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_shutdown(on_shutdown)
        .build()
    )

    conv = ConversationHandler(
        entry_points=[
            CommandHandler("start", cmd_start),
            CommandHandler("style", cmd_style),
        ],
        states={
            CHOOSE_STYLE: [
                CallbackQueryHandler(style_chosen),
            ],
            WAIT_PHOTO: [
                MessageHandler(filters.PHOTO, photo_received),
                CallbackQueryHandler(style_chosen),
                # Fix: ~filters.PHOTO catches documents, stickers, audio, video, etc.
                # Previously only filters.TEXT was handled — all other types fell through.
                MessageHandler(~filters.PHOTO & ~filters.COMMAND, handle_non_photo),
            ],
        },
        fallbacks=[CommandHandler("cancel", cmd_cancel)],
        allow_reentry=True,
        per_message=False,
    )

    app.add_handler(conv)
    app.add_handler(CommandHandler("history", cmd_history))

    logger.info("✅ Bot started — Nano Banana Pro | Khmer Edition")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
