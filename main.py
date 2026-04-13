"""
Immigration TikTok Pipeline - v3
----------------------------------
Architecture:
- n8n reads Data Table, passes seen_ids to POST /run
- Script scrapes curated TikTok accounts (no keyword search)
- Caption generated from actual video description via DeepSeek
- Returns JSON; n8n writes the row to Data Table

n8n HTTP Request node config:
  Method: POST
  URL: http://<your-server>:8000/run
  Body (JSON):
    {
      "seen_ids": ["123456789", "987654321", ...],
      "seen_urls": ["https://www.tiktok.com/...", ...]
    }
"""

import glob
import os
import re
import time

import cloudinary
import cloudinary.uploader
import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

APIFY_API_KEY = os.environ["APIFY_API_KEY"]
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]

cloudinary.config(
    cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
    api_key=os.environ["CLOUDINARY_API_KEY"],
    api_secret=os.environ["CLOUDINARY_API_SECRET"]
)

# ---------------------------------------------------------------------------
# Curated immigration TikTok accounts
# Add more usernames here any time — no code changes needed elsewhere
# ---------------------------------------------------------------------------
IMMIGRATION_ACCOUNTS = [
    # --- UK & Europe ---
    "theimmigrationlawyer",      # UK immigration lawyer, very active
    "immigration_solicitors",    # UK immigration solicitors London
    "gbnews",                    # GB News (UK immigration coverage)
    "itvnews",                   # ITV News
    "itvpolitics",               # ITV Politics
    "bbcnews",                   # BBC News
    "skynews",                   # Sky News
    "c4news",                    # Channel 4 News
    "youngeuropeans",            # European mobility & immigration news

    # --- Canada ---
    "gloriaofcanada",            # Canadian immigration creator, large following
    "immigrationnewscanada",     # Immigration News Canada (INC) — 244K followers
    "canadianimmlawyer",         # Paul, Canadian immigration lawyer — 172K followers
    "srimmigrationsolutions",    # Sri immigration solutions
    "moving2canadatok",          # Moving to Canada content
    "visaplaceimmigration",      # VisaPlace Canada
    "immigration.tips",          # Alex Canadian Lawyer
    "cadimmigrationlawyer",      # Jatin Shory — Canadian immigration lawyer
    "canadaimmigrationnewz",     # Canada immigration news
    "crossbridgeimmigration",    # Crossbridge Immigration Canada
    "gpsinghimmigration",        # GP Singh Immigration Canada
    "theimmigrationpro",         # Shelina, Pirani Immigration Canada
    "osmiumimmigration",         # Gurpreet Kaur RCIC Canada
    "mworldimmigration",         # M World Immigration Canada
    "todmaffin",                 # Canadian news/immigration commentary

    # --- USA ---
    "themigrantchannel",         # US migration & immigration news channel
    "mcbeanlaw",                 # McBean Law — US immigration breaking news
    "cbsnews",                   # CBS News
    "immigration.abcs",          # Immigration ABCs (Scott Legal NYC)

    # --- General / Multi-region ---
    "aatlantis.facilit",         # Immigration facilitation content
]


# ---------------------------------------------------------------------------
# Request model — n8n sends seen IDs/URLs so the script stays stateless
# ---------------------------------------------------------------------------
class RunRequest(BaseModel):
    seen_ids: Optional[List[str]] = []
    seen_urls: Optional[List[str]] = []
    recent_summaries: Optional[List[str]] = []  # Last 20 posted descriptions for topic dedup


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def normalise_url(url: str) -> str:
    return (url or "").split("?", 1)[0].rstrip("/")


def extract_video_id(url: str) -> str:
    m = re.search(r"/video/(\d+)", url or "")
    return m.group(1) if m else ""


def resolve_short_url(url: str) -> str:
    if not url or ("vm.tiktok" not in url and "vt.tiktok" not in url):
        return url
    try:
        r = requests.head(url, allow_redirects=True, timeout=10,
                          headers={"User-Agent": "Mozilla/5.0"})
        return normalise_url(r.url)
    except Exception:
        return url


def is_immigration_related(text: str) -> bool:
    """
    Fast first-pass keyword filter — removes obvious junk (sports, war, entertainment)
    before spending a DeepSeek API call. Intentionally broad so we don't miss
    anything legitimate. DeepSeek is the real final judge.
    """
    keywords = {
        # Core immigration terms
        "visa", "immigration", "immigrant", "immigrants", "immigrate",
        "migrants", "migrant", "migration",
        # Students
        "international student", "international students", "student visa",
        "study abroad", "study permit", "studying abroad",
        "student permit", "overseas student", "foreign student",
        # Nationality & citizenship
        "passport", "citizenship", "naturalisation", "naturalization",
        "permanent resident", "pr status", "green card",
        # Protection
        "asylum", "refugee", "refugees", "deportation", "deported", "removal order",
        # Work
        "work permit", "work visa", "skilled worker", "work authorization",
        "sponsorship", "sponsored", "sponsor", "employer sponsor",
        # Country-specific official terms
        "ircc", "uscis", "home office", "uk visa",
        "express entry", "points based", "global talent",
        "tier 2", "tier 4", "leave to remain", "right to remain",
        "biometric residence", "indefinite leave", "entry clearance",
        "eea", "brp", "settlement",
        # Legal
        "immigration lawyer", "immigration attorney", "immigration court",
        "travel ban", "visa ban", "immigration ban",
    }
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def is_immigration_related_ai(description: str) -> bool:
    """
    Second-pass AI gate — asks DeepSeek to confirm the video is genuinely
    about immigration, visas, or international students before we accept it.
    Returns True only if DeepSeek answers YES.
    Falls back to True on API errors to avoid blocking on outage.
    """
    if not description or not description.strip():
        return False
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        prompt = (
            f"You are a content classifier for an immigration and international education news account. "
            f"Read the following TikTok video description and answer with ONE word only: "
            f"YES if the video is specifically about immigration, visas, work permits, "
            f"student visas, international students, study abroad, citizenship, deportation, "
            f"refugees, asylum, or related immigration and international education topics. "
            f"Answer NO if it is about war, crime, domestic politics unrelated to immigration, "
            f"sports, entertainment, or any other unrelated topic.\n\n"
            f"Description: \"{description[:500]}\""
        )
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 5,
            "temperature": 0.0,
        }
        r = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=15,
        )
        r.raise_for_status()
        answer = r.json()["choices"][0]["message"]["content"].strip().upper()
        print(f"[AI GATE] DeepSeek says: {answer} | {description[:60]}")
        return answer.startswith("YES")
    except Exception as e:
        print(f"[WARN] AI gate failed ({e}), allowing through")
        return True


def is_duplicate_topic_ai(description: str, recent_summaries: List[str]) -> bool:
    """
    Third-pass duplicate topic check — asks DeepSeek if this video covers
    the same topic as any of the recently posted summaries.
    Returns True if it IS a duplicate (should be skipped).
    Falls back to False on API errors (allow through rather than block).
    """
    if not recent_summaries:
        return False  # No history to compare against
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        recent_list = "\n".join(
            f"- {s[:150]}" for s in recent_summaries[:20]
        )
        prompt = (
            f"You are a duplicate content checker for a social media account. "
            f"Determine if the NEW video covers the same core topic as any of the RECENTLY POSTED videos. "
            f"Answer with ONE word only: YES if it is a duplicate topic, NO if it is genuinely different.\n\n"
            f"NEW VIDEO DESCRIPTION:\n\"{description[:400]}\"\n\n"
            f"RECENTLY POSTED TOPICS:\n{recent_list}"
        )
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 5,
            "temperature": 0.0,
        }
        r = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=15,
        )
        r.raise_for_status()
        answer = r.json()["choices"][0]["message"]["content"].strip().upper()
        print(f"[DUPE CHECK] DeepSeek says: {answer} | {description[:60]}")
        return answer.startswith("YES")
    except Exception as e:
        print(f"[WARN] Duplicate check failed ({e}), allowing through")
        return False


# ---------------------------------------------------------------------------
# Apify — scrape latest videos from curated account list
# ---------------------------------------------------------------------------
def scrape_accounts(accounts: List[str], videos_per_account: int = 5) -> List[dict]:
    """
    Use Apify's TikTok Profile Scraper to fetch the latest N videos
    from each curated account.
    """
    run_url = f"https://api.apify.com/v2/acts/clockworks~tiktok-profile-scraper/runs?token={APIFY_API_KEY}"

    profiles = [f"https://www.tiktok.com/@{a}" for a in accounts]

    payload = {
        "profiles": profiles,
        "maxPostsPerProfile": videos_per_account,
    }

    print(f"[INFO] Scraping {len(accounts)} accounts, {videos_per_account} videos each")
    r = requests.post(run_url, json=payload, timeout=30)
    r.raise_for_status()
    run_id = r.json()["data"]["id"]

    # Poll until done (max ~6 minutes)
    run_info = None
    for _ in range(40):
        time.sleep(10)
        status_r = requests.get(
            f"https://api.apify.com/v2/actor-runs/{run_id}?token={APIFY_API_KEY}",
            timeout=15
        )
        status_r.raise_for_status()
        run_info = status_r.json()["data"]
        print(f"[INFO] Apify status: {run_info['status']}")
        if run_info["status"] == "SUCCEEDED":
            break
        if run_info["status"] in ["FAILED", "ABORTED", "TIMED-OUT"]:
            raise Exception(f"Apify profile scraper failed: {run_info['status']}")

    results = requests.get(
        f"https://api.apify.com/v2/datasets/{run_info['defaultDatasetId']}/items?token={APIFY_API_KEY}",
        timeout=15
    ).json()

    print(f"[INFO] Got {len(results)} raw videos from Apify")
    return results


# ---------------------------------------------------------------------------
# Pick the best unseen, relevant video
# ---------------------------------------------------------------------------
def pick_video(all_videos: List[dict], seen_ids: set, seen_urls: set, recent_summaries: List[str] = None) -> Optional[dict]:
    candidates = []
    recent_summaries = recent_summaries or []

    for video in all_videos:
        url = normalise_url(video.get("webVideoUrl", ""))
        url = resolve_short_url(url)
        vid = extract_video_id(url)

        if not url or not vid:
            continue

        # Skip already-posted videos (exact ID/URL match)
        if url in seen_urls or vid in seen_ids:
            continue

        # Layer 1: Fast keyword pre-filter (free, no API call)
        description = video.get("text", "") or ""
        if not is_immigration_related(description):
            print(f"[SKIP] Keyword filter rejected: {description[:80]}")
            continue

        # Layer 2: DeepSeek AI gate (strict YES/NO relevance classification)
        if not is_immigration_related_ai(description):
            print(f"[SKIP] AI gate rejected: {description[:80]}")
            continue

        # Layer 3: DeepSeek duplicate topic check
        if is_duplicate_topic_ai(description, recent_summaries):
            print(f"[SKIP] Duplicate topic rejected: {description[:80]}")
            continue

        candidates.append(video)

    if not candidates:
        return None

    # Sort by play count — highest first
    candidates.sort(key=lambda x: x.get("playCount") or 0, reverse=True)
    chosen = candidates[0]
    print(f"[INFO] Picked video: {chosen.get('webVideoUrl')} | plays: {chosen.get('playCount')}")
    return chosen


# ---------------------------------------------------------------------------
# DeepSeek — generate caption from the video's own description
# ---------------------------------------------------------------------------
def generate_caption(video_description: str, author: str) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    prompt = (
        f"You are a social media manager for an international study abroad consultancy (Cohby Consult). "
        f"Write an engaging Instagram Reels caption based on this TikTok video description:\n\n"
        f"\"{video_description}\"\n\n"
        f"Requirements:\n"
        f"- 2-4 sentences max, energetic and informative\n"
        f"- Include 4-6 relevant hashtags (immigration, study abroad, visa, destination country)\n"
        f"- Use 1-2 relevant emojis\n"
        f"- Do NOT include any source attribution, contact info, or URLs — those are added separately\n"
        f"- Output ONLY the caption text, nothing else"
    )

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300,
        "temperature": 0.7,
    }

    r = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    caption = r.json()["choices"][0]["message"]["content"].strip()
    return caption


# ---------------------------------------------------------------------------
# Download & upload
# ---------------------------------------------------------------------------
def download_video(video: dict) -> str:
    direct_url = (
        video.get("videoUrl")
        or video.get("downloadAddr")
        or (video.get("videoMeta") or {}).get("downloadAddr")
        or (video.get("videoMeta") or {}).get("originalDownloadAddr")
    )

    tmp_path = f"/tmp/immigration_{int(time.time())}.mp4"

    if direct_url:
        try:
            resp = requests.get(
                direct_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Referer": "https://www.tiktok.com/"
                },
                timeout=60,
                stream=True
            )
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 10000:
                return tmp_path
        except Exception as e:
            print(f"[WARN] Direct download failed: {e} — falling back to yt-dlp")

    return _ytdlp_download(normalise_url(video.get("webVideoUrl", "")))


def _ytdlp_download(tiktok_url: str) -> str:
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        raise Exception("yt-dlp not installed. Run: pip install yt-dlp")

    outtmpl = f"/tmp/immigration_{int(time.time())}.%(ext)s"
    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "noprogress": True,
        "noplaylist": True,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.tiktok.com/",
        },
        "extractor_args": {
            "tiktok": {"app_version": ["20.9.3"], "manifest_app_version": ["291"]}
        },
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(tiktok_url, download=True)
        file_path = ydl.prepare_filename(info)

    if not os.path.exists(file_path):
        base = os.path.splitext(file_path)[0]
        matches = glob.glob(base + ".*")
        mp4s = [m for m in matches if m.endswith(".mp4")]
        file_path = mp4s[0] if mp4s else (matches[0] if matches else file_path)

    if not os.path.exists(file_path):
        raise Exception("Video file not found after download")
    if os.path.getsize(file_path) < 10000:
        raise Exception("Downloaded file too small — not a valid video")

    return file_path


def upload_to_cloudinary(file_path: str) -> str:
    result = cloudinary.uploader.upload(
        file_path,
        resource_type="video",
        folder="immigration_news",
        use_filename=True,
        unique_filename=True
    )
    return result["secure_url"]


# ---------------------------------------------------------------------------
# Main endpoint — called by n8n
# ---------------------------------------------------------------------------
@app.post("/run")
def run_pipeline(body: RunRequest):
    """
    Called by n8n HTTP Request node (POST).
    Expects: { "seen_ids": [...], "seen_urls": [...] }
    Returns all fields needed for n8n Data Table insert.
    """
    tmp_path = None
    try:
        seen_ids = set(str(i).strip() for i in (body.seen_ids or []) if i)
        seen_urls = set(str(u).strip() for u in (body.seen_urls or []) if u)
        recent_summaries = [str(s).strip() for s in (body.recent_summaries or []) if s]
        print(f"[INFO] Received {len(seen_ids)} seen IDs, {len(seen_urls)} seen URLs, {len(recent_summaries)} recent summaries")

        # 1. Scrape all curated accounts (10 videos each for wider selection pool)
        all_videos = scrape_accounts(IMMIGRATION_ACCOUNTS, videos_per_account=10)
        if not all_videos:
            return JSONResponse({"status": "no_videos", "message": "Apify returned no videos"})

        # 2. Pick best unseen, relevant, non-duplicate video
        video = pick_video(all_videos, seen_ids, seen_urls, recent_summaries)
        if not video:
            return JSONResponse({
                "status": "no_fresh_video",
                "message": "All videos from curated accounts already posted or not immigration-related"
            })

        # 3. Extract metadata
        author = (video.get("authorMeta", {}).get("name", "") or "unknown").strip()
        tiktok_url = normalise_url(video.get("webVideoUrl", ""))
        video_id = extract_video_id(tiktok_url)
        play_count = int(video.get("playCount") or 0)
        video_description = (video.get("text", "") or "").strip()

        print(f"[INFO] Selected @{author} | ID: {video_id} | plays: {play_count}")
        print(f"[INFO] Description: {video_description[:120]}")

        if not tiktok_url or not video_id:
            return JSONResponse({"status": "no_video_url", "message": "No usable TikTok URL"})

        # 4. Generate caption from the actual video description (no more Perplexity mismatch)
        raw_caption = generate_caption(video_description, author)

        # 5. Append attribution footer
        full_caption = (
            f"{raw_caption}\n\n"
            f"TikTok source: {author}\n"
            f"Contact us: www.cohbyconsult.com"
        )

        # 6. Download + upload to Cloudinary
        tmp_path = download_video(video)
        cloudinary_url = upload_to_cloudinary(tmp_path)

        # 7. Return everything — n8n handles the Data Table write
        return JSONResponse({
            "status": "success",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "headline": video_description[:200],
            "summary": video_description,
            "caption": full_caption,
            "cloudinary_url": cloudinary_url,
            "tiktok_source": author,
            "tiktok_url": tiktok_url,
            "video_id": video_id,
            "play_count": play_count,
        })

    except Exception as e:
        import traceback
        print(f"[ERROR] {e}\n{traceback.format_exc()}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/run")
def run_pipeline_get():
    return JSONResponse({
        "message": "Use POST /run with body: {seen_ids: [], seen_urls: []}. "
                   "n8n should call this via HTTP Request node (POST)."
    })


@app.get("/health")
def health():
    return {"status": "ok", "accounts": len(IMMIGRATION_ACCOUNTS)}


@app.get("/accounts")
def list_accounts():
    """Check which accounts are in rotation."""
    return {"count": len(IMMIGRATION_ACCOUNTS), "accounts": IMMIGRATION_ACCOUNTS}
