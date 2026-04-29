"""
General News TikTok Pipeline - v1
----------------------------------
Architecture:
- n8n reads Data Table, passes seen_ids to POST /run
- Script scrapes curated trusted news TikTok accounts (no keyword search)
- Filters for important general news only via DeepSeek AI gate
- Minimum 1,000 views/likes threshold
- Caption generated from video description via DeepSeek (clean, brand-neutral)
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

DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]

cloudinary.config(
    cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
    api_key=os.environ["CLOUDINARY_API_KEY"],
    api_secret=os.environ["CLOUDINARY_API_SECRET"]
)

# ---------------------------------------------------------------------------
# Curated trusted news TikTok accounts — major media only, no personal pages
# ---------------------------------------------------------------------------
NEWS_ACCOUNTS = [
    # --- UK ---
    "bbcnews",           # BBC News
    "bbcworldservice",   # BBC World Service
    "skynews",           # Sky News
    "itvnews",           # ITV News
    "c4news",            # Channel 4 News
    "theguardian",       # The Guardian
    "thetimes",          # The Times
    "standardnews",      # Evening Standard
    "telegraphnews",     # The Telegraph

    # --- USA ---
    "cnn",               # CNN
    "cbsnews",           # CBS News
    "abcnews",           # ABC News
    "nbcnews",           # NBC News
    "foxnews",           # Fox News
    "reuters",           # Reuters
    "apnews",            # Associated Press
    "nprnews",           # NPR News
    "washingtonpost",    # Washington Post
    "nytimes",           # New York Times
    "thehill",           # The Hill
    "politico",          # Politico
    "axios",             # Axios

    # --- Canada ---
    "cbcnews",           # CBC News
    "globalnews",        # Global News
    "ctvnews",           # CTV News
    "torontostar",       # Toronto Star

    # --- Australia ---
    "abcaustralia",      # ABC Australia
    "skynewsaustralia",  # Sky News Australia
    "9newsaus",          # 9News Australia
    "7newsaustralia",    # 7News Australia
    "theaustralian",     # The Australian

    # --- New Zealand ---
    "rnz_news",          # RNZ (Radio New Zealand)
    "1newsnz",           # 1News NZ
    "stuffnz",           # Stuff NZ
    "nzherald",          # NZ Herald

    # --- International ---
    "aljazeera",         # Al Jazeera English
    "dwnews",            # Deutsche Welle
    "france24english",   # France 24 English
    "euronews",          # Euronews
    "trtworld",          # TRT World
]

MIN_ENGAGEMENT = 1000  # Minimum views OR likes to qualify


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
class RunRequest(BaseModel):
    seen_ids: Optional[List[str]] = []
    seen_urls: Optional[List[str]] = []
    recent_summaries: Optional[List[str]] = []


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


def meets_engagement_threshold(video: dict) -> bool:
    play_count = int(video.get("playCount") or 0)
    like_count = int(video.get("diggCount") or 0)
    return play_count >= MIN_ENGAGEMENT or like_count >= MIN_ENGAGEMENT


def is_important_news_ai(description: str) -> bool:
    """
    DeepSeek AI gate — confirms the video is important general news.
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
            "You are a news importance classifier. "
            "Read the following TikTok video description from a trusted news media account. "
            "Answer with ONE word only: YES if this is important general news "
            "(politics, economy, conflict, disaster, major world events, breaking news, science breakthroughs, health crises). "
            "Answer NO if it is sport, entertainment, celebrity gossip, lifestyle, "
            "cooking, fashion, travel tips, or any non-news content.\n\n"
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
    Duplicate topic check — skips if same core story already posted recently.
    Falls back to False on API errors (allow through rather than block).
    """
    if not recent_summaries:
        return False
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }
        recent_list = "\n".join(f"- {s[:150]}" for s in recent_summaries[:20])
        prompt = (
            "You are a duplicate content checker for a news social media account. "
            "Determine if the NEW video covers the same core story as any RECENTLY POSTED video. "
            "Answer with ONE word only: YES if it is the same story, NO if it is genuinely different.\n\n"
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


def generate_caption(video_description: str) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    prompt = (
        "You are a social media manager for a general news TikTok account. "
        "Rewrite the following news video description as a clean, punchy TikTok caption.\n\n"
        "Requirements:\n"
        "- 2-3 sentences max, informative and engaging\n"
        "- Include 3-5 relevant news hashtags (topic, country, breaking news etc.)\n"
        "- Use 1-2 relevant emojis\n"
        "- Do NOT include any URLs, contact info, or source attribution\n"
        "- Output ONLY the caption text, nothing else\n\n"
        f"Description: \"{video_description[:500]}\""
    )
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.7,
    }
    r = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def scrape_accounts(accounts: List[str], videos_per_account: int = 10) -> List[dict]:
    from yt_dlp import YoutubeDL

    all_videos = []

    ydl_opts = {
        "quiet": True,
        "noprogress": True,
        "skip_download": True,
        "playlistend": videos_per_account,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        },
    }

    for account in accounts:
        url = f"https://www.tiktok.com/@{account}"
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                entries = info.get("entries", []) if info else []
                for entry in entries:
                    if not entry:
                        continue
                    all_videos.append({
                        "webVideoUrl": entry.get("webpage_url", ""),
                        "text": entry.get("description", ""),
                        "playCount": entry.get("view_count", 0),
                        "diggCount": entry.get("like_count", 0),
                        "videoUrl": entry.get("url", ""),
                        "authorMeta": {"name": account},
                    })
                print(f"[INFO] @{account}: {len(entries)} videos fetched")
        except Exception as e:
            print(f"[WARN] Failed to scrape @{account}: {e}")
            continue

    print(f"[INFO] Total videos fetched: {len(all_videos)}")
    return all_videos


# ---------------------------------------------------------------------------
# Pick the best unseen, important, non-duplicate video
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

        # Skip already-posted
        if url in seen_urls or vid in seen_ids:
            continue

        # Layer 1: Engagement threshold (>=1,000 views or likes)
        if not meets_engagement_threshold(video):
            print(f"[SKIP] Below engagement threshold: plays={video.get('playCount')} likes={video.get('diggCount')}")
            continue

        description = (video.get("text", "") or "").strip()

        # Layer 2: DeepSeek AI gate — important news only
        if not is_important_news_ai(description):
            print(f"[SKIP] AI gate rejected (not important news): {description[:80]}")
            continue

        # Layer 3: Duplicate topic check
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
# Download & upload
# ---------------------------------------------------------------------------
def download_video(video: dict) -> str:
    direct_url = (
        video.get("videoUrl")
        or video.get("downloadAddr")
        or (video.get("videoMeta") or {}).get("downloadAddr")
        or (video.get("videoMeta") or {}).get("originalDownloadAddr")
    )

    tmp_path = f"/tmp/general_news_{int(time.time())}.mp4"

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

    outtmpl = f"/tmp/general_news_{int(time.time())}.%(ext)s"
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
        folder="general_news",
        use_filename=True,
        unique_filename=True
    )
    return result["secure_url"]


# ---------------------------------------------------------------------------
# Main endpoint — called by n8n
# ---------------------------------------------------------------------------
@app.post("/run")
def run_pipeline(body: RunRequest):
    tmp_path = None
    try:
        seen_ids = set(str(i).strip() for i in (body.seen_ids or []) if i)
        seen_urls = set(str(u).strip() for u in (body.seen_urls or []) if u)
        recent_summaries = [str(s).strip() for s in (body.recent_summaries or []) if s]
        print(f"[INFO] Received {len(seen_ids)} seen IDs, {len(seen_urls)} seen URLs, {len(recent_summaries)} recent summaries")

        # 1. Scrape all curated accounts
        all_videos = scrape_accounts(NEWS_ACCOUNTS, videos_per_account=10)
        if not all_videos:
            return JSONResponse({"status": "no_videos", "message": "yt-dlp returned no videos"})

        # 2. Pick best unseen, important, non-duplicate video
        video = pick_video(all_videos, seen_ids, seen_urls, recent_summaries)
        if not video:
            return JSONResponse({
                "status": "no_fresh_video",
                "message": "No fresh important news videos found across all accounts"
            })

        # 3. Extract metadata
        author = (video.get("authorMeta", {}).get("name", "") or "unknown").strip()
        tiktok_url = normalise_url(video.get("webVideoUrl", ""))
        video_id = extract_video_id(tiktok_url)
        play_count = int(video.get("playCount") or 0)
        like_count = int(video.get("diggCount") or 0)
        video_description = (video.get("text", "") or "").strip()

        print(f"[INFO] Selected @{author} | ID: {video_id} | plays: {play_count} | likes: {like_count}")

        if not tiktok_url or not video_id:
            return JSONResponse({"status": "no_video_url", "message": "No usable TikTok URL"})

        # 4. Generate caption from video description
        caption = generate_caption(video_description)

        # 5. Download + upload to Cloudinary
        tmp_path = download_video(video)
        cloudinary_url = upload_to_cloudinary(tmp_path)

        # 6. Return everything — n8n handles the Data Table write
        return JSONResponse({
            "status": "success",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "headline": video_description[:200],
            "summary": video_description,
            "caption": caption,
            "cloudinary_url": cloudinary_url,
            "tiktok_source": author,
            "tiktok_url": tiktok_url,
            "video_id": video_id,
            "play_count": play_count,
            "like_count": like_count,
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
    return {"status": "ok", "accounts": len(NEWS_ACCOUNTS)}


@app.get("/accounts")
def list_accounts():
    return {"count": len(NEWS_ACCOUNTS), "accounts": NEWS_ACCOUNTS}
