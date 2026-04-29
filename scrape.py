"""
General News TikTok Pipeline - GitHub Actions Edition
------------------------------------------------------
- Runs on a cron schedule via GitHub Actions
- Scrapes trusted news TikTok accounts via yt-dlp
- Filters for important general news via DeepSeek AI gate
- Minimum 1,000 views/likes threshold
- Generates caption via DeepSeek
- Uploads to Cloudinary
- POSTs result to n8n webhook
"""

import os
import re
import glob
import time
import requests
import cloudinary
import cloudinary.uploader
from typing import List, Optional

DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
N8N_WEBHOOK_URL = os.environ["N8N_WEBHOOK_URL"]

cloudinary.config(
    cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
    api_key=os.environ["CLOUDINARY_API_KEY"],
    api_secret=os.environ["CLOUDINARY_API_SECRET"]
)

NEWS_ACCOUNTS = [
    # --- UK ---
    "bbcnews",
    "bbcworldservice",
    "skynews",
    "itvnews",
    "c4news",
    "theguardian",
    "thetimes",
    "standardnews",
    "telegraphnews",

    # --- USA ---
    "cnn",
    "cbsnews",
    "abcnews",
    "nbcnews",
    "foxnews",
    "reuters",
    "apnews",
    "nprnews",
    "washingtonpost",
    "nytimes",
    "thehill",
    "politico",
    "axios",

    # --- Canada ---
    "cbcnews",
    "globalnews",
    "ctvnews",
    "torontostar",

    # --- Australia ---
    "abcaustralia",
    "9newsaus",
    "7newsaustralia",
    "theaustralian",

    # --- New Zealand ---
    "rnz_news",
    "1newsnz",
    "stuffnz",
    "nzherald",

    # --- International ---
    "aljazeera",
    "dwnews",
    "france24english",
    "euronews",
    "trtworld",
]

MIN_ENGAGEMENT = 1000


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


# ---------------------------------------------------------------------------
# DeepSeek helpers
# ---------------------------------------------------------------------------
def is_important_news_ai(description: str) -> bool:
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
            "Answer with ONE word only: YES if this is BREAKING NEWS only "
            "(developing stories, urgent events, major disasters, sudden political crises, terror attacks, major conflict escalations). "
            "Answer NO if it is anything else — including regular news, analysis, opinion, features, sport, entertainment or lifestyle.\n\n "
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
            headers=headers, json=payload, timeout=15
        )
        r.raise_for_status()
        answer = r.json()["choices"][0]["message"]["content"].strip().upper()
        print(f"[AI GATE] {answer} | {description[:60]}")
        return answer.startswith("YES")
    except Exception as e:
        print(f"[WARN] AI gate failed ({e}), allowing through")
        return True


def is_duplicate_topic_ai(description: str, recent_summaries: List[str]) -> bool:
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
            headers=headers, json=payload, timeout=15
        )
        r.raise_for_status()
        answer = r.json()["choices"][0]["message"]["content"].strip().upper()
        print(f"[DUPE CHECK] {answer} | {description[:60]}")
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
        headers=headers, json=payload, timeout=30
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# yt-dlp scraper
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
# Pick best video
# ---------------------------------------------------------------------------
def pick_video(all_videos: List[dict], seen_ids: set, seen_urls: set, recent_summaries: List[str]) -> Optional[dict]:
    candidates = []

    for video in all_videos:
        url = normalise_url(video.get("webVideoUrl", ""))
        url = resolve_short_url(url)
        vid = extract_video_id(url)

        if not url or not vid:
            continue
        if url in seen_urls or vid in seen_ids:
            continue
        if not meets_engagement_threshold(video):
            print(f"[SKIP] Below threshold: plays={video.get('playCount')} likes={video.get('diggCount')}")
            continue

        description = (video.get("text", "") or "").strip()

        if not is_important_news_ai(description):
            print(f"[SKIP] Not important news: {description[:80]}")
            continue
        if is_duplicate_topic_ai(description, recent_summaries):
            print(f"[SKIP] Duplicate topic: {description[:80]}")
            continue

        candidates.append(video)

    if not candidates:
        return None

    candidates.sort(key=lambda x: x.get("playCount") or 0, reverse=True)
    chosen = candidates[0]
    print(f"[INFO] Picked: {chosen.get('webVideoUrl')} | plays: {chosen.get('playCount')}")
    return chosen


# ---------------------------------------------------------------------------
# Download & upload
# ---------------------------------------------------------------------------
def download_video(video: dict) -> str:
    direct_url = (
        video.get("videoUrl")
        or video.get("downloadAddr")
        or (video.get("videoMeta") or {}).get("downloadAddr")
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
    from yt_dlp import YoutubeDL

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
        raise Exception("Downloaded file too small")

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
# Main
# ---------------------------------------------------------------------------
def main():
    # 1. Fetch seen IDs/URLs and recent summaries from n8n webhook
    # n8n sends these via a separate GET webhook you set up (see below)
    seen_ids = set()
    seen_urls = set()
    recent_summaries = []

    try:
        resp = requests.get(
            os.environ["N8N_DATA_FETCH_URL"],
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        seen_ids = set(str(i) for i in data.get("seen_ids", []) if i)
        seen_urls = set(str(u) for u in data.get("seen_urls", []) if u)
        recent_summaries = [str(s) for s in data.get("recent_summaries", []) if s]
        print(f"[INFO] Fetched {len(seen_ids)} seen IDs, {len(seen_urls)} seen URLs")
    except Exception as e:
        print(f"[WARN] Could not fetch seen data ({e}), proceeding with empty sets")

    # 2. Scrape accounts
    all_videos = scrape_accounts(NEWS_ACCOUNTS, videos_per_account=10)
    if not all_videos:
        print("[INFO] No videos fetched — exiting")
        return

    # 3. Pick best video
    video = pick_video(all_videos, seen_ids, seen_urls, recent_summaries)
    if not video:
        print("[INFO] No fresh important news video found — exiting")
        return

    # 4. Extract metadata
    author = (video.get("authorMeta", {}).get("name", "") or "unknown").strip()
    tiktok_url = normalise_url(video.get("webVideoUrl", ""))
    video_id = extract_video_id(tiktok_url)
    play_count = int(video.get("playCount") or 0)
    like_count = int(video.get("diggCount") or 0)
    video_description = (video.get("text", "") or "").strip()

    print(f"[INFO] Selected @{author} | ID: {video_id} | plays: {play_count}")

    # 5. Generate caption
    caption = generate_caption(video_description)

    # 6. Download + upload
    tmp_path = download_video(video)
    try:
        cloudinary_url = upload_to_cloudinary(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 7. POST result to n8n webhook
    payload = {
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
    }

    r = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=30)
    r.raise_for_status()
    print(f"[INFO] Posted to n8n webhook — status: {r.status_code}")


if __name__ == "__main__":
    main()
