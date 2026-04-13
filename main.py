import glob
import os
import re
import time

import cloudinary
import cloudinary.uploader
import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

APIFY_API_KEY = os.environ["APIFY_API_KEY"]
PERPLEXITY_API_KEY = os.environ["PERPLEXITY_API_KEY"]
APPS_SCRIPT_URL = os.environ["APPS_SCRIPT_URL"]

cloudinary.config(
    cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
    api_key=os.environ["CLOUDINARY_API_KEY"],
    api_secret=os.environ["CLOUDINARY_API_SECRET"]
)

TRUSTED_ACCOUNTS = [
    "bbc", "bbcnews", "cnn", "skynews", "itvnews",
    "channel4news", "guardiannews", "independent",
    "reuters", "apnews", "gbnews", "dwnews", "timesradio"
]


def normalise_url(url):
    return (url or "").split("?", 1)[0].rstrip("/")


def extract_video_id(url):
    # FIX 1: was r"/video/(\\d+)" — double backslash in raw string never matched digits
    m = re.search(r"/video/(\d+)", url or "")
    return m.group(1) if m else ""


def resolve_short_url(url):
    """Follow redirects on shortened TikTok URLs to get the canonical URL."""
    if not url or ("vm.tiktok" not in url and "vt.tiktok" not in url):
        return url
    try:
        r = requests.head(url, allow_redirects=True, timeout=10,
                          headers={"User-Agent": "Mozilla/5.0"})
        return normalise_url(r.url)
    except Exception:
        return url


def extract_keywords(text):
    """Extract meaningful keywords from a headline for topic comparison."""
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'and', 'or', 'but', 'not', 'no', 'it', 'its', 'this', 'that',
        'uk', 'new', 'will', 'has', 'have', 'had', 'may', 'could',
        'would', 'should', 'as', 'up', 'out', 'over', 'after', 'into',
        'about', 'says', 'said', 'set', 'get', 'gets', 'got', 'more',
        'what', 'how', 'why', 'when', 'who', 'which', 'their', 'they',
        'plan', 'plans', 'update', 'latest', 'breaking', 'just', 'now',
        'news', 'report', 'reports', 'announced', 'announces', 'amid',
    }
    words = re.findall(r'[a-z]+', text.lower())
    return set(w for w in words if w not in stop_words and len(w) > 2)


def is_duplicate_topic(new_headline, used_headlines, threshold=0.5):
    """Return True if new_headline covers the same topic as any used headline.

    Uses keyword overlap: if >= 50 % of the smaller keyword set overlaps,
    the headlines are considered the same topic.
    """
    new_kw = extract_keywords(new_headline)
    if not new_kw:
        return False
    for old in used_headlines:
        old_kw = extract_keywords(old)
        if not old_kw:
            continue
        overlap = len(new_kw & old_kw) / min(len(new_kw), len(old_kw))
        if overlap >= threshold:
            return True
    return False


def get_posted_history():
    try:
        r = requests.get(APPS_SCRIPT_URL, timeout=20, allow_redirects=True)
        r.raise_for_status()
        rows = r.json()
    except Exception as e:
        print(f"[WARN] Could not fetch history: {e} — continuing with empty history")
        return set(), set(), []

    seen_urls, seen_ids, used_headlines = set(), set(), []
    for row in rows:
        u = normalise_url(str(row.get("tiktok_url", "") or ""))
        v = str(row.get("video_id", "") or "").strip()
        h = str(row.get("headline", "") or "").strip()
        if u: seen_urls.add(u)
        if v: seen_ids.add(v)
        if h: used_headlines.append(h)

    return seen_urls, seen_ids, used_headlines


def get_immigration_news(used_headlines=None):
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    exclusion = ""
    if used_headlines:
        # Build topic keywords from used headlines to prevent same-topic suggestions
        all_topics = set()
        for h in used_headlines:
            all_topics.update(extract_keywords(h))
        topic_str = ", ".join(sorted(all_topics)[:40])

        exclusion = (
            "\n\nCRITICAL — Do NOT suggest any of these stories OR any story on the same topic. "
            "Each story below has ALREADY been covered:\n"
            + "\n".join(f"- {h}" for h in used_headlines)
            + f"\n\nAvoid stories involving these topics/keywords: {topic_str}"
            + "\n\nYou MUST suggest a COMPLETELY DIFFERENT immigration story that is not related to any of the above."
        )

    payload = {
        "model": "sonar",
        "messages": [
            {
                "role": "system",
                "content": "You are a news assistant. Always reply in the exact format requested. No extra text, no markdown, no preamble. Never repeat a topic you have been told to avoid."
            },
            {
                "role": "user",
                "content": (
                    "What is the single most important UK immigration news story from the last 20 days? "
                    "Reply in exactly this format and nothing else:\n"
                    "HEADLINE: <headline>\n"
                    "SUMMARY: <two sentence summary>\n"
                    "CAPTION: <engaging Instagram caption with relevant emojis and hashtags>"
                    + exclusion
                )
            }
        ]
    }

    r = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers=headers,
        json=payload,
        timeout=30
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"].strip()

    headline, summary, caption = "", "", ""
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("HEADLINE:"):
            headline = line.split("HEADLINE:", 1)[1].strip()
        elif line.startswith("SUMMARY:"):
            summary = line.split("SUMMARY:", 1)[1].strip()
        elif line.startswith("CAPTION:"):
            caption = line.split("CAPTION:", 1)[1].strip()

    return headline, summary, caption


def search_tiktok(keyword, seen_urls, seen_ids):
    run_url = f"https://api.apify.com/v2/acts/clockworks~tiktok-scraper/runs?token={APIFY_API_KEY}"
    r = requests.post(run_url, json={"searchQueries": [keyword], "resultsPerPage": 10}, timeout=30)
    r.raise_for_status()
    run_id = r.json()["data"]["id"]

    run_info = None
    for _ in range(36):
        time.sleep(10)
        status_r = requests.get(
            f"https://api.apify.com/v2/actor-runs/{run_id}?token={APIFY_API_KEY}",
            timeout=15
        )
        status_r.raise_for_status()
        run_info = status_r.json()["data"]
        if run_info["status"] == "SUCCEEDED":
            break
        if run_info["status"] in ["FAILED", "ABORTED", "TIMED-OUT"]:
            raise Exception(f"Apify run failed: {run_info['status']}")

    results = requests.get(
        f"https://api.apify.com/v2/datasets/{run_info['defaultDatasetId']}/items?token={APIFY_API_KEY}",
        timeout=15
    ).json()

    if not results:
        return None

    trusted, others = [], []
    for video in results:
        url = normalise_url(video.get("webVideoUrl", ""))
        # Resolve shortened links so we can always extract the video ID
        url = resolve_short_url(url)
        vid = extract_video_id(url)
        if not url:
            continue
        if url in seen_urls or (vid and vid in seen_ids):
            continue
        author = (video.get("authorMeta", {}).get("name", "") or "").lower()
        if any(t in author for t in TRUSTED_ACCOUNTS):
            trusted.append(video)
        else:
            others.append(video)

    trusted.sort(key=lambda x: x.get("playCount") or 0, reverse=True)
    others.sort(key=lambda x: x.get("playCount") or 0, reverse=True)

    return trusted[0] if trusted else (others[0] if others else None)


def download_video(video):
    # FIX 3: Use Apify's direct URL first — yt-dlp is unreliable for TikTok
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
        except Exception:
            pass  # fall through to yt-dlp

    # Fallback: yt-dlp
    return _ytdlp_download(normalise_url(video.get("webVideoUrl", "")))


def _ytdlp_download(tiktok_url):
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


def upload_to_cloudinary(file_path):
    result = cloudinary.uploader.upload(
        file_path,
        resource_type="video",
        folder="immigration_news",
        use_filename=True,
        unique_filename=True
    )
    return result["secure_url"]


def write_to_sheet(headline, summary, caption, cloudinary_url, tiktok_source, tiktok_url, video_id):
    r = requests.post(APPS_SCRIPT_URL, json={
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "headline": headline,
        "summary": summary,
        "caption": caption,
        "cloudinary_url": cloudinary_url,
        "tiktok_source": tiktok_source,
        "tiktok_url": tiktok_url,
        "video_id": video_id
    }, timeout=15)
    r.raise_for_status()


@app.get("/run")
def run_pipeline():
    tmp_path = None
    try:
        seen_urls, seen_ids, used_headlines = get_posted_history()

        # Try up to 3 times to get a genuinely new topic
        headline, summary, caption = "", "", ""
        rejected = []
        for attempt in range(3):
            headline, summary, caption = get_immigration_news(used_headlines + rejected)
            if not headline:
                return JSONResponse({"status": "no_news", "message": "No immigration news found"})
            if is_duplicate_topic(headline, used_headlines):
                print(f"[INFO] Attempt {attempt+1}: Rejected duplicate topic: {headline}")
                rejected.append(headline)
                headline = ""
                continue
            break

        if not headline:
            return JSONResponse({
                "status": "no_fresh_topic",
                "message": "Could not find a fresh news topic after 3 attempts",
                "rejected": rejected
            })

        video = search_tiktok(f"UK immigration {headline[:60]}", seen_urls, seen_ids)
        if not video:
            return JSONResponse({"status": "no_fresh_video", "message": "No new unseen TikTok video found"})

        author = (video.get("authorMeta", {}).get("name", "") or "unknown").strip()
        tiktok_url = normalise_url(video.get("webVideoUrl", ""))
        video_id = extract_video_id(tiktok_url)
        play_count = int(video.get("playCount") or 0)

        if not tiktok_url:
            return JSONResponse({"status": "no_video_url", "message": "No usable TikTok URL on selected video"})

        tmp_path = download_video(video)  # passes full dict now
        cloudinary_url = upload_to_cloudinary(tmp_path)
        write_to_sheet(headline, summary, caption, cloudinary_url, author, tiktok_url, video_id)

        return JSONResponse({
            "status": "success",
            "cloudinary_url": cloudinary_url,
            "caption": caption,
            "headline": headline,
            "summary": summary,
            "tiktok_source": author,
            "tiktok_url": tiktok_url,
            "video_id": video_id,
            "play_count": play_count
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/health")
def health():
    return {"status": "ok"}
