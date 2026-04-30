"""
Breaking News TikTok Pipeline — Cohby Consulting Services
----------------------------------------------------------
GitHub Actions Edition (runs every 6 hours via cron)

Architecture:
  1. Fetch seen IDs/URLs + recent summaries from n8n Data Fetch webhook
  2. Scrape tiered news accounts via yt-dlp (Tier 1 UK-first, expand only if needed)
  3. Deduplicate by video ID/URL — no AI cost yet
  4. Filter by engagement threshold (MIN_ENGAGEMENT)
  5. Single DeepSeek judge call — picks the ONE best breaking news story
     from all candidates simultaneously, with duplicate awareness
  6. Download winner → upload to Cloudinary
  7. Generate Cohby-branded caption via DeepSeek
  8. POST result JSON to n8n webhook for social publishing
"""

import os
import re
import glob
import time
import json
import requests
import cloudinary
import cloudinary.uploader
from typing import List, Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEEPSEEK_API_KEY   = os.environ["DEEPSEEK_API_KEY"]
N8N_WEBHOOK_URL    = os.environ["N8N_WEBHOOK_URL"]
N8N_DATA_FETCH_URL = os.environ["N8N_DATA_FETCH_URL"]

cloudinary.config(
    cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"],
    api_key=os.environ["CLOUDINARY_API_KEY"],
    api_secret=os.environ["CLOUDINARY_API_SECRET"],
)

MIN_ENGAGEMENT          = 1000   # minimum views OR likes to qualify
VIDEOS_PER_ACCOUNT      = 10     # how many recent videos to pull per account
TIER1_MIN_CANDIDATES    = 5      # expand to Tier 2 if Tier 1 yields fewer than this
TIER2_MIN_CANDIDATES    = 5      # expand to Tier 3 if Tier 1+2 yields fewer than this
MAX_CANDIDATES_TO_JUDGE = 40     # cap sent to DeepSeek judge to control prompt size

# ---------------------------------------------------------------------------
# Tiered account pools
# Tier 1 — UK primary sources (scraped first, always)
# Tier 2 — Major US + international wire services
# Tier 3 — Regional/secondary (AU, CA, NZ) — fallback only
# ---------------------------------------------------------------------------
ACCOUNTS_TIER1 = [
    # UK
    "bbcnews",
    "bbcworldservice",
    "skynews",
    "itvnews",
    "c4news",
    "theguardian",
    "thetimes",
    "standardnews",
    "telegraphnews",
    # High-tier international wires
    "reuters",
    "apnews",
    "aljazeeraenglish",
]

ACCOUNTS_TIER2 = [
    # USA
    "cnn",
    "nbcnews",
    "cbsnews",
    "abcnews",
    "washingtonpost",
    "nytimes",
    "thehill",
    "politico",
    "axios",
    "nprnews",
    # Europe
    "dwnews",
    "france24_en",
    "euronews",
    "trtworld",
]

ACCOUNTS_TIER3 = [
    # Canada
    "cbcnews",
    "globalnews",
    "ctvnews",
    # Australia
    "abcnewsaus",
    "9newsaus",
    "7newsaustralia",
    # New Zealand (last resort)
    "rnz_news",
    "1newsnz",
    "stuffnz",
    "nzherald",
]

# Fox News excluded intentionally — high opinion/partisan content ratio
# foxnews removed from pool to avoid AI gate confusion


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
        r = requests.head(
            url, allow_redirects=True, timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        return normalise_url(r.url)
    except Exception:
        return url


def meets_engagement(video: dict) -> bool:
    plays = int(video.get("playCount") or 0)
    likes = int(video.get("diggCount") or 0)
    return plays >= MIN_ENGAGEMENT or likes >= MIN_ENGAGEMENT


# ---------------------------------------------------------------------------
# yt-dlp scraper — returns raw video dicts for given accounts
# ---------------------------------------------------------------------------
def scrape_accounts(accounts: List[str], videos_per_account: int = VIDEOS_PER_ACCOUNT) -> List[dict]:
    from yt_dlp import YoutubeDL

    all_videos   = []
    failed       = []
    zero_results = []

    ydl_opts = {
        "quiet":       True,
        "noprogress":  True,
        "skip_download": True,
        "playlistend": videos_per_account,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        },
    }

    for account in accounts:
        url = f"https://www.tiktok.com/@{account}"
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info    = ydl.extract_info(url, download=False)
                entries = (info.get("entries") or []) if info else []

            fetched = 0
            for entry in entries:
                if not entry:
                    continue
                all_videos.append({
                    "webVideoUrl": entry.get("webpage_url", ""),
                    "text":        entry.get("description", ""),
                    "playCount":   entry.get("view_count",  0),
                    "diggCount":   entry.get("like_count",  0),
                    "videoUrl":    entry.get("url",         ""),
                    "authorMeta":  {"name": account},
                    "tier":        _account_tier(account),
                })
                fetched += 1

            if fetched == 0:
                zero_results.append(account)
                print(f"[WARN] @{account}: 0 videos returned (account may be inactive/renamed)")
            else:
                print(f"[OK]   @{account}: {fetched} videos")

        except Exception as e:
            failed.append(account)
            print(f"[FAIL] @{account}: {e}")

    # Pre-flight health summary
    if failed:
        print(f"\n[PREFLIGHT] {len(failed)} accounts failed to scrape: {failed}")
    if zero_results:
        print(f"[PREFLIGHT] {len(zero_results)} accounts returned 0 videos: {zero_results}")
    print(f"[INFO] Total raw videos fetched: {len(all_videos)}")

    return all_videos


def _account_tier(account: str) -> int:
    if account in ACCOUNTS_TIER1:
        return 1
    if account in ACCOUNTS_TIER2:
        return 2
    return 3


# ---------------------------------------------------------------------------
# Candidate builder — deduplicates, filters engagement, resolves URLs
# No AI cost at this stage
# ---------------------------------------------------------------------------
def build_candidates(
    all_videos: List[dict],
    seen_ids:   set,
    seen_urls:  set,
) -> List[dict]:
    candidates = []
    seen_in_batch = set()  # avoid processing the same video twice across accounts

    for video in all_videos:
        url = normalise_url(video.get("webVideoUrl", ""))
        url = resolve_short_url(url)
        vid = extract_video_id(url)

        if not url or not vid:
            continue
        if vid in seen_in_batch:
            continue
        if url in seen_urls or vid in seen_ids:
            print(f"[SKIP] Already seen: {vid}")
            continue
        if not meets_engagement(video):
            print(f"[SKIP] Below threshold — plays={video.get('playCount')} likes={video.get('diggCount')} | {url}")
            continue

        video["_url_clean"] = url
        video["_vid_id"]    = vid
        candidates.append(video)
        seen_in_batch.add(vid)

    # Sort by tier first, then by play count descending — gives judge best signal
    candidates.sort(key=lambda x: (x.get("tier", 3), -(x.get("playCount") or 0)))
    print(f"[INFO] Qualified candidates after dedup + engagement filter: {len(candidates)}")
    return candidates


# ---------------------------------------------------------------------------
# DeepSeek Judge — single call to pick the best breaking news story
# Handles breaking-news classification, geographic relevance, AND duplicate
# check all in one prompt. Returns the chosen candidate dict or None.
# ---------------------------------------------------------------------------
def deepseek_judge(
    candidates:       List[dict],
    recent_summaries: List[str],
) -> Optional[dict]:
    if not candidates:
        return None

    # Cap to avoid very long prompts
    pool = candidates[:MAX_CANDIDATES_TO_JUDGE]

    # Build numbered story list
    stories_block = ""
    for i, v in enumerate(pool, 1):
        desc    = (v.get("text", "") or "").strip()[:300]
        account = v.get("authorMeta", {}).get("name", "unknown")
        plays   = v.get("playCount", 0)
        stories_block += f"{i}. [@{account} | {plays:,} views]\n{desc}\n\n"

    # Recent topics block for duplicate awareness
    if recent_summaries:
        recent_block = "\n".join(f"- {s[:150]}" for s in recent_summaries[:20])
        recent_section = (
            f"\nALREADY POSTED RECENTLY (do NOT pick a story covering the same topic as any of these):\n"
            f"{recent_block}\n"
        )
    else:
        recent_section = ""

    prompt = (
        "You are the senior news editor for a UK-based professional consultancy's social media channel.\n"
        "Your job is to select the single most important breaking news story from the list below.\n\n"
        "SELECTION CRITERIA (all must be true):\n"
        "1. BREAKING NEWS ONLY — an actively unfolding event: major disaster, terror attack, "
        "political crisis, military escalation, significant geopolitical development, or sudden "
        "high-impact event. NOT analysis, opinion, feature, sport, entertainment, or lifestyle.\n"
        "2. UK OR INTERNATIONAL SIGNIFICANCE — the story must matter to a UK audience or be of "
        "clear global importance. Reject stories that are purely local to a small country or region "
        "(e.g. a domestic story only relevant inside New Zealand, Australia, or a single US state).\n"
        "3. NOT A DUPLICATE — do not pick any story that covers the same core topic as the "
        "recently posted stories listed below.\n\n"
        f"{recent_section}"
        "CANDIDATE STORIES:\n\n"
        f"{stories_block}"
        "INSTRUCTIONS:\n"
        "- Reply with ONLY the number of the best qualifying story (e.g. '7').\n"
        "- If multiple stories qualify, pick the most urgent and globally significant one.\n"
        "- If NO story qualifies (all are irrelevant, local, or duplicate), reply with: NONE\n"
        "- Do NOT explain your reasoning. Output only the number or NONE."
    )

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":    "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 0.0,
    }

    try:
        r = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers, json=payload, timeout=30
        )
        r.raise_for_status()
        answer = r.json()["choices"][0]["message"]["content"].strip().upper()
        print(f"[JUDGE] DeepSeek selected: {answer}")

        if answer == "NONE":
            return None

        # Parse number
        match = re.search(r"\d+", answer)
        if not match:
            print(f"[WARN] Judge returned unexpected response: {answer}")
            return None

        idx = int(match.group()) - 1  # convert to 0-based index
        if 0 <= idx < len(pool):
            chosen = pool[idx]
            print(f"[JUDGE] Chosen: @{chosen.get('authorMeta', {}).get('name')} | {chosen.get('_url_clean')}")
            return chosen
        else:
            print(f"[WARN] Judge index {idx+1} out of range (pool size: {len(pool)})")
            return None

    except Exception as e:
        print(f"[ERROR] DeepSeek judge failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Caption generator — Cohby Consulting Services brand voice
# ---------------------------------------------------------------------------
def generate_caption(video_description: str) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type":  "application/json",
    }
    prompt = (
        "You write social media captions for Cohby Consulting Services, a professional UK-based "
        "immigration and business consultancy. Your audience is professionals, entrepreneurs, and "
        "individuals who care about UK and global current affairs.\n\n"
        "Write a punchy, professional caption for the following breaking news story.\n\n"
        "Requirements:\n"
        "- 2-3 sentences, factual, clear, and engaging — no sensationalism\n"
        "- Professional tone fitting a UK consultancy brand\n"
        "- Include 3-5 relevant hashtags (topic, geography, #BreakingNews)\n"
        "- 1-2 appropriate emojis maximum\n"
        "- Do NOT include URLs, phone numbers, or source attribution\n"
        "- Output ONLY the caption text, nothing else\n\n"
        f"News description:\n\"{video_description[:600]}\""
    )
    payload = {
        "model":    "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 220,
        "temperature": 0.6,
    }
    r = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers, json=payload, timeout=30
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Download video (direct URL first, yt-dlp fallback)
# ---------------------------------------------------------------------------
def download_video(video: dict) -> str:
    direct_url = (
        video.get("videoUrl")
        or video.get("downloadAddr")
        or (video.get("videoMeta") or {}).get("downloadAddr")
    )
    tmp_path = f"/tmp/cohby_news_{int(time.time())}.mp4"

    if direct_url:
        try:
            resp = requests.get(
                direct_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Referer":    "https://www.tiktok.com/",
                },
                timeout=60,
                stream=True,
            )
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 10_000:
                print("[INFO] Downloaded via direct URL")
                return tmp_path
        except Exception as e:
            print(f"[WARN] Direct download failed ({e}), falling back to yt-dlp")

    return _ytdlp_download(video.get("_url_clean", normalise_url(video.get("webVideoUrl", ""))))


def _ytdlp_download(tiktok_url: str) -> str:
    from yt_dlp import YoutubeDL

    outtmpl  = f"/tmp/cohby_news_{int(time.time())}.%(ext)s"
    ydl_opts = {
        "outtmpl":              outtmpl,
        "format":               "best[ext=mp4]/best",
        "merge_output_format":  "mp4",
        "quiet":                True,
        "noprogress":           True,
        "noplaylist":           True,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.tiktok.com/",
        },
        "extractor_args": {
            "tiktok": {
                "app_version":      ["20.9.3"],
                "manifest_app_version": ["291"],
            }
        },
    }

    with YoutubeDL(ydl_opts) as ydl:
        info      = ydl.extract_info(tiktok_url, download=True)
        file_path = ydl.prepare_filename(info)

    if not os.path.exists(file_path):
        base    = os.path.splitext(file_path)[0]
        matches = glob.glob(base + ".*")
        mp4s    = [m for m in matches if m.endswith(".mp4")]
        file_path = mp4s[0] if mp4s else (matches[0] if matches else file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found after yt-dlp download: {file_path}")
    if os.path.getsize(file_path) < 10_000:
        raise ValueError(f"Downloaded file suspiciously small: {file_path}")

    print(f"[INFO] Downloaded via yt-dlp: {file_path}")
    return file_path


# ---------------------------------------------------------------------------
# Cloudinary upload
# ---------------------------------------------------------------------------
def upload_to_cloudinary(file_path: str) -> str:
    result = cloudinary.uploader.upload(
        file_path,
        resource_type="video",
        folder="cohby_breaking_news",
        use_filename=True,
        unique_filename=True,
    )
    url = result["secure_url"]
    print(f"[INFO] Uploaded to Cloudinary: {url}")
    return url


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Cohby Consulting — Breaking News TikTok Pipeline")
    print(f"Run started: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1 — Fetch seen data from n8n
    # ------------------------------------------------------------------
    seen_ids         = set()
    seen_urls        = set()
    recent_summaries = []

    try:
        resp = requests.get(N8N_DATA_FETCH_URL, timeout=30)
        resp.raise_for_status()
        data             = resp.json()
        seen_ids         = set(str(i) for i in data.get("seen_ids",         []) if i)
        seen_urls        = set(str(u) for u in data.get("seen_urls",        []) if u)
        recent_summaries = [str(s) for s in data.get("recent_summaries", []) if s]
        print(f"[INFO] n8n data fetched — {len(seen_ids)} seen IDs, {len(recent_summaries)} recent summaries")
    except Exception as e:
        print(f"[WARN] Could not fetch n8n data ({e}) — proceeding with empty state")

    # ------------------------------------------------------------------
    # Step 2 — Tiered scraping (expand tiers only if needed)
    # ------------------------------------------------------------------
    all_videos = scrape_accounts(ACCOUNTS_TIER1)
    candidates = build_candidates(all_videos, seen_ids, seen_urls)

    if len(candidates) < TIER1_MIN_CANDIDATES:
        print(f"[INFO] Tier 1 yielded only {len(candidates)} candidates — expanding to Tier 2")
        tier2_videos = scrape_accounts(ACCOUNTS_TIER2)
        all_videos  += tier2_videos
        candidates   = build_candidates(all_videos, seen_ids, seen_urls)

    if len(candidates) < TIER2_MIN_CANDIDATES:
        print(f"[INFO] Tier 1+2 yielded only {len(candidates)} candidates — expanding to Tier 3")
        tier3_videos = scrape_accounts(ACCOUNTS_TIER3)
        all_videos  += tier3_videos
        candidates   = build_candidates(all_videos, seen_ids, seen_urls)

    if not candidates:
        print("[INFO] No candidates after filtering — nothing to post. Exiting.")
        return

    # ------------------------------------------------------------------
    # Step 3 — DeepSeek judge: single call picks the best story
    # ------------------------------------------------------------------
    print(f"\n[JUDGE] Sending {min(len(candidates), MAX_CANDIDATES_TO_JUDGE)} candidates to DeepSeek judge...")
    chosen = deepseek_judge(candidates, recent_summaries)

    if not chosen:
        print("[INFO] DeepSeek judge found no qualifying breaking news story. Exiting.")
        return

    # ------------------------------------------------------------------
    # Step 4 — Extract metadata
    # ------------------------------------------------------------------
    author      = (chosen.get("authorMeta", {}).get("name", "unknown") or "unknown").strip()
    tiktok_url  = chosen.get("_url_clean", normalise_url(chosen.get("webVideoUrl", "")))
    video_id    = chosen.get("_vid_id", extract_video_id(tiktok_url))
    play_count  = int(chosen.get("playCount") or 0)
    like_count  = int(chosen.get("diggCount") or 0)
    description = (chosen.get("text", "") or "").strip()
    tier        = chosen.get("tier", "?")

    print(f"\n[INFO] Winner — @{author} (Tier {tier}) | ID: {video_id} | plays: {play_count:,}")
    print(f"[INFO] Description: {description[:120]}")

    # ------------------------------------------------------------------
    # Step 5 — Generate branded caption
    # ------------------------------------------------------------------
    print("\n[INFO] Generating Cohby caption...")
    try:
        caption = generate_caption(description)
        print(f"[INFO] Caption: {caption[:100]}...")
    except Exception as e:
        print(f"[WARN] Caption generation failed ({e}) — using raw description")
        caption = description[:500]

    # ------------------------------------------------------------------
    # Step 6 — Download + upload to Cloudinary
    # ------------------------------------------------------------------
    print("\n[INFO] Downloading video...")
    tmp_path = download_video(chosen)
    try:
        print("[INFO] Uploading to Cloudinary...")
        cloudinary_url = upload_to_cloudinary(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            print(f"[INFO] Temp file cleaned up")

    # ------------------------------------------------------------------
    # Step 7 — POST to n8n webhook
    # ------------------------------------------------------------------
    payload = {
        "status":          "success",
        "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "headline":        description[:200],
        "summary":         description,
        "caption":         caption,
        "cloudinary_url":  cloudinary_url,
        "tiktok_source":   author,
        "tiktok_url":      tiktok_url,
        "video_id":        video_id,
        "play_count":      play_count,
        "like_count":      like_count,
        "source_tier":     tier,
    }

    print("\n[INFO] Posting to n8n webhook...")
    r = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=30)
    r.raise_for_status()
    print(f"[INFO] n8n webhook response: {r.status_code}")
    print("\n[DONE] Pipeline complete.")


if __name__ == "__main__":
    main()
