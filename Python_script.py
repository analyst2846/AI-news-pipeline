# Quebec City news intelligence pipeline - ported from the jupyter notebook
# for Cloud Run Jobs. Scrapes sitemaps, scores relevance via Gemini, scrapes
# full articles, runs sentiment/clustering/classification, then dumps to BQ.


import subprocess
import sys

try:
    import pyarrow
    print(f"pyarrow {pyarrow.__version__} OK ({pyarrow.__file__})")
except ImportError as e:
    print(f"pyarrow missing: {e}")
    subprocess.check_call([sys.executable, '-m', 'pip', 'freeze'])



import gc
import os
import sys
import json
import csv
import re
import math
import time
import random
import asyncio
import logging
import threading
import numpy as np
import pandas as pd
import requests
import torch
import xml.etree.ElementTree as ET

from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Optional, Union, List, Dict, Literal
from email.utils import parsedate_to_datetime

from trafilatura import extract as trafilatura_extract
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field, field_validator, ValidationError

from google import genai
from google.genai import types
from google.cloud import bigquery
import vertexai

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from playwright_stealth import Stealth

# --- config (env vars w/ defaults) ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "---")
DATASET_ID = os.environ.get("BQ_DATASET_ID", "---")
TABLE_ID = os.environ.get("BQ_TABLE_ID", "---")
LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
GEMINI_LOCATION = os.environ.get("GEMINI_LOCATION", "global")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# gpu if available, else cpu
TORCH_DEVICE = 0 if torch.cuda.is_available() else -1
TORCH_DEVICE_STR = "cuda:0" if torch.cuda.is_available() else "cpu"
log.info(f"torch device: {TORCH_DEVICE_STR}")


# -------------------------------------------------------
#  STAGE 1 - sitemap scraping
# -------------------------------------------------------

# domain-level rate limiter (threadsafe)
_last_request = {}
_domain_locks = {}
_global_lock = threading.Lock()

SITEMAP_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# pick one UA for the whole session so we don't look like a bot
_SESSION_UA = random.choice(SITEMAP_USER_AGENTS)

SITEMAPS = [
    "https://www.lapresse.ca/newsSitemap.xml",
    "https://ici.radio-canada.ca/info/sitemaps.xml",
    "https://www.tvanouvelles.ca/news.xml",
    "https://www.24heures.ca/news.xml",
    "https://www.journaldequebec.com/news.xml",
    "https://www.carrefourdequebec.com/feed/",
    "https://www.lesoleil.com/arc/outboundfeeds/sitemap-news?outputType=xml",
    "https://www.latribune.ca/arc/outboundfeeds/sitemap-news/?outputType=xml",
    "https://www.ledevoir.com/rss/section/politique/ville-de-quebec.xml",
    "https://www.ledevoir.com/rss/section/actualites.xml?id=10",
    "https://www.ledevoir.com/rss/section/economie.xml?id=49",
    "https://www.ledevoir.com/rss/section/culture.xml?id=48",
    "https://www.ledevoir.com/rss/section/plaisirs.xml?id=50",
    "https://globalnews.ca/news-sitemap.xml",
    "https://www.narcity.com/feeds/sitemaps/news_1.xml",
    "https://montrealgazette.com/sitemap-news.xml",
    "https://nationalpost.com/sitemap-news.xml",
    "https://www.theglobeandmail.com/arc/outboundfeeds/rss/category/canada/",
    "https://www.mtlblog.com/feeds/sitemaps/news_1.xml",
    "https://www.journaldemontreal.com/news.xml"
  
]

# serpapi config
SERPAPI_KEY = "---"
SERPAPI_MAX_CALLS = 8

SERPAPI_SEARCHES = [
    {"q": "quebec when:1d", "hl": "en", "gl": "ca", "max_pages": 2},
    {"q": "quebec when:1d", "hl": "fr", "gl": "ca", "max_pages": 2},
    {"q": "quebec tourism when:1d", "hl": "en", "gl": "us", "max_pages": 1},
    {"q": "Tourisme Québec when:1d", "hl": "fr", "gl": "fr", "max_pages": 1},
    {"q": "Quebec tourisme when:1d site:grenier.qc.ca OR site:lanouvelle.net OR site:lequotidien.com OR site:noovomoi.ca", "hl": "fr", "gl": "ca", "max_pages": 1},
    {"q": "Quebec when:1d site:monvieuxquebec.com OR site:silo57.ca OR site:lhebdojournal.com OR site:monquartier.quebec OR site:urbania.ca OR site:autourdelile.com OR site:tourismexpress.com", "hl": "fr", "gl": "ca", "max_pages": 1}
]

# strip these before comparing urls for dedup
_STRIP_PARAMS = {'utm_source', 'utm_medium', 'utm_campaign', 'utm_content', 'utm_term', 'ref', 'fbclid', 'gclid'}


def _normalize_url_for_dedup(url):
    """Remove tracking junk so we can dedup properly."""
    try:
        parsed = urlparse(url)
        params = {k: v for k, v in parse_qs(parsed.query).items() if k not in _STRIP_PARAMS}
        clean = parsed._replace(query=urlencode(params, doseq=True), fragment="")
        return urlunparse(clean)
    except Exception:
        return url


def _get_domain(url):
    return urlparse(url).netloc


def _get_domain_lock(domain):
    with _global_lock:
        if domain not in _domain_locks:
            _domain_locks[domain] = threading.Lock()
        return _domain_locks[domain]


def _respectful_request(url, timeout=15, max_retries=3):
    domain = _get_domain(url)
    domain_lock = _get_domain_lock(domain)
    with domain_lock:
        min_gap = random.uniform(5.8, 7.6) if "news.google.com" in domain else 10
        if domain in _last_request:
            elapsed = time.time() - _last_request[domain]
            if elapsed < min_gap:
                time.sleep(min_gap - elapsed)
        headers = {
            "User-Agent": _SESSION_UA,
            "Accept": "application/xml, text/xml, */*",
            "Accept-Language": "en-US,en;q=0.9,fr-CA;q=0.8",
            "DNT": "1",
        }
        last_exc = None
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=timeout)
                response.raise_for_status()
                return response
            except requests.exceptions.HTTPError as e:
                last_exc = e
                status = e.response.status_code if e.response is not None else 0
                # back off on 429/503
                ra = e.response.headers.get("Retry-After") if e.response is not None else None
                if status in (429, 503) and attempt < max_retries - 1:
                    backoff = int(ra) if (ra and ra.isdigit()) else (15 * (attempt + 1))
                    backoff += random.uniform(1, 5)
                    log.warning(f"  ⏳ {domain} returned {status}, backing off {backoff:.0f}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(backoff)
                else:
                    raise
            except requests.exceptions.RequestException as e:
                last_exc = e
                if attempt < max_retries - 1:
                    backoff = 10 * (attempt + 1)
                    log.warning(f"  ⏳ {domain} request failed ({e}), retrying in {backoff}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(backoff)
                else:
                    raise
            finally:
                # always stamp last request time even on failures
                _last_request[domain] = time.time()
        raise last_exc



def _parse_date(date_string):
    if not date_string:
        return None
    date_string = date_string.strip()
    try:
        try:
            dt = parsedate_to_datetime(date_string)
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        except Exception:
            pass
        if "T" in date_string:
            if date_string.endswith("Z"):
                date_string = date_string.replace("Z", "+00:00")
            dt = datetime.fromisoformat(date_string)
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S"]:
            try:
                return datetime.strptime(date_string, fmt)
            except Exception:
                continue
    except Exception:
        pass
    return None


def _parse_relative_date(date_str):
    """Handle serpapi dates -- '2 hours ago' style or 'Nov 17, 2025' absolute."""
    if not date_str:
        return None
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    date_str_lower = date_str.lower().strip()

    # relative: "X minutes/hours/days/weeks ago"
    match = re.match(r"(\d+)\s+(minute|hour|day|week)s?\s+ago", date_str_lower)
    if match:
        val = int(match.group(1))
        unit = match.group(2)
        if unit == "minute":
            return now - timedelta(minutes=val)
        elif unit == "hour":
            return now - timedelta(hours=val)
        elif unit == "day":
            return now - timedelta(days=val)
        elif unit == "week":
            return now - timedelta(weeks=val)

    if "yesterday" in date_str_lower:
        return now - timedelta(days=1)
    if "just now" in date_str_lower:
        return now

    # absolute dates
    date_str_clean = date_str.strip()
    for fmt in [
        "%b %d, %Y, %I:%M:%S %p",   # Nov 17, 2025, 12:52:00 PM
        "%b %d, %Y, %I:%M %p",       # Nov 17, 2025, 12:52 PM
        "%b %d, %Y",                  # Nov 17, 2025
        "%B %d, %Y, %I:%M:%S %p",    # November 17, 2025, 12:52:00 PM
        "%B %d, %Y, %I:%M %p",       # November 17, 2025, 12:52 PM
        "%B %d, %Y",                  # November 17, 2025
        "%m/%d/%Y %I:%M %p",         # 11/17/2025 12:52 PM
        "%m/%d/%Y",                   # 11/17/2025
        "%Y-%m-%d %H:%M:%S",         # 2025-11-17 12:52:00
        "%Y-%m-%d",                   # 2025-11-17
        "%d %b %Y",                   # 17 Nov 2025
        "%d %B %Y",                   # 17 November 2025
    ]:
        try:
            return datetime.strptime(date_str_clean, fmt)
        except ValueError:
            continue

    # fall back to ISO/RFC parser
    return _parse_date(date_str)


def _parse_sitemap(sitemap_url):
    articles = []
    past_24h = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=24)
    is_google_news = "news.google.com" in sitemap_url
    try:
        log.info(f"  → Fetching {sitemap_url}")
        response = _respectful_request(sitemap_url)
        root = ET.fromstring(response.content)
        total = 0
        recent = 0

        if root.tag == "rss" or root.tag.endswith("}rss"):
            for item in root.findall(".//item"):
                total += 1
                link = item.find("link")
                pub_date = item.find("pubDate")
                title = item.find("title")
                if link is not None and link.text:
                    url = link.text.strip()

                    # keep raw google url for now, decode later in stage 3
                    date_string = pub_date.text if pub_date is not None else None
                    article_date = _parse_date(date_string)
                    if article_date and article_date >= past_24h:
                        articles.append({
                            "url": url,
                            "date": date_string,
                            "title": title.text if title is not None else "",
                            "source": _get_domain(sitemap_url),
                            "is_google_redirect": is_google_news,
                        })
                        recent += 1
        else:
            namespaces = [
                "http://www.sitemaps.org/schemas/sitemap/0.9",
                "https://www.sitemaps.org/schemas/sitemap/0.9",
            ]
            news_namespaces = [
                "http://www.google.com/schemas/sitemap-news/0.9",
                "https://www.google.com/schemas/sitemap-news/0.9",
            ]
            for ns in namespaces:
                for url_elem in root.findall(f".//{{{ns}}}url"):
                    total += 1
                    loc = url_elem.find(f"{{{ns}}}loc")
                    lastmod = url_elem.find(f"{{{ns}}}lastmod")
                    news_title = None
                    news_date = None
                    for news_ns in news_namespaces:
                        if news_title is None:
                            news_title = url_elem.find(f".//{{{news_ns}}}title")
                        if news_date is None:
                            news_date = url_elem.find(f".//{{{news_ns}}}publication_date")
                    if loc is not None and loc.text:
                        date_string = (
                            news_date.text if news_date is not None
                            else (lastmod.text if lastmod is not None else None)
                        )
                        article_date = _parse_date(date_string)
                        if article_date and article_date >= past_24h:
                            articles.append({
                                "url": loc.text.strip(),
                                "date": date_string,
                                "title": news_title.text if news_title is not None else "",
                                "source": _get_domain(sitemap_url),
                                "is_google_redirect": False,
                            })
                            recent += 1

        log.info(f"  ✓ {_get_domain(sitemap_url)}: {recent}/{total} articles from past 24h")
        return articles
    except Exception as e:
        log.warning(f"  ✗ {_get_domain(sitemap_url)}: {e}")
        return []


def _fetch_serpapi() -> List[Dict]:
    """Pull articles from google news via serpapi."""
    articles = []
    total_calls = 0
    past_24h = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=24)
    skipped = 0

    for config in SERPAPI_SEARCHES:
        lang = config["hl"]
        max_pages = config["max_pages"]
        start = 0
        page = 1

        log.info(f"  → SerpAPI: q={config['q']} hl={lang} gl={config['gl']}")

        while page <= max_pages and total_calls < SERPAPI_MAX_CALLS:
            try:
                params = {
                    "engine": "google_news",
                    "q": config["q"],
                    "api_key": SERPAPI_KEY,
                    "gl": config["gl"],
                    "hl": lang,
                    "start": start,
                }

                response = requests.get(
                    "https://serpapi.com/search.json",
                    params=params,
                    timeout=30,
                )
                total_calls += 1

                if response.status_code != 200:
                    log.warning(f"  ✗ SerpAPI error {response.status_code}")
                    break

                data = response.json()
                news = data.get("news_results", [])

                if not news:
                    log.info(f"  SerpAPI [{lang}] page {page}: no results, stopping")
                    break

                page_added = 0
                for item in news:
                    link = item.get("link", "")
                    if not link:
                        continue

                    # skip old stuff
                    article_date = _parse_relative_date(item.get("date", ""))
                    if article_date and article_date < past_24h:
                        skipped += 1
                        continue  # skip articles confirmed older than 24h

                    source_info = item.get("source", {})
                    source_name = (
                        source_info.get("name", "serpapi")
                        if isinstance(source_info, dict)
                        else str(source_info)
                    )

                    articles.append({
                        "url": link,
                        "date": item.get("date", ""),
                        "title": item.get("title", ""),
                        "source": source_name,
                        "is_google_redirect": "news.google.com" in link,
                    })
                    page_added += 1

                log.info(f"  ✓ SerpAPI [{lang}] page {page}: {page_added} articles kept, {len(news) - page_added} skipped (old/no-date)")

                pagination = data.get("serpapi_pagination", {})
                if "next" not in pagination:
                    break

                page += 1
                start += len(news)
                time.sleep(1)

            except Exception as e:
                log.warning(f"  ✗ SerpAPI [{lang}] page {page} failed: {e}")
                break

    log.info(f"  SerpAPI total: {len(articles)} articles ({total_calls}/{SERPAPI_MAX_CALLS} calls used, {skipped} skipped as >24h)")
    return articles


def stage_1_scrape_sitemaps() -> List[Dict]:
    """Grab articles from all the sitemaps, dedup, return last 24h."""
    log.info("--- stage 1: sitemap scraping (last 24h) ---")

    all_articles = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_parse_sitemap, url) for url in SITEMAPS]
        for future in as_completed(futures):
            all_articles.extend(future.result())

    # also pull from serpapi
    serpapi_articles = _fetch_serpapi()
    all_articles.extend(serpapi_articles)

    log.info(f"Total raw articles: {len(all_articles)}")

    # dedup on normalized urls + titles
    seen_urls = set()
    seen_titles = set()
    deduplicated = []
    for article in all_articles:
        normalized_url = _normalize_url_for_dedup(article["url"])
        title = article["title"].strip() if article["title"] else ""
        if normalized_url not in seen_urls and (not title or title not in seen_titles):
            seen_urls.add(normalized_url)
            if title:
                seen_titles.add(title)
            deduplicated.append(article)

    log.info(f"After dedup: {len(deduplicated)} unique articles")
    return deduplicated



# -------------------------------------------------------
#  STAGE 2 - gemini relevance scoring
# -------------------------------------------------------

RELEVANCE_SYSTEM_PROMPT = """You are an elite Media Intelligence Analyst and Regional Data Specialist focused exclusively on the Capitale-Nationale region of Quebec. Your specific domain of expertise lies in identifying, filtering, and classifying news content based on its impact on the region's tourism economy, international reputation, cultural vitality, and major infrastructure.

You will be presented with a raw list of news articles. Your mandate is to evaluate EACH article individually with extreme precision, filtering out noise to highlight content that matters to tourism stakeholders, destination management organizations, and economic developers in the Quebec City area.

### I. GEOGRAPHIC SCOPE & DEFINITIONS
To be considered "Local," an article must specifically pertain to one of these geographic zones. General mentions of "Quebec" usually refer to the province and are NOT sufficient unless the context is explicitly the capital city.

1. **Quebec City (Agglomeration):** La Cité-Limoilou, Les Rivières, Sainte-Foy–Sillery–Cap-Rouge, Charlesbourg, Beauport, La Haute-Saint-Charles, L'Ancienne-Lorette, Saint-Augustin-de-Desmaures, Wendake et les MRC de Portneuf, de La Jacques-Cartier, de l'île d'Orléans et de la Côte-de-Beaupré.
2. **MRC de La Jacques-Cartier:** Stoneham-et-Tewkesbury, Lac-Beauport, Sainte-Brigitte-de-Laval, Saint-Gabriel-de-Valcartier, Shannon.
3. **MRC de La Côte-de-Beaupré:** Beaupré, Sainte-Anne-de-Beaupré, Château-Richer, L'Ange-Gardien, Saint-Ferréol-les-Neiges, Saint-Tite-des-Caps.
4. **MRC de L'Île-d'Orléans:** Sainte-Pétronille, Saint-Laurent, Sainte-Famille, Saint-François, Saint-Jean, Saint-Pierre.
5. **MRC de Portneuf:** Pont-Rouge, Donnacona, Saint-Raymond, Deschambault-Grondines, Neuville, Portneuf, Saint-Casimir.

### II. CLASSIFICATION RUBRIC & SCORING LOGIC

#### A. HIGHLY RELEVANT (Score: 90-100)
The "Must-Read." Content directly about the defined geographic scope AND pertaining to tourism/visitor economy.
* **Major Events:** Winter Carnival, Festival d'été de Québec (FEQ), New France Festival, Grand Prix Cycliste, Red Bull events, major international conventions at Centre des congrès de Québec. Mainly involving attractions/restaurants/lodging and tourism news related to quebec city.
* **Tourism Infrastructure:** Jean-Lesage International Airport (YQB), Port of Quebec cruise terminal operations, VIA Rail local station updates, Tramway project specifically regarding impact on tourist zones like Old Quebec.
* **Hospitality Sector:** Hotel openings/closings/renovations (e.g. Fairmont Le Château Frontenac), significant restaurant openings in tourist hubs (Petit Champlain, Grande Allée), short-term rental regulations (Airbnb) specific to the city.
* **Cultural Attractions:** Musée de la civilisation, MNBAQ, Aquarium du Québec, Wendake Indigenous tourism, Montmorency Falls.
* **Major Sports:** Only high-level events driving tourism (NHL expansion rumors, Olympics events, major sports activities). Disregard sports scores reporting.
* **Natural Disasters/Safety:** Only if impacting major tourist zones (fire in Old Quebec, closure of bridge to Île d'Orléans).

#### B. MODERATELY RELEVANT (Score: 80-89)
The "Ripple Effect." Provincial scope or indirect but tangible downstream effect on Quebec City's image or visitor numbers.
* New Ministry of Tourism funding announcements including the Capitale-Nationale region.
* Highway closures or transportation/infrastructure news (Highway 40/20) significantly blocking tourist access.
* Strikes affecting provincial museums, SAQ (liquor stores), or public transit inconveniencing visitors in the capital.
* General labour shortages in restaurant/lodging/attractions industry across the province that specifically mention Quebec City examples.

#### C. LOW RELEVANCE (Score: 10-79)
General provincial/Canadian news lacking specific leverage on Quebec City tourism: political debates in National Assembly (unless affecting tourism activity), generic business, routine municipal works in non-tourist residential suburbs, general weather.

#### D. NOT RELEVANT (Score: 0-9)
Content to discard: garbage collection schedules,weather (unless its clear that affecting tourism evenst/activities), heath sector news, school board meetings, minor house fires, local community centre bingo nights (unless tourism related), minor hockey scores, local high school soccer, recreational leagues, ER wait times (unless crisis affecting tourists), domestic disputes, minor traffic accidents, petty theft (unless involving a tourist or major landmark), news about "Quebec" referring to nearby cities or the province without local specificity or tourism impact on Quebec City, "unless" it is news about their "major" event/restaurant/lodging.

### III. OPERATIONAL INSTRUCTIONS
1. **Analyze Context:** Do not just look for keywords. "Traffic" in a residential suburb is irrelevant; "Traffic" blocking the Quebec Bridge during tourist season is Highly Relevant.
2. **Determine Score:** Pick a precise integer between 0 and 100 based on the rubric.
3. **Draft Reasoning:** Compose a concise reasoning string. English. Strictly capped at 10 words. Must explain why the score was assigned.
4. **Extract Key Indicators:** Identify 1 to 3 distinct entities or keywords (e.g. "Le Château Frontenac", "YQB", "Winter Carnival") that justified the classification.

### IV. OUTPUT FORMATTING RULES
* Respond with a **JSON ARRAY** only.
* Do not include markdown code block ticks or conversational filler before/after the JSON.
* The JSON must be strictly valid RFC 8259.
* There must be one object for every article provided in the input.

JSON Schema:
[
  {
    "article_id": "String (Exact ID from input)",
    "relevance_score": Integer (0-100),
    "category": "String (One of: HIGHLY_RELEVANT, MODERATELY_RELEVANT, LOW_RELEVANCE, NOT_RELEVANT)",
    "reasoning": "String (Max 10 words, English)",
    "key_indicators": ["String", "String"]
  }
]

Begin classification now."""

# response schema for structured output
class RelevanceResult(BaseModel):
    article_id: str
    relevance_score: int
    category: Literal["HIGHLY_RELEVANT", "MODERATELY_RELEVANT", "LOW_RELEVANCE", "NOT_RELEVANT"]
    reasoning: str
    key_indicators: List[str]


RELEVANCE_BATCH_SIZE = 25
RELEVANCE_DELAY_BETWEEN_CALLS = 2
RELEVANCE_MAX_RETRIES = 5
RELEVANCE_RETRY_WAIT = 30


def stage_2_relevance_analysis(articles: List[Dict]) -> List[Dict]:
    """Run each article through gemini to get a relevance score."""
    log.info("--- stage 2: relevance analysis ---")

    if not articles:
        log.warning("No articles to analyze.")
        return []

    client = genai.Client(vertexai=True, project=PROJECT_ID, location=GEMINI_LOCATION)
    job_start = time.time()

    def _analyze_batch(articles_batch, batch_number):
        articles_text = "\n\n".join([
            f'**Article {i+1} (ID: "{i}"):**\nTitle: {a["title"]}\nURL: {a["url"]}'
            for i, a in enumerate(articles_batch)
        ])
        prompt = (
            f"Analyze the relevance of these {len(articles_batch)} articles to Quebec City "
            f"and the Capitale-Nationale region.\n\n{articles_text}\n\n"
            f"Return a JSON array with analysis for each article, maintaining the article_id order."
        )
        for attempt in range(1, RELEVANCE_MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=RELEVANCE_SYSTEM_PROMPT,
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=list[RelevanceResult],
                        max_output_tokens=16000,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )

                # try the SDK parsed response first
                try:
                    parsed_results = [r.model_dump() for r in response.parsed]
                except Exception as parse_err:
                    log.warning(
                        f"  Batch {batch_number} response.parsed failed ({str(parse_err)[:80]}), "
                        f"falling back to manual JSON parsing"
                    )
                    # fallback: parse json ourselves
                    result = response.text.strip()
                    if result.startswith("```"):
                        result = result.split("```")[1]
                        if result.startswith("json"):
                            result = result[4:]
                    # Fix malformed \uXXXX escapes
                    result = re.sub(r'\\u(?![0-9a-fA-F]{4})[0-9a-fA-F]{0,3}', '', result)
                    parsed_results = json.loads(result.strip())

                # sanity check count
                if len(parsed_results) != len(articles_batch):
                    log.warning(
                        f"  Batch {batch_number}: expected {len(articles_batch)} results, "
                        f"got {len(parsed_results)}"
                    )

                return {"batch_number": batch_number, "results": parsed_results, "articles": articles_batch}

            except (json.JSONDecodeError, ValidationError) as e:
                if attempt < RELEVANCE_MAX_RETRIES:
                    wait = RELEVANCE_RETRY_WAIT * attempt
                    log.warning(
                        f"  Batch {batch_number} parse/validation error (attempt {attempt}/{RELEVANCE_MAX_RETRIES}): "
                        f"{str(e)[:80]}. Retrying in {wait}s..."
                    )
                    # strip non-ascii from titles on retry
                    if attempt >= 2:
                        _non_ascii = re.compile(r"[^\x00-\x7F]+")
                        articles_text = "\n\n".join([
                            f'**Article {i+1} (ID: "{i}"):**\nTitle: {_non_ascii.sub(" ", a["title"])}\nURL: {a["url"]}'
                            for i, a in enumerate(articles_batch)
                        ])
                        prompt = (
                            f"Analyze the relevance of these {len(articles_batch)} articles to Quebec City "
                            f"and the Capitale-Nationale region.\n\n{articles_text}\n\n"
                            f"Return a JSON array with analysis for each article, maintaining the article_id order."
                        )
                    time.sleep(wait)
                    continue
                log.error(f"  Batch {batch_number} failed after {RELEVANCE_MAX_RETRIES} attempts: {str(e)[:100]}")
                return {
                    "batch_number": batch_number,
                    "results": [{"article_id": str(i), "relevance_score": 0, "category": "ERROR", "reasoning": str(e)[:50], "key_indicators": []} for i in range(len(articles_batch))],
                    "articles": articles_batch,
                }

            except Exception as e:
                error_str = str(e)
                is_retryable = any(code in error_str for code in [
                    "429", "500", "503", "RESOURCE_EXHAUSTED"
                ])
                if is_retryable and attempt < RELEVANCE_MAX_RETRIES:
                    wait = RELEVANCE_RETRY_WAIT * attempt
                    log.warning(f"  Batch {batch_number} error ({error_str[:80]}), waiting {wait}s (attempt {attempt})")
                    time.sleep(wait)
                    continue
                log.error(f"  Batch {batch_number} failed: {error_str[:100]}")
                return {
                    "batch_number": batch_number,
                    "results": [{"article_id": str(i), "relevance_score": 0, "category": "ERROR", "reasoning": error_str[:50], "key_indicators": []} for i in range(len(articles_batch))],
                    "articles": articles_batch,
                }

    # chunk into batches
    all_batches = []
    for batch_start in range(0, len(articles), RELEVANCE_BATCH_SIZE):
        batch = articles[batch_start : batch_start + RELEVANCE_BATCH_SIZE]
        all_batches.append((len(all_batches), batch))

    analyzed = []
    for batch_idx, (batch_number, batch) in enumerate(all_batches):
        if batch_idx > 0:
            log.info(f"  Waiting {RELEVANCE_DELAY_BETWEEN_CALLS}s before batch {batch_number}...")
            time.sleep(RELEVANCE_DELAY_BETWEEN_CALLS)

        batch_result = _analyze_batch(batch, batch_number)
        results = batch_result["results"]

        # index results by article_id
        results_by_id = {}
        for r in results:
            aid = str(r.get("article_id", ""))
            results_by_id[aid] = r

        for i, article in enumerate(batch_result["articles"]):
            analysis = results_by_id.get(str(i))

            if analysis is None:
                log.warning(
                    f"  No matching result for article index {i}: {article['title'][:60]}. "
                    f"Gemini returned IDs: {list(results_by_id.keys())}"
                )
                analysis = {"relevance_score": 0, "category": "ERROR", "reasoning": "No matching result from Gemini", "key_indicators": []}

            analyzed.append({
                "relevance_score": analysis["relevance_score"],
                "category": analysis["category"],
                "source": article["source"],
                "title": article["title"],
                "url": article["url"],
                "date": article["date"],
                "reasoning": analysis["reasoning"],
                "key_indicators": ", ".join(analysis.get("key_indicators", [])),
                "is_google_redirect": article.get("is_google_redirect", False),
            })

    log.info(f"Relevance analysis completed in {time.time() - job_start:.1f}s")
    relevant = sum(1 for a in analyzed if a["relevance_score"] >= 50)
    log.info(f"Relevance complete: {relevant}/{len(analyzed)} articles scored ≥50")
    return analyzed



# -------------------------------------------------------
#  STAGE 3 - playwright scraping
# -------------------------------------------------------



# scraping config

SCRAPE_CONCURRENCY = 5
SCRAPE_BATCH_SIZE = 10
SCRAPE_DOMAIN_DELAY_MIN = 6.8
SCRAPE_DOMAIN_DELAY_MAX = 13.6
SCRAPE_TIMEOUT_MS = 30000
SCRAPE_RELEVANCE_THRESHOLD = 79.5

# google decode concurrency (too many => 429s)
GOOGLE_DECODE_CONCURRENCY = 2
GOOGLE_DECODE_DELAY_MIN = 2.5
GOOGLE_DECODE_DELAY_MAX = 4.5

BLOCKED_DOMAINS = ["mtltimes.ca", "themalaysianreserve.com", "keloland.com","torontolife.com","passportmagazine.com", "la1ere.franceinfo.fr","msn.com"]

FRENCH_DOMAIN_PATTERNS = [
    ".qc.ca", ".quebec", "radio-canada.ca", "lapresse.ca", "lesoleil.com",
    "journaldequebec.com", "ledevoir.com", "ici.radio-canada.ca", "tvanouvelles.ca",
]

SCRAPE_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
]

VIEWPORTS = [
    {"width": 1440, "height": 900},
    {"width": 1920, "height": 1080},
]

COOKIE_SELECTORS = [
    'button:has-text("Accept")', 'button:has-text("Accept All")', 'button:has-text("Agree")',
    'button:has-text("Allow")', 'button:has-text("OK")', 'button:has-text("Got it")',
    'button:has-text("Accepter")', 'button:has-text("J\'accepte")', 'button:has-text("Tout accepter")',
    "#onetrust-accept-btn-handler", "#accept-cookies", ".cookie-accept", '[aria-label="Accept cookies"]',
]


# google news url decoding

def _gnews_decode_sync(url: str) -> str | None:
    """Blocking gnewsdecoder call, meant for executor threads."""
    try:
        from googlenewsdecoder import gnewsdecoder

        decoded = gnewsdecoder(url, interval=3)
        log.debug(f"  gnewsdecoder raw response for {url[:60]}: {decoded}")

        if not isinstance(decoded, dict):
            log.warning(f"  gnewsdecoder unexpected type {type(decoded).__name__}: {decoded}")
            return None

        if not decoded.get("status"):
            log.warning(f"  gnewsdecoder status=False for {url[:60]}: {decoded.get('message', decoded)}")
            return None

        real_url = decoded.get("decoded_url")
        if not real_url:
            log.warning(f"  gnewsdecoder no decoded_url for {url[:60]}: {decoded}")
            return None

        if "news.google.com" in real_url:
            log.warning(f"  gnewsdecoder returned Google URL back: {real_url[:80]}")
            return None

        return real_url

    except Exception as e:
        log.warning(f"  gnewsdecoder exception for {url[:60]}: {type(e).__name__}: {e}")
    return None


async def _resolve_google_news_url(url: str, sem: asyncio.Semaphore) -> str | None:
    """Resolve a single google news redirect to the actual publisher url."""
    async with sem:
        # Throttle between requests to avoid 429s
        await asyncio.sleep(random.uniform(GOOGLE_DECODE_DELAY_MIN, GOOGLE_DECODE_DELAY_MAX))

        loop = asyncio.get_event_loop()
        real_url = await loop.run_in_executor(None, _gnews_decode_sync, url)
        if real_url:
            log.info(f"  ✓ Decoded (gnewsdecoder) → {real_url[:80]}")
            return real_url

    return None


async def _resolve_all_google_urls(urls: List[str]) -> Dict[str, str]:
    """Batch-resolve all google news urls. Returns {google_url: real_url}."""
    google_urls = [u for u in urls if "news.google.com" in u]
    if not google_urls:
        return {}

    log.info(f"  Pre-resolving {len(google_urls)} Google News URLs "
             f"(concurrency={GOOGLE_DECODE_CONCURRENCY})")

    sem = asyncio.Semaphore(GOOGLE_DECODE_CONCURRENCY)
    tasks = [_resolve_google_news_url(u, sem) for u in google_urls]
    results = await asyncio.gather(*tasks)

    resolved = {}
    success = 0
    for original, real in zip(google_urls, results):
        if real:
            resolved[original] = real
            success += 1

    log.info(f"  Google decode complete: {success}/{len(google_urls)} resolved")
    return resolved


# domain helpers

def _is_blocked_domain(url: str) -> bool:
    domain = urlparse(url).netloc.lower()
    return any(b in domain for b in BLOCKED_DOMAINS)


def _is_french_domain(url: str) -> bool:
    domain = urlparse(url).netloc.lower()
    return any(p in domain for p in FRENCH_DOMAIN_PATTERNS)


# stealth headers for chromium

def _build_stealth_headers(ua: str) -> dict:
    """Generates matching sec-ch-ua headers for a given User-Agent."""
    if "Chrome" not in ua:
        return {}

    match = re.search(r"Chrome/(\d+)", ua)
    version = match.group(1) if match else "122"

    if "Windows" in ua:
        platform = '"Windows"'
    elif "Mac OS X" in ua:
        platform = '"macOS"'
    elif "Linux" in ua:
        platform = '"Linux"'
    else:
        platform = '"Unknown"'

    if "Edg/" in ua:
        sec_ch_ua = f'"Chromium";v="{version}", "Not(A:Brand";v="24", "Microsoft Edge";v="{version}"'
    else:
        sec_ch_ua = f'"Chromium";v="{version}", "Not(A:Brand";v="24", "Google Chrome";v="{version}"'

    return {
        "sec-ch-ua": sec_ch_ua,
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": platform,
    }


# playwright scraping

async def _block_heavy_resources(route):
    if route.request.resource_type in ["image", "media"]:
        await route.abort()
    else:
        await route.continue_()


async def _handle_cookie_consent(page):
    try:
        pattern = re.compile(r"(accept|agree|allow|ok|accepter|autoriser)", re.IGNORECASE)
        btn = page.get_by_role("button", name=pattern).first
        if await btn.count() > 0 and await btn.is_visible():
            await btn.click(timeout=1000)
            await asyncio.sleep(2)
            return True
        for selector in COOKIE_SELECTORS:
            if await page.is_visible(selector, timeout=500):
                await page.click(selector)
                await asyncio.sleep(2)
                return True
    except Exception:
        pass
    return False


async def _resolve_google_via_playwright(url: str, browser, sem) -> str | None:
    """Last resort: use playwright to follow the JS redirect."""
    context = None
    page = None
    try:
        async with sem:
            ua = random.choice(SCRAPE_USER_AGENTS)
            context = await browser.new_context(
                user_agent=ua,
                viewport=random.choice(VIEWPORTS),
                extra_http_headers=_build_stealth_headers(ua),
            )
            page = await context.new_page()

            await page.goto(url, wait_until="networkidle", timeout=SCRAPE_TIMEOUT_MS)
            try:
                await page.wait_for_url(
                    lambda u: "news.google.com" not in u,
                    timeout=15000,
                )
                real_url = page.url
                if real_url and "news.google.com" not in real_url:
                    log.info(f"  ✓ Playwright redirect → {real_url[:80]}")
                    return real_url
            except Exception:
                log.debug(f"  ✗ Playwright redirect didn't complete for {url[:60]}")
    except Exception as e:
        log.debug(f"  ✗ Playwright redirect error for {url[:60]}: {e}")
    finally:
        try:
            if page:
                await page.close()
            if context:
                await context.close()
        except Exception:
            pass
    return None


async def _scrape_url(sem, url, browser, domain_locks, domain_last_request,
                      google_resolved: Dict[str, str]):
    """Scrape one url. Google redirects should already be resolved at this point."""
    if _is_blocked_domain(url):
        return {"url": url, "content": None, "status": "skipped", "resolved_url": url}

    original_url = url

    # swap in the pre-resolved url if we have one
    if "news.google.com" in url:
        resolved = google_resolved.get(url)
        if resolved:
            url = resolved
        else:
            log.info(f"  ℹ No decoded URL available, trying Playwright redirect: {url[:80]}")
            playwright_resolved = await _resolve_google_via_playwright(url, browser, sem)
            if playwright_resolved:
                url = playwright_resolved
            else:
                return {
                    "url": original_url, "content": None,
                    "status": "redirect_failed", "resolved_url": url,
                }

    # resolved domain might also be blocked
    if url != original_url and _is_blocked_domain(url):
        return {"url": original_url, "content": None, "status": "skipped", "resolved_url": url}

    domain = urlparse(url).netloc
    context = None
    page = None

    try:
        async with sem:
            async with domain_locks[domain]:
                last_time = domain_last_request.get(domain, 0)
                elapsed = time.time() - last_time
                delay = random.uniform(SCRAPE_DOMAIN_DELAY_MIN, SCRAPE_DOMAIN_DELAY_MAX)
                if elapsed < delay:
                    await asyncio.sleep(delay - elapsed + random.uniform(0.5, 1.5))
                domain_last_request[domain] = time.time()

                locale = "fr-CA" if _is_french_domain(url) else "en-US"

                ua = random.choice(SCRAPE_USER_AGENTS)

                context = await browser.new_context(
                    user_agent=ua,
                    viewport=random.choice(VIEWPORTS),
                    locale=locale,
                    extra_http_headers=_build_stealth_headers(ua),
                )
                page = await context.new_page()
                await page.route("**/*", _block_heavy_resources)

                try:
                    response = await page.goto(url, wait_until="domcontentloaded",
                                               timeout=SCRAPE_TIMEOUT_MS)
                    if response and response.status in [403, 429]:
                        return {
                            "url": original_url, "content": None,
                            "status": f"error: {response.status}", "resolved_url": url,
                        }
                except PlaywrightTimeout:
                    return {
                        "url": original_url, "content": None,
                        "status": "timeout", "resolved_url": url,
                    }


            resolved_url = page.url


            await _handle_cookie_consent(page)
            await asyncio.sleep(0.5)

            html = await page.content()
            await page.close()
            await context.close()
            page = None
            context = None

            extracted = trafilatura_extract(html, include_tables=True, include_comments=False)
            if extracted and len(extracted) > 100:
                return {
                    "url": original_url, "content": extracted,
                    "status": "success", "resolved_url": resolved_url,
                }
            else:
                return {
                    "url": original_url, "content": extracted,
                    "status": "partial", "resolved_url": resolved_url,
                }

    except Exception as e:
        return {
            "url": original_url, "content": None,
            "status": f"error: {str(e)[:50]}", "resolved_url": url,
        }
    finally:
        try:
            if page:
                await page.close()
            if context:
                await context.close()
        except Exception:
            pass


# scraper orchestrator

async def _run_scraper(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[df["relevance_score"] > SCRAPE_RELEVANCE_THRESHOLD].copy()
    urls = list(dict.fromkeys(filtered["url"].tolist()))
    random.shuffle(urls)

    log.info(f"Scraping {len(urls)} URLs (relevance > {SCRAPE_RELEVANCE_THRESHOLD}) "
             f"in batches of {SCRAPE_BATCH_SIZE}")

    # resolve google news redirects first
    google_resolved = await _resolve_all_google_urls(urls)

    # now scrape everything
    domain_locks = defaultdict(asyncio.Lock)
    domain_last_request = {}
    all_results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--headless=new",
            ],
        )
        try:
            sem = asyncio.Semaphore(SCRAPE_CONCURRENCY)
            for i in range(0, len(urls), SCRAPE_BATCH_SIZE):
                batch = urls[i: i + SCRAPE_BATCH_SIZE]
                log.info(f"  Batch {i // SCRAPE_BATCH_SIZE + 1} ({len(batch)} URLs)")
                tasks = [
                    _scrape_url(sem, url, browser, domain_locks,
                                domain_last_request, google_resolved)
                    for url in batch
                ]
                batch_results = await asyncio.gather(*tasks)
                all_results.extend(batch_results)
                await asyncio.sleep(2)
        finally:
            await browser.close()

    result_dict = {r["url"]: r for r in all_results}
    filtered["scraped_content"] = filtered["url"].map(
        lambda u: result_dict.get(u, {}).get("content")
    )
    filtered["scrape_status"] = filtered["url"].map(
        lambda u: result_dict.get(u, {}).get("status")
    )

    # swap in the resolved urls
    filtered["url"] = filtered["url"].map(
        lambda u: result_dict.get(u, {}).get("resolved_url", u)
    )

    success = sum(1 for r in all_results if r["status"] == "success")
    log.info(f"Scraping complete: {success}/{len(all_results)} successful")
    return filtered


def stage_3_scrape_articles(analyzed: List[Dict]) -> pd.DataFrame:
    """Scrape the actual article text for anything scored high enough."""
    log.info("--- stage 3: scraping articles ---")

    df = pd.DataFrame(analyzed)
    if df.empty:
        return df

    return asyncio.run(_run_scraper(df))

# -------------------------------------------------------
#  STAGE 4 - sentiment
# -------------------------------------------------------

SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
SENTIMENT_BATCH_SIZE = 16


def _split_into_phrases(text, min_len=20, max_len=350):
    if not isinstance(text, str):
        return []
    raw = re.split(r"(?<=[\.\!\?])\s+", text)
    sentences = [s.strip() for s in raw if len(s.strip()) >= min_len]
    phrases = []
    for s in sentences:
        if len(s) <= max_len:
            phrases.append(s)
        else:
            for i in range(0, len(s), max_len):
                chunk = s[i : i + max_len].strip()
                if len(chunk) >= min_len:
                    phrases.append(chunk)
    return phrases


def stage_4_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Run cardiffnlp sentiment on scraped articles."""
    log.info("--- stage 4: sentiment ---")

    if df.empty:
        df["sentiment"] = 0.0
        return df

    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL, local_files_only=True)
    sentiment_pipe = hf_pipeline(
        "sentiment-analysis", model=model, tokenizer=tokenizer,
        device=TORCH_DEVICE, truncation=True, max_length=512, padding=True,
    )

    df_success = df[df["scrape_status"] == "success"].copy()
    log.info(f"Running sentiment on {len(df_success)} successfully scraped articles")

    scores = []
    for _, row in df_success.iterrows():
        phrases = _split_into_phrases(row["scraped_content"])
        if not phrases:
            scores.append(0.0)
            continue
        pos = neg = 0
        for i in range(0, len(phrases), SENTIMENT_BATCH_SIZE):
            batch = phrases[i : i + SENTIMENT_BATCH_SIZE]
            results = sentiment_pipe(batch)
            for res in results:
                label = res["label"].lower()
                if "positive" in label:
                    pos += 1
                elif "negative" in label:
                    neg += 1
        non_neu = pos + neg
        scores.append(round((pos - neg) / non_neu, 2) if non_neu > 0 else 0.0)

    df_success["sentiment"] = scores
    df["sentiment"] = 0.0
    df.loc[df_success.index, "sentiment"] = df_success["sentiment"]

    log.info("Sentiment analysis complete")
    return df


# -------------------------------------------------------
#  STAGE 5 - clustering
# -------------------------------------------------------

def stage_5_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """Group similar articles together using cosine sim on embeddings."""
    log.info("--- stage 5: clustering ---")

    df["cluster_id"] = 0
    if "scraped_content" not in df.columns:
        return df

    valid_indices = df[df["scraped_content"].notna() & (df["scraped_content"].str.strip() != "")].index.tolist()
    if not valid_indices:
        log.warning("No valid content for clustering")
        return df

    contents = df.loc[valid_indices, "scraped_content"].tolist()
    log.info(f"Clustering {len(contents)} articles")

    model = SentenceTransformer("BAAI/bge-m3", local_files_only=True)
    embeddings = model.encode(contents, show_progress_bar=True, batch_size=8)

    # free up vram
    del model
    gc.collect()
    torch.cuda.empty_cache()

    sim_matrix = cosine_similarity(embeddings)

    adjacency = defaultdict(set)
    for i in range(len(sim_matrix)):
        for j in range(i + 1, len(sim_matrix)):
            if sim_matrix[i][j] >= 0.85:
                adjacency[i].add(j)
                adjacency[j].add(i)

    visited = set()
    cluster_mapping = {}
    cluster_num = 1

    def dfs(node, cid):
        visited.add(node)
        cluster_mapping[node] = cid
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                dfs(neighbor, cid)

    for i in range(len(valid_indices)):
        if i not in visited and i in adjacency:
            dfs(i, cluster_num)
            cluster_num += 1

    for local_idx, original_idx in enumerate(valid_indices):
        if local_idx in cluster_mapping:
            df.at[original_idx, "cluster_id"] = cluster_mapping[local_idx]

    log.info(f"Found {cluster_num - 1} clusters, {(df['cluster_id'] > 0).sum()} articles clustered")
    return df


# -------------------------------------------------------
#  STAGE 6 - IPTC topics
# -------------------------------------------------------

FRENCH_LABEL_MAP = {
    "arts, culture, entertainment and media": "Arts, culture, divertissement et médias",
    "conflict, war and peace": "Conflit, guerre et paix",
    "crime, law and justice": "Criminalité, droit et justice",
    "disaster, accident and emergency incident": "Désastres et accidents",
    "economy, business and finance": "Économie et finances",
    "education": "Éducation",
    "environment": "Environnement",
    "health": "Santé",
    "human interest": "Intérêt humain",
    "labour": "Travail",
    "lifestyle and leisure": "Style de vie et loisirs",
    "politics": "Politique",
    "religion": "Religion et croyance",
    "science and technology": "Science et technologie",
    "society": "Société",
    "sport": "Sport",
    "weather": "Météo",
}


def stage_6_iptc(df: pd.DataFrame) -> pd.DataFrame:
    """Tag each article with an IPTC news topic."""
    log.info("--- stage 6: IPTC classification ---")

    df["iptc_topic"] = None
    df["iptc_topic_fr"] = None

    valid_mask = (
        df["scraped_content"].notna() & (df["scraped_content"].str.strip() != "")
        & df["title"].notna() & (df["title"].str.strip() != "")
    )
    valid_indices = df[valid_mask].index.tolist()

    if not valid_indices:
        return df

    classifier = hf_pipeline(
        "text-classification",
        model="classla/multilingual-IPTC-news-topic-classifier",
        device=TORCH_DEVICE, max_length=512, truncation=True,
        model_kwargs={"local_files_only": True},
    )

    texts = [f"{str(df.at[idx, 'title']).strip()}. {str(df.at[idx, 'scraped_content']).strip()[:400]}" for idx in valid_indices]

    all_predictions = []
    for i in range(0, len(texts), 8):
        batch = texts[i : i + 8]
        all_predictions.extend(classifier(batch))

    for idx, pred in zip(valid_indices, all_predictions):
        df.at[idx, "iptc_topic"] = pred["label"]
        df.at[idx, "iptc_topic_fr"] = FRENCH_LABEL_MAP.get(pred["label"], pred["label"])

    log.info(f"IPTC classification complete: {len(valid_indices)} articles")
    return df


# -------------------------------------------------------
#  STAGE 7 - gliclass multi-label
# -------------------------------------------------------

def stage_7_gliclass(df: pd.DataFrame) -> pd.DataFrame:
    """Multi-label: is it about an attraction, hotel, or restaurant?"""
    log.info("--- stage 7: gliclass ---")

    from gliclass import GLiClassModel, ZeroShotClassificationPipeline as GliPipeline

    df["Attrait"] = 0.0
    df["Hebergement"] = 0.0
    df["Restaurant"] = 0.0

    valid_mask = (
        df["scraped_content"].notna() & (df["scraped_content"].str.strip() != "")
        & df["title"].notna() & (df["title"].str.strip() != "")
    )
    valid_indices = df[valid_mask].index.tolist()

    if not valid_indices:
        return df

    log.info("Loading GLiClass model...")
    gli_model = GLiClassModel.from_pretrained("knowledgator/gliclass-x-base", local_files_only=True)
    gli_tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-x-base", local_files_only=True, add_prefix_space=True)

    gli_pipeline = GliPipeline(gli_model, gli_tokenizer, classification_type="multi-label", device=TORCH_DEVICE_STR)

    labels = ["tourism event/attraction", "hotel/lodging", "restaurant"]

    def classify_long_text(text, threshold=0, max_tokens=400):
        tokens = gli_tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return gli_pipeline(text, labels, threshold=threshold)[0]
        chunks = [gli_tokenizer.decode(tokens[i : i + max_tokens]) for i in range(0, len(tokens), max_tokens - 50)]
        label_scores = {l: 0.0 for l in labels}
        for chunk in chunks:
            for result in gli_pipeline(chunk, labels, threshold=0)[0]:
                label_scores[result["label"]] = max(label_scores[result["label"]], result["score"])
        return sorted(
            [{"label": l, "score": round(s, 2)} for l, s in label_scores.items() if s >= threshold],
            key=lambda x: x["score"], reverse=True,
        )

    log.info(f"Classifying {len(valid_indices)} articles...")
    for i, idx in enumerate(valid_indices, 1):
        combined = f"{str(df.at[idx, 'title']).strip()}. {str(df.at[idx, 'scraped_content']).strip()}"
        results = classify_long_text(combined, threshold=0)
        for r in results:
            score = round(r["score"], 2)
            if r["label"] == "tourism event/attraction":
                df.at[idx, "Attrait"] = score
            elif r["label"] == "hotel/lodging":
                df.at[idx, "Hebergement"] = score
            elif r["label"] == "restaurant":
                df.at[idx, "Restaurant"] = score
        if i % 10 == 0:
            log.info(f"  {i}/{len(valid_indices)} articles classified")

    log.info("GLiClass classification complete")
    return df


# -------------------------------------------------------
#  STAGE 8 - gemini summarization + perception
# -------------------------------------------------------

SUMMARY_SYSTEM_PROMPT = """You are the Senior Strategic Reputation Analyst for Destination Québec. Your mandate is to monitor, analyze, and categorize media narratives that impact Quebec City and its surrounding regions as a tourism, cultural, and economic destination.

Your analysis feeds directly into the executive dashboard of tourism stakeholders, municipal leaders, and economic development officers. Precision is non-negotiable. When summarization stick "ONLY" to whats included in text , capture main idea and do not infer ideas from your own knowledge.

---

### I. THE FLAG LOGIC (DETERMINE THIS FIRST)

**Default: Flag = 0 (Relevant). Keep the article unless you are highly confident it has ZERO connection to Quebec City tourism, economy, or reputation.**

**Flag 0 — KEEP the article if ANY of these apply:**
- Directly mentions Quebec City, its boroughs, landmarks, or surrounding MRCs (see Section II).
- Provincial Quebec news that could affect Quebec City (e.g., tourism budgets, transport policy, provincial economic shifts, cultural funding).
- National Canadian news with a tourism or travel angle that could impact Quebec City (e.g., VIA Rail expansions, Air Canada route changes, federal heritage funding, Canda funding tourism organizations in quebec).
- Events, festivals, restaurants, hotels, attractions, or businesses in the Quebec City region.
- safety, infrastructure, or political news specific to Quebec City or its surrounding area with can directly or indirectly impact quebec city tourism.
- Any article where a reasonable tourism professional would say "this could affect visitor perception of Quebec City."

**Flag 1 — DISCARD the article ONLY if it meets ANY of these criteria:**
- Purely international news (foreign wars,  foreign elections, foreign markets)
- General weather forecasts not affectng specifc "touristic" events/infrastructures (e.g : if article speaks about a severe storm disrupting transport/electricity with no mention of touristic place/event , disregard it. If article speaks about a storm or weather conditions forcing a ski resort or a winter carnaval outdoor activity , don't disregard )
- Provincial health sector operations (hospital wait times, staff negotiations)
- Minor local incidents: small house fires, minor traffic accidents, petty theft, domestic disputes, local high school sports
- General Canada-wide news unrelated to travel or tourism and with no impact on quebec city directly or indirectly

**When in doubt, Flag = 0.** It is better to keep a borderline article than to discard one that matters.

---

### II. GEOGRAPHIC JURISDICTION ("quebec" field)

Set "quebec": "oui" if the article directly addresses or significantly impacts any of these zones:

**1. Agglomération de Québec (The Core):**
Neighbourhoods: Old Quebec (Vieux-Québec), Petit-Champlain, Saint-Roch, Saint-Sauveur, Limoilou, Sainte-Foy, Sillery, Cap-Rouge, Beauport, Charlesbourg, Val-Bélair, La Haute-Saint-Charles, L'Ancienne-Lorette, Saint-Augustin-de-Desmaures, Wendake.
Landmarks: Château Frontenac, Parliament Building, Plains of Abraham, Le Grand Théâtre, Centre Vidéotron, Université Laval, Jean-Lesage Airport (YQB).

**2. MRC de La Jacques-Cartier:**
Municipalities: Stoneham-et-Tewkesbury, Lac-Beauport, Sainte-Brigitte-de-Laval, Shannon, Saint-Gabriel-de-Valcartier.
Assets: Ski resorts (Stoneham, Le Relais), Village Vacances Valcartier, Jacques-Cartier National Park.

**3. MRC de La Côte-de-Beaupré:**
Municipalities: Beaupré, Sainte-Anne-de-Beaupré, Château-Richer, L'Ange-Gardien, Saint-Ferréol-les-Neiges.
Assets: Mont-Sainte-Anne, Basilica of Sainte-Anne-de-Beaupré, Canyon Sainte-Anne, Montmorency Falls.

**4. MRC de L'Île-d'Orléans:**
Municipalities: Sainte-Pétronille, Saint-Laurent, Sainte-Famille, Saint-François, Saint-Jean, Saint-Pierre.
Assets: Vineyards, cideries, heritage agriculture.

**5. MRC de Portneuf:**
Municipalities: Pont-Rouge, Donnacona, Saint-Raymond, Deschambault-Grondines, Neuville, Portneuf.
Assets: Vallée Bras-du-Nord, Chemin du Roy heritage route.

Set "quebec": "non" if the article says "Quebec" but refers only to the provincial government or province-wide policy with no specific mention of the capital city's infrastructure, local economy, or tourism.

---

### III. PERCEPTION ANALYSIS

Evaluate how this article affects a potential visitor's desire to visit or an investor's desire to build in Quebec City.

**POSITIVE:** Awards/rankings, new flights to YQB, hotel openings, cultural event success, high festival attendance, infrastructure improvements, positive media coverage, economic vitality, new conferences.

**NEGATIVE:** Safety concerns in tourist zones, service failures at YQB, labor shortage impacts on hospitality, negative media portrayal, political scandals at City Hall, crumbling infrastructure, "tourist trap" criticism.

**NEUTRE:** Routine administrative news, general statistics where Quebec City is just one data point, factual reporting with no clear positive or negative framing.

---

### IV. OUTPUT INSTRUCTIONS
1. **First:** Determine Flag (0 or 1). When in doubt, Flag = 0.
2. **Then:** Determine quebec (oui/non).
3. **Then:** Determine perception (positive/negative/neutre).
4. **Finally:** Write a 1-2 sentence summary in the language of the article.

Output a single JSON object:
{"summary": "...", "quebec": "oui/non", "perception": "positive/negative/neutre", "Flag": "0/1"}"""

SUMMARY_DELAY_BETWEEN_CALLS = 2


class ArticleAnalysis(BaseModel):
    summary: str = Field(..., min_length=1, max_length=500)
    quebec: str = Field(...)
    perception: str = Field(...)
    Flag: Union[str, int] = Field(...)

    @field_validator("summary")
    @classmethod
    def summary_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Summary cannot be empty")
        return v.strip()

    @field_validator("quebec")
    @classmethod
    def normalize_quebec(cls, v):
        v = v.lower().strip()
        if v not in ["oui", "non"]:
            raise ValueError(f"quebec must be 'oui' or 'non', got: {v}")
        return v

    @field_validator("perception")
    @classmethod
    def normalize_perception(cls, v):
        v = v.lower().strip()
        if v in ["negative", "négative", "negatif", "négatif"]:
            return "negative"
        elif v in ["neutral", "neutralle", "neutre"]:
            return "neutre"
        elif v in ["positive", "positif"]:
            return "positive"
        raise ValueError(f"Invalid perception: {v}")

    @field_validator("Flag")
    @classmethod
    def normalize_flag(cls, v):
        v_str = str(v).strip()
        if v_str in ["0", "1"]:
            return v_str
        raise ValueError(f"Flag must be '0' or '1', got: {v}")


def stage_8_summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Have gemini write a short summary + flag/perception for each article."""
    log.info("--- stage 8: summarization ---")

    client = genai.Client(vertexai=True, project=PROJECT_ID, location=GEMINI_LOCATION)

    df["summary"] = None
    df["quebec"] = None
    df["perception"] = None
    df["Flag"] = None

    valid_mask = (
        df["scraped_content"].notna() & (df["scraped_content"].str.strip() != "")
        & df["title"].notna() & (df["title"].str.strip() != "")
    )
    valid_indices = df[valid_mask].index.tolist()

    def _analyze_article(content, title, max_retries=5):
        prompt = f"Title: {title}\n\nArticle:\n{content}\n\nAnalyze this article from Destination Québec's perspective."

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SUMMARY_SYSTEM_PROMPT,
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=ArticleAnalysis,
                        max_output_tokens=768,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )

                # try the SDK parsed response first
                try:
                    parsed = response.parsed
                    if isinstance(parsed, list) and len(parsed) > 0:
                        parsed = parsed[0]
                    if isinstance(parsed, ArticleAnalysis):
                        return parsed
                    return ArticleAnalysis(**parsed)
                except Exception as parse_err:
                    log.warning(f"response.parsed failed for '{title[:60]}' ({str(parse_err)[:80]}), falling back to manual parsing")

                # sdk parse failed, do it manually
                if not response.text:
                    log.warning(f"Empty response on attempt {attempt + 1} for: {title[:80]}")
                    time.sleep(10 * attempt + 10)
                    continue

                raw = response.text.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                raw = re.sub(r'\\u(?![0-9a-fA-F]{4})[0-9a-fA-F]{0,3}', '', raw)
                parsed = json.loads(raw)
                if isinstance(parsed, list) and len(parsed) > 0:
                    parsed = parsed[0]
                return ArticleAnalysis(**parsed)

            except json.JSONDecodeError as e:
                log.warning(f"JSON parse error attempt {attempt + 1} for '{title[:80]}': {e}")
                # Truncate content on retry to simplify task for model
                if len(content) > 5000:
                    content = content[:5000] + "..."
                    prompt = f"Title: {title}\n\nArticle:\n{content}\n\nAnalyze this article from Destination Québec's perspective."
                time.sleep(10 * attempt + 10)

            except ValidationError as e:
                log.warning(f"Validation error attempt {attempt + 1} for '{title[:80]}': {e}")
                time.sleep(10 * attempt + 10)

            except Exception as e:
                log.warning(f"API error attempt {attempt + 1} for '{title[:80]}': {e}")
                time.sleep(10 * attempt + 10)

        log.error(f"All {max_retries} retries failed for: {title[:80]}")
        return None

    log.info(f"Analyzing {len(valid_indices)} articles sequentially...")
    success = fail = 0

    for i, idx in enumerate(valid_indices):
        if i > 0:
            time.sleep(SUMMARY_DELAY_BETWEEN_CALLS)

        title = str(df.at[idx, "title"]).strip()
        content = str(df.at[idx, "scraped_content"]).strip()
        if len(content) > 9000:
            content = content[:9000] + "..."

        result = _analyze_article(content, title)

        if result:
            df.at[idx, "summary"] = result.summary
            df.at[idx, "quebec"] = result.quebec
            df.at[idx, "perception"] = result.perception
            df.at[idx, "Flag"] = result.Flag
            success += 1
        else:
            fail += 1

    # log what we're dropping
    original_count = len(df)
    flagged_mask = df["Flag"] == "1"
    flagged_df = df[flagged_mask]

    if len(flagged_df) > 0:
        log.info(f"--- FLAG=1 ARTICLES BEING DROPPED ({len(flagged_df)} total) ---")
        for _, row in flagged_df.iterrows():
            title = str(row.get("title", "N/A"))[:120]
            summary = str(row.get("summary", "N/A"))[:150]
            source = str(row.get("source", "N/A"))[:50]
            log.info(f"  DROPPED | {source} | {title}")
            log.info(f"           Summary: {summary}")
        log.info(f"--- END FLAG=1 LIST ---")

    # log failures too
    failed_mask = df["Flag"].isna()
    if failed_mask.sum() > 0:
        log.warning(f"--- FAILED ARTICLES (no analysis, kept in dataset as nulls): {failed_mask.sum()} ---")
        for _, row in df[failed_mask].iterrows():
            log.warning(f"  FAILED | {str(row.get('title', 'N/A'))[:120]}")

    # drop flagged international stuff
    df = df[df["Flag"] != "1"].copy()
    dropped = original_count - len(df)
    log.info(f"Summarization complete: {success} ok, {fail} failed, {dropped} international dropped")

    return df

# -------------------------------------------------------
#  STAGE 9 - bigquery export
# -------------------------------------------------------

def stage_9_save_to_bigquery(df: pd.DataFrame):
    """Push the final dataframe into BQ."""
    log.info("--- stage 9: saving to bigquery ---")

    required_cols = [
        "sentiment", "cluster_id", "iptc_topic", "iptc_topic_fr",
        "Attrait", "Hebergement", "Restaurant", "quebec", "perception", "summary",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        log.error(f"Missing required columns: {missing}")
        raise KeyError(f"Missing columns: {missing}")

    bq_client = bigquery.Client(project=PROJECT_ID)

    dataset_ref = f"{PROJECT_ID}.{DATASET_ID}"
    try:
        bq_client.get_dataset(dataset_ref)
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = LOCATION
        bq_client.create_dataset(dataset)
        log.info(f"Created dataset {DATASET_ID}")

    schema = [
        bigquery.SchemaField("url", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("title", "STRING"),
        bigquery.SchemaField("source", "STRING"),
        bigquery.SchemaField("date", "STRING"),
        bigquery.SchemaField("relevance_score", "INTEGER"),
        bigquery.SchemaField("category", "STRING"),
        bigquery.SchemaField("reasoning", "STRING"),
        bigquery.SchemaField("key_indicators", "STRING"),
        bigquery.SchemaField("scraped_content", "STRING"),
        bigquery.SchemaField("scrape_status", "STRING"),
        bigquery.SchemaField("sentiment", "FLOAT64"),
        bigquery.SchemaField("cluster_id", "INTEGER"),
        bigquery.SchemaField("iptc_topic", "STRING"),
        bigquery.SchemaField("iptc_topic_fr", "STRING"),
        bigquery.SchemaField("Attrait", "FLOAT64"),
        bigquery.SchemaField("Hebergement", "FLOAT64"),
        bigquery.SchemaField("Restaurant", "FLOAT64"),
        bigquery.SchemaField("quebec", "STRING"),
        bigquery.SchemaField("perception", "STRING"),
        bigquery.SchemaField("summary", "STRING"),
        bigquery.SchemaField("processed_timestamp", "TIMESTAMP"),
    ]

    df["processed_timestamp"] = pd.Timestamp.now()

    columns_to_save = [
        "url", "title", "source", "date", "relevance_score", "category",
        "reasoning", "key_indicators", "scraped_content", "scrape_status",
        "sentiment", "cluster_id", "iptc_topic", "iptc_topic_fr",
        "Attrait", "Hebergement", "Restaurant", "quebec", "perception",
        "summary", "processed_timestamp",
    ]
    df_to_save = df[columns_to_save].copy()

    # dedup by url
    before_dedup = len(df_to_save)
    df_to_save = df_to_save.drop_duplicates(subset=["url"], keep="first")
    if before_dedup > len(df_to_save):
        log.info(f"  Dedup: removed {before_dedup - len(df_to_save)} duplicate URLs, {len(df_to_save)} rows remain")

    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )
    job = bq_client.load_table_from_dataframe(df_to_save, table_ref, job_config=job_config)
    job.result()

    log.info(f"Loaded {len(df_to_save)} rows to {table_ref}")


# -------------------------------------------------------
#  main
# -------------------------------------------------------

def main():
    pipeline_start = time.time()
    log.info(f"pipeline started at {datetime.now().isoformat()}")

    vertexai.init(project=PROJECT_ID, location=LOCATION)

    articles = stage_1_scrape_sitemaps()
    if not articles:
        log.warning("No articles found. Pipeline ending.")
        return

    analyzed = stage_2_relevance_analysis(articles)
    if not analyzed:
        log.warning("No analyzed articles. Pipeline ending.")
        return

    df = stage_3_scrape_articles(analyzed)
    if df.empty:
        log.warning("No scraped articles. Pipeline ending.")
        return

    df = stage_4_sentiment(df)
    df = stage_5_clustering(df)
    df = stage_6_iptc(df)
    df = stage_7_gliclass(df)
    df = stage_8_summarize(df)
    stage_9_save_to_bigquery(df)

    elapsed = time.time() - pipeline_start
    log.info(f"done -- took {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
