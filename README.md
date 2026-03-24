# AI-news-pipeline 
Ingesting and analyzing news with Deep Learning NLP techniques and LLM's in GCP

# Quebec City News Intelligence Pipeline

Automated media monitoring system for Destination Québec cité. Runs daily as a Google Cloud Run Job, collecting and analyzing French/English news coverage relevant to Quebec City's tourism economy and regional reputation. Results are stored in BigQuery.

## How It Works

The pipeline runs in 9 sequential stages:

1. **Collection** — Scrapes 20+ Canadian news sitemaps and RSS feeds, plus Google News via SerpAPI. Deduplicates by URL and title, keeping only the last 24 hours.

2. **Relevance Scoring** — Gemini scores each article 0–100 based on its relevance to Quebec City tourism using a detailed geographic rubric. Only articles scoring above 80 move forward.

3. **Scraping** — Playwright (headless Chromium) scrapes the full article text with stealth measures, cookie handling, and per-domain rate limiting. Google News redirects are resolved beforehand.

4. **Sentiment Analysis** — Each article gets a sentiment score from -1 to +1 using the cardiffnlp/twitter-xlm-roberta model.

5. **Clustering** — Similar articles are grouped together using cosine similarity on BAAI/bge-m3 embeddings.

6. **Topic Classification** — Articles are tagged with IPTC news topics in English and French.

7. **Multi-Label Classification** — GLiClass scores each article for relevance to three tourism verticals: attractions, lodging, and restaurants.

8. **Summarization** — Gemini writes a short summary, flags whether the article is specific to Quebec City, assesses perception (positive/negative/neutral), and removes irrelevant international content.

9. **Export** — Final dataset is pushed to BigQuery with schema enforcement and dedup.

## Configuration

All settings come from environment variables: GCP_PROJECT_ID, BQ_DATASET_ID, BQ_TABLE_ID, GCP_LOCATION, GEMINI_LOCATION, GEMINI_MODEL. Defaults point to the production project.

## Deployment

Containerized for Cloud Run Jobs. Requires a GCP service account with BigQuery and Vertex AI access, Playwright/Chromium dependencies, and pre-downloaded HuggingFace models baked into the image.
