# Task 9: Web Scraper & Scheduler Integration

## Objective

These modules (`src/web_scraper.py` and `src/scheduler.py`) are ALREADY IMPLEMENTED.
Your task is to integrate them into the existing system.

## What These Modules Do

### web_scraper.py

- Automatically downloads CPI PDFs from PBS website
- Parses PBS page to find PDF links
- Downloads files with retry logic
- Tracks download status
- Identifies missing files

### scheduler.py

- Runs automatic monthly updates
- Triggers web scraper on schedule
- Runs preprocessing after new download
- Tracks success/failure status
- Provides status for Streamlit dashboard

## Integration Tasks

### 1. Update requirements.txt

Add these dependencies:

```
beautifulsoup4>=4.12.0
lxml>=4.9.0
schedule>=1.2.0
```

### 2. Update config.py

Add PBS URL configuration if not present:

```python
# Web Scraping Configuration
PBS_CPI_BASE_URL = "https://www.pbs.gov.pk/publications/consumer-price-index-cpi"
DOWNLOAD_CHECK_INTERVAL = 3600  # seconds
```

### 3. Update pipeline.py

Add option to download missing PDFs before analysis:

```python
def run_full_analysis(config, download_missing=False):
    if download_missing:
        from src.web_scraper import download_pdfs_for_date_range
        download_pdfs_for_date_range(config['start_date'], config['end_date'])

    # ... rest of pipeline
```

### 4. Add CLI Commands

Create standalone scripts:

**download_pdfs.py:**

```python
from src.web_scraper import download_pdfs_for_date_range
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start', default='2023-03-01')
parser.add_argument('--end', default='2025-11-01')
args = parser.parse_args()

download_pdfs_for_date_range(args.start, args.end)
```

**start_scheduler.py:**

```python
from src.scheduler import run_scheduler_loop
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--time', default='09:00')
args = parser.parse_args()

run_scheduler_loop(args.time)
```

### 5. Update Streamlit Dashboard

Add new page for Scheduler Status showing:

- Current scheduler health
- Last successful/failed update
- Next scheduled run
- Manual trigger button
- Recent download failures
- Option to download missing files

## Testing

```bash
# Test web scraper
python -m src.web_scraper --start 2024-01-01 --end 2024-03-01

# Test scheduler
python -m src.scheduler --test

# Check status
python -m src.scheduler --status
```

## Success Criteria

✅ PDFs download automatically
✅ Scheduler runs monthly
✅ Status tracked in JSON/CSV
✅ Failures visible in dashboard
✅ Manual override available

## Files Created

- `src/web_scraper.py` ✅ (already done)
- `src/scheduler.py` ✅ (already done)
- Need to create: `download_pdfs.py`, `start_scheduler.py`
- Need to update: Streamlit app with scheduler page
