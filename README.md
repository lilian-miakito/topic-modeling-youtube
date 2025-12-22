# YouTube Topic Modeling

> **Work in Progress** - This project is under active development. Some features may be incomplete or subject to change.

A Flask application to extract YouTube comments and perform topic modeling analysis.

## Features

### 1. Comment Extraction
- Search YouTube channels by handle (`@channelname`) or ID
- **Multi-channel support**: Extract multiple channels at once (comma-separated)
- **Parallel extraction** with configurable worker count (1 to 2x CPU cores)
- **Queue system**: Add multiple channels to a queue, processed sequentially
- **Real-time progress bar** with live updates
- **Stop button** to cancel extraction mid-process
- **Skip already downloaded videos** to resume interrupted extractions
- Progressive saving: each video saved individually (no data loss on interruption)

### 2. Data Structure
Each channel is saved in its own folder:
```
data/
  @ChannelName/
    info.json              # Channel metadata (subscribers, description, etc.)
    videos/
      <video_id>.json      # One file per video with comments
      <video_id>.json
      ...
```

### 3. Data Insights
- View all extracted channels
- Channel statistics (subscribers, videos, comments)
- Comments per video chart
- Comments timeline visualization
- Video list sorted by engagement

### 4. Topic Modeling (Coming Soon)
Pipeline for topic modeling:
1. **Data Loading** - Select channel data
2. **Preprocessing** - Text cleaning (lowercase, stopwords, lemmatization)
3. **Vectorization** - Transform to numerical vectors
4. **Topic Modeling** - Available algorithms:
   - LDA (Latent Dirichlet Allocation)
   - NMF (Non-negative Matrix Factorization)
   - BERTopic
   - Top2Vec
5. **Dimensionality Reduction** - UMAP, t-SNE, PCA

## Installation

> Want to contribute? Fork the repository first, then follow the steps below.

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

Open http://localhost:4242

To use a different port:
```bash
python app.py --port 8080
```

### Multi-Channel Extraction

Enter multiple channels separated by commas:
```
@MrBeast, @Fireship, @TechWithTim
```

All channels will be added to the queue and processed one after another.

### Configurable Workers

Use the slider to adjust the number of parallel workers (1 to 2x your CPU cores).
More workers = faster extraction, but may hit YouTube rate limits.

## Project Structure

```
youtube-comments-scraper/
├── app.py              # Flask application
├── requirements.txt    # Python dependencies
├── README.md           # Documentation
├── templates/
│   └── index.html      # Web interface
└── data/               # Extracted data (per channel)
    └── @ChannelName/
        ├── info.json
        └── videos/
            └── *.json
```

## Extracted Data Format

### info.json (Channel Metadata)
```json
{
  "channel_name": "ChannelName",
  "channel_id": "UCxxxxxx",
  "channel_url": "https://www.youtube.com/channel/UCxxxxxx",
  "description": "Channel description...",
  "subscriber_count": 1500000,
  "total_videos": 150,
  "videos_extracted": 150,
  "total_comments": 25000,
  "last_updated": "2025-12-22T15:30:00"
}
```

### videos/<video_id>.json
```json
{
  "video_id": "abc123",
  "title": "Video Title",
  "url": "https://www.youtube.com/watch?v=abc123",
  "comment_count": 500,
  "comments": [
    {
      "author": "User1",
      "author_id": "UC...",
      "text": "Great video!",
      "likes": 42,
      "timestamp": 1703257800,
      "parent": "root",
      "is_reply": false
    }
  ]
}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/channel-info` | POST | Get channel info |
| `/api/scrape-comments` | POST | Queue channel(s) extraction |
| `/api/extraction-status` | GET | Get real-time extraction progress |
| `/api/stop-extraction` | POST | Stop current extraction |
| `/api/clear-queue` | POST | Clear completed queue items |
| `/api/system-info` | GET | Get CPU/worker info |
| `/api/files-stats` | GET | List channels with statistics |
| `/api/file-detail/<folder>` | GET | Get channel details |

## Tech Stack

- **Backend**: Flask, yt-dlp, ThreadPoolExecutor
- **Frontend**: HTML/CSS/JavaScript, Plotly.js
- **Topic Modeling** (planned): scikit-learn, BERTopic, Gensim
- **NLP** (planned): spaCy, NLTK
- **Dimensionality Reduction** (planned): UMAP, t-SNE

## Roadmap

- [x] YouTube comment extraction
- [x] Parallel extraction (configurable workers)
- [x] Multi-channel queue system
- [x] Real-time progress bar
- [x] Stop/cancel extraction
- [x] Skip already downloaded videos
- [x] Per-video JSON storage
- [x] Channel metadata (subscribers, description)
- [x] Web interface with tabs
- [x] Data insights dashboard
- [ ] NLP preprocessing pipeline
- [ ] LDA/NMF implementation
- [ ] BERTopic integration
- [ ] Interactive visualization
- [ ] Results export

## License

MIT
