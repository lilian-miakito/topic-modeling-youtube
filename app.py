import os
import json
import argparse
import threading
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from flask import Flask, render_template, request, jsonify, send_file
import yt_dlp

# Number of parallel workers for comment extraction
MAX_WORKERS = 5

app = Flask(__name__)
app.config['OUTPUT_DIR'] = 'data'

# Créer le dossier data s'il n'existe pas
os.makedirs(app.config['OUTPUT_DIR'], exist_ok=True)

# Global state for extraction tracking
extraction_state = {
    'active': False,
    'stop_requested': False,
    'current_channel': None,
    'current_video': None,
    'videos_total': 0,
    'videos_completed': 0,
    'comments_extracted': 0,
    'filename': None
}
extraction_lock = threading.Lock()

# Queue for multi-channel extraction
extraction_queue = Queue()
queue_list = []  # For display purposes
queue_lock = threading.Lock()


def get_already_downloaded_video_ids():
    """Get all video IDs that have already been downloaded from existing JSON files."""
    downloaded_ids = set()
    output_dir = app.config['OUTPUT_DIR']
    
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(output_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for video in data.get('videos', []):
                            if video.get('video_id'):
                                downloaded_ids.add(video['video_id'])
                except Exception:
                    pass
    
    return downloaded_ids


def get_channel_videos(channel_url):
    """Récupère la liste de toutes les vidéos d'une chaîne."""
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'force_generic_extractor': False,
    }

    # Construire l'URL de la chaîne si ce n'est pas déjà une URL complète
    if not channel_url.startswith('http'):
        if channel_url.startswith('@'):
            channel_url = f'https://www.youtube.com/{channel_url}/videos'
        else:
            channel_url = f'https://www.youtube.com/channel/{channel_url}/videos'
    elif '/videos' not in channel_url:
        channel_url = channel_url.rstrip('/') + '/videos'

    videos = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(channel_url, download=False)

        if result and 'entries' in result:
            for entry in result['entries']:
                if entry:
                    videos.append({
                        'id': entry.get('id'),
                        'title': entry.get('title'),
                        'url': f"https://www.youtube.com/watch?v={entry.get('id')}"
                    })

    return videos, result.get('channel', result.get('uploader', 'Unknown'))


def get_video_comments(video_url):
    """Fetch all comments from a video."""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'getcomments': True,
        'extract_flat': False,
        'extractor_args': {'youtube': {'comment_sort': ['top'], 'skip': ['dash', 'hls']}},
    }

    comments = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(video_url, download=False)

        if result and 'comments' in result:
            for comment in result['comments']:
                comments.append({
                    'author': comment.get('author'),
                    'author_id': comment.get('author_id'),
                    'text': comment.get('text'),
                    'likes': comment.get('like_count', 0),
                    'timestamp': comment.get('timestamp'),
                    'parent': comment.get('parent', 'root'),
                    'is_reply': comment.get('parent') != 'root'
                })

    return comments


def scrape_video_comments(video):
    """Helper function to scrape comments from a single video (for parallel execution)."""
    try:
        comments = get_video_comments(video['url'])
        return {
            'video_id': video['id'],
            'title': video['title'],
            'url': video['url'],
            'comment_count': len(comments),
            'comments': comments,
            'error': None
        }
    except Exception as e:
        return {
            'video_id': video['id'],
            'title': video['title'],
            'url': video['url'],
            'comment_count': 0,
            'comments': [],
            'error': str(e)
        }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/channel-info', methods=['POST'])
def get_channel_info():
    """Endpoint pour récupérer les infos de la chaîne."""
    data = request.json
    channel_input = data.get('channel', '')

    if not channel_input:
        return jsonify({'error': 'Veuillez fournir un nom ou ID de chaîne'}), 400

    try:
        videos, channel_name = get_channel_videos(channel_input)
        return jsonify({
            'channel_name': channel_name,
            'video_count': len(videos),
            'videos': videos
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def save_progress(filepath, data, lock):
    """Save current progress to JSON file (thread-safe)."""
    with lock:
        # Update stats before saving
        data['total_comments'] = sum(v.get('comment_count', 0) for v in data['videos'])
        data['total_videos'] = len(data['videos'])
        data['videos_completed'] = len(data['videos'])
        data['last_updated'] = datetime.now().isoformat()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def update_extraction_state(**kwargs):
    """Update global extraction state."""
    with extraction_lock:
        extraction_state.update(kwargs)


def reset_extraction_state():
    """Reset extraction state."""
    with extraction_lock:
        extraction_state.update({
            'active': False,
            'stop_requested': False,
            'current_channel': None,
            'current_video': None,
            'videos_total': 0,
            'videos_completed': 0,
            'comments_extracted': 0,
            'filename': None
        })


def do_extraction(channel_input, limit=None, skip_existing=False):
    """Worker function for extraction (runs in background thread)."""
    try:
        update_extraction_state(active=True, stop_requested=False)
        
        videos, channel_name = get_channel_videos(channel_input)
        total_available = len(videos)
        skipped_count = 0

        update_extraction_state(current_channel=channel_name)

        # Filter out already downloaded videos if skip_existing is enabled
        if skip_existing:
            already_downloaded = get_already_downloaded_video_ids()
            original_count = len(videos)
            videos = [v for v in videos if v['id'] not in already_downloaded]
            skipped_count = original_count - len(videos)
            print(f"Skipping {skipped_count} already downloaded videos")

        # Apply limit if specified
        if limit and limit > 0:
            videos = videos[:limit]

        # Prepare filename - just channel name, no timestamp
        safe_channel_name = "".join(c for c in channel_name if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"{safe_channel_name}.json"
        filepath = os.path.join(app.config['OUTPUT_DIR'], filename)
        
        update_extraction_state(filename=filename)
        
        # If file already exists, load existing videos to merge
        existing_videos = []
        existing_video_ids = set()
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    existing_videos = existing_data.get('videos', [])
                    existing_video_ids = {v.get('video_id') for v in existing_videos}
                    print(f"Found existing file with {len(existing_videos)} videos, will merge new data")
            except Exception:
                pass

        # Filter out videos that are already in the existing file
        videos = [v for v in videos if v['id'] not in existing_video_ids]
        
        if len(videos) == 0:
            print("All videos already extracted, nothing new to do")
            reset_extraction_state()
            return {
                'success': True,
                'channel_name': channel_name,
                'total_videos': len(existing_videos),
                'message': 'All videos already extracted'
            }

        update_extraction_state(
            videos_total=len(videos),
            videos_completed=0,
            comments_extracted=sum(v.get('comment_count', 0) for v in existing_videos)
        )

        all_comments = {
            'channel_name': channel_name,
            'last_updated': datetime.now().isoformat(),
            'total_videos': len(existing_videos) + len(videos),
            'videos_completed': len(existing_videos),
            'total_comments': sum(v.get('comment_count', 0) for v in existing_videos),
            'videos': existing_videos.copy()
        }

        # Lock for thread-safe file writing
        file_lock = threading.Lock()

        # Save initial file with existing data
        save_progress(filepath, all_comments, file_lock)

        print(f"Starting parallel extraction for {len(videos)} NEW videos with {MAX_WORKERS} workers...")
        print(f"Progress will be saved to: {filename}")

        # Parallel extraction using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_video = {executor.submit(scrape_video_comments, video): video for video in videos}

            completed = 0
            for future in as_completed(future_to_video):
                # Check if stop was requested
                if extraction_state['stop_requested']:
                    print("Stop requested, cancelling remaining tasks...")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                
                completed += 1
                result = future.result()
                video_title = result['title'][:50] if result['title'] else 'Unknown'

                if result['error']:
                    print(f"[{completed}/{len(videos)}] Error: {video_title} - {result['error']}")
                else:
                    print(f"[{completed}/{len(videos)}] Done: {video_title} ({result['comment_count']} comments)")

                # Add result and save progress
                with file_lock:
                    all_comments['videos'].append(result)
                save_progress(filepath, all_comments, file_lock)
                
                # Update global state
                total_comments = sum(v.get('comment_count', 0) for v in all_comments['videos'])
                update_extraction_state(
                    videos_completed=completed,
                    current_video=video_title,
                    comments_extracted=total_comments
                )

        # Final stats
        total_comments = sum(v.get('comment_count', 0) for v in all_comments['videos'])
        was_stopped = extraction_state['stop_requested']

        if was_stopped:
            print(f"Extraction stopped! {total_comments} comments saved to {filename}")
        else:
            print(f"Extraction complete! {total_comments} comments saved to {filename}")

        reset_extraction_state()
        
        return {
            'success': True,
            'channel_name': channel_name,
            'total_videos': len(all_comments['videos']),
            'total_comments': total_comments,
            'filename': filename,
            'stopped': was_stopped
        }
    except Exception as e:
        print(f"Extraction error: {e}")
        reset_extraction_state()
        return {'error': str(e)}


def queue_worker():
    """Background worker to process extraction queue."""
    while True:
        job = extraction_queue.get()
        if job is None:
            break
        
        job_id, channel_input, limit, skip_existing = job
        
        # Update queue status
        with queue_lock:
            for item in queue_list:
                if item['id'] == job_id:
                    item['status'] = 'running'
                    break
        
        # Do the extraction
        result = do_extraction(channel_input, limit, skip_existing)
        
        # Update queue status
        with queue_lock:
            for item in queue_list:
                if item['id'] == job_id:
                    item['status'] = 'completed' if result.get('success') else 'error'
                    item['result'] = result
                    break
        
        extraction_queue.task_done()


# Start the queue worker thread
queue_thread = threading.Thread(target=queue_worker, daemon=True)
queue_thread.start()


@app.route('/api/scrape-comments', methods=['POST'])
def scrape_comments():
    """Endpoint to queue a channel extraction."""
    data = request.json
    channel_input = data.get('channel', '')
    limit = data.get('limit')
    skip_existing = data.get('skip_existing', False)

    if not channel_input:
        return jsonify({'error': 'Please provide a channel name or ID'}), 400

    # Create job
    job_id = str(uuid.uuid4())[:8]
    job = (job_id, channel_input, limit, skip_existing)
    
    # Add to queue
    with queue_lock:
        queue_list.append({
            'id': job_id,
            'channel': channel_input,
            'status': 'queued',
            'result': None
        })
    
    extraction_queue.put(job)
    
    return jsonify({
        'success': True,
        'job_id': job_id,
        'message': 'Extraction queued',
        'queue_position': extraction_queue.qsize()
    })


@app.route('/api/extraction-status')
def get_extraction_status():
    """Get current extraction status for real-time progress."""
    with extraction_lock:
        status = extraction_state.copy()
    
    with queue_lock:
        status['queue'] = queue_list.copy()
    
    return jsonify(status)


@app.route('/api/stop-extraction', methods=['POST'])
def stop_extraction():
    """Stop the current extraction."""
    with extraction_lock:
        if extraction_state['active']:
            extraction_state['stop_requested'] = True
            return jsonify({'success': True, 'message': 'Stop requested'})
        else:
            return jsonify({'success': False, 'message': 'No extraction in progress'})


@app.route('/api/clear-queue', methods=['POST'])
def clear_queue():
    """Clear completed/errored items from queue."""
    with queue_lock:
        queue_list[:] = [item for item in queue_list if item['status'] in ('queued', 'running')]
    return jsonify({'success': True})


@app.route('/api/download/<filename>')
def download_file(filename):
    """Télécharger le fichier JSON généré."""
    filepath = os.path.join(app.config['OUTPUT_DIR'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'Fichier non trouvé'}), 404


@app.route('/api/files')
def list_files():
    """Lister tous les fichiers JSON disponibles."""
    files = []
    output_dir = app.config['OUTPUT_DIR']

    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(output_dir, filename)
                size = os.path.getsize(filepath)
                # Formater la taille
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f} MB"

                files.append({
                    'name': filename,
                    'size': size_str,
                    'path': filepath
                })

    # Trier par date de modification (plus récent en premier)
    files.sort(key=lambda x: os.path.getmtime(x['path']), reverse=True)

    return jsonify({'files': files})


@app.route('/api/files-stats')
def list_files_with_stats():
    """Lister tous les fichiers JSON avec leurs statistiques."""
    files = []
    output_dir = app.config['OUTPUT_DIR']
    total_videos = 0
    total_comments = 0
    channels = set()

    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(output_dir, filename)
                size = os.path.getsize(filepath)

                # Formater la taille
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f} MB"

                file_info = {
                    'name': filename,
                    'size': size_str,
                    'path': filepath
                }

                # Lire les métadonnées du fichier JSON
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        file_info['channel_name'] = data.get('channel_name', '')
                        file_info['video_count'] = data.get('total_videos', 0)
                        file_info['comment_count'] = data.get('total_comments', 0)
                        file_info['last_updated'] = data.get('last_updated', data.get('scraped_at', ''))

                        # Accumuler les stats globales
                        if file_info['channel_name']:
                            channels.add(file_info['channel_name'])
                        total_videos += file_info['video_count']
                        total_comments += file_info['comment_count']
                except Exception:
                    pass

                files.append(file_info)

    # Trier par date de mise à jour (plus récent en premier)
    files.sort(key=lambda x: x.get('last_updated', ''), reverse=True)

    return jsonify({
        'files': files,
        'total_channels': len(channels),
        'total_videos': total_videos,
        'total_comments': total_comments
    })


@app.route('/api/file-detail/<filename>')
def get_file_detail(filename):
    """Récupérer le contenu détaillé d'un fichier JSON."""
    filepath = os.path.join(app.config['OUTPUT_DIR'], filename)

    if not os.path.exists(filepath):
        return jsonify({'error': 'Fichier non trouvé'}), 404

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YouTube Comments Scraper')
    parser.add_argument('--port', type=int, default=4242, help='Port to run the server on (default: 4242)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on (default: 127.0.0.1)')
    args = parser.parse_args()

    app.run(debug=True, host=args.host, port=args.port)
