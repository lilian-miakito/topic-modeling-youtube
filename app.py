import os
import json
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, render_template, request, jsonify, send_file
import yt_dlp

# Number of parallel workers for comment extraction (default to CPU count)
DEFAULT_WORKERS = os.cpu_count() or 4
MAX_WORKERS = DEFAULT_WORKERS * 2  # Allow up to 2x CPU count

app = Flask(__name__)
app.config['OUTPUT_DIR'] = 'data'

# Créer le dossier data s'il n'existe pas
os.makedirs(app.config['OUTPUT_DIR'], exist_ok=True)


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


def scrape_video_comments(video, channel_dir):
    """Helper function to scrape comments from a single video and save to individual file."""
    video_id = video['id']
    video_file = os.path.join(channel_dir, f"{video_id}.json")

    try:
        comments = get_video_comments(video['url'])
        video_data = {
            'video_id': video_id,
            'title': video['title'],
            'url': video['url'],
            'scraped_at': datetime.now().isoformat(),
            'comment_count': len(comments),
            'comments': comments
        }

        # Save immediately to individual file
        with open(video_file, 'w', encoding='utf-8') as f:
            json.dump(video_data, f, ensure_ascii=False, indent=2)

        return {
            'video_id': video_id,
            'title': video['title'],
            'url': video['url'],
            'comment_count': len(comments),
            'error': None
        }
    except Exception as e:
        return {
            'video_id': video_id,
            'title': video['title'],
            'url': video['url'],
            'comment_count': 0,
            'error': str(e)
        }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/system-info')
def system_info():
    """Get system info for UI configuration."""
    return jsonify({
        'cpu_count': os.cpu_count() or 4,
        'default_workers': DEFAULT_WORKERS,
        'max_workers': MAX_WORKERS
    })


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


@app.route('/api/scrape-comments', methods=['POST'])
def scrape_comments():
    """Endpoint to scrape all comments from a channel (parallelized, saves per video)."""
    data = request.json
    channel_input = data.get('channel', '')
    num_workers = min(data.get('workers', DEFAULT_WORKERS), MAX_WORKERS)

    if not channel_input:
        return jsonify({'error': 'Please provide a channel name or ID'}), 400

    try:
        videos, channel_name = get_channel_videos(channel_input)

        # Create channel directory
        safe_channel_name = "".join(c for c in channel_name if c.isalnum() or c in (' ', '-', '_')).strip()
        channel_dir = os.path.join(app.config['OUTPUT_DIR'], safe_channel_name)
        os.makedirs(channel_dir, exist_ok=True)

        print(f"Starting parallel extraction for {len(videos)} videos with {num_workers} workers...")
        print(f"Saving to: {channel_dir}/")

        video_results = []

        # Parallel extraction using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks with channel_dir
            future_to_video = {
                executor.submit(scrape_video_comments, video, channel_dir): video
                for video in videos
            }

            completed = 0
            for future in as_completed(future_to_video):
                completed += 1
                result = future.result()
                video_title = result['title'][:50] if result['title'] else 'Unknown'

                if result['error']:
                    print(f"[{completed}/{len(videos)}] Error: {video_title} - {result['error']}")
                else:
                    print(f"[{completed}/{len(videos)}] Saved: {video_title} ({result['comment_count']} comments)")

                video_results.append(result)

        # Calculate total comments
        total_comments = sum(v.get('comment_count', 0) for v in video_results)

        # Save/update channel metadata
        metadata = {
            'channel_name': channel_name,
            'last_scraped_at': datetime.now().isoformat(),
            'total_videos': len(videos),
            'total_comments': total_comments,
            'videos': [
                {
                    'video_id': v['video_id'],
                    'title': v['title'],
                    'url': v['url'],
                    'comment_count': v['comment_count'],
                    'error': v.get('error')
                }
                for v in video_results
            ]
        }

        metadata_file = os.path.join(channel_dir, '_metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"Extraction complete! {total_comments} comments saved to {channel_dir}/")

        return jsonify({
            'success': True,
            'channel_name': channel_name,
            'channel_dir': safe_channel_name,
            'total_videos': len(videos),
            'total_comments': total_comments
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
    """List all channel directories with their statistics."""
    channels_list = []
    output_dir = app.config['OUTPUT_DIR']
    total_videos = 0
    total_comments = 0

    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)

            # Check for channel directories (new structure)
            if os.path.isdir(item_path):
                metadata_file = os.path.join(item_path, '_metadata.json')
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)

                        # Calculate folder size
                        folder_size = sum(
                            os.path.getsize(os.path.join(item_path, f))
                            for f in os.listdir(item_path)
                            if os.path.isfile(os.path.join(item_path, f))
                        )

                        if folder_size < 1024:
                            size_str = f"{folder_size} B"
                        elif folder_size < 1024 * 1024:
                            size_str = f"{folder_size / 1024:.1f} KB"
                        else:
                            size_str = f"{folder_size / (1024 * 1024):.1f} MB"

                        channel_info = {
                            'name': item,
                            'channel_name': metadata.get('channel_name', item),
                            'video_count': metadata.get('total_videos', 0),
                            'comment_count': metadata.get('total_comments', 0),
                            'scraped_at': metadata.get('last_scraped_at', ''),
                            'size': size_str,
                            'is_folder': True
                        }

                        total_videos += channel_info['video_count']
                        total_comments += channel_info['comment_count']
                        channels_list.append(channel_info)
                    except Exception:
                        pass

            # Also check for legacy single JSON files
            elif item.endswith('.json'):
                try:
                    with open(item_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    size = os.path.getsize(item_path)
                    if size < 1024:
                        size_str = f"{size} B"
                    elif size < 1024 * 1024:
                        size_str = f"{size / 1024:.1f} KB"
                    else:
                        size_str = f"{size / (1024 * 1024):.1f} MB"

                    channel_info = {
                        'name': item,
                        'channel_name': data.get('channel_name', item.replace('.json', '')),
                        'video_count': data.get('total_videos', 0),
                        'comment_count': data.get('total_comments', 0),
                        'scraped_at': data.get('scraped_at', ''),
                        'size': size_str,
                        'is_folder': False
                    }

                    total_videos += channel_info['video_count']
                    total_comments += channel_info['comment_count']
                    channels_list.append(channel_info)
                except Exception:
                    pass

    # Sort by scrape date (most recent first)
    channels_list.sort(key=lambda x: x.get('scraped_at', ''), reverse=True)

    return jsonify({
        'files': channels_list,
        'total_channels': len(channels_list),
        'total_videos': total_videos,
        'total_comments': total_comments
    })


@app.route('/api/file-detail/<path:name>')
def get_file_detail(name):
    """Get detailed content of a channel folder or legacy JSON file."""
    item_path = os.path.join(app.config['OUTPUT_DIR'], name)

    if not os.path.exists(item_path):
        return jsonify({'error': 'Not found'}), 404

    try:
        # New folder structure
        if os.path.isdir(item_path):
            metadata_file = os.path.join(item_path, '_metadata.json')
            if not os.path.exists(metadata_file):
                return jsonify({'error': 'No metadata found'}), 404

            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Load all video files and merge comments
            videos_with_comments = []
            for video_info in metadata.get('videos', []):
                video_id = video_info['video_id']
                video_file = os.path.join(item_path, f"{video_id}.json")

                if os.path.exists(video_file):
                    with open(video_file, 'r', encoding='utf-8') as f:
                        video_data = json.load(f)
                    videos_with_comments.append(video_data)
                else:
                    # Video file missing, use metadata info
                    videos_with_comments.append({
                        'video_id': video_id,
                        'title': video_info.get('title', ''),
                        'url': video_info.get('url', ''),
                        'comment_count': 0,
                        'comments': []
                    })

            return jsonify({
                'channel_name': metadata.get('channel_name', name),
                'scraped_at': metadata.get('last_scraped_at', ''),
                'total_videos': metadata.get('total_videos', 0),
                'total_comments': metadata.get('total_comments', 0),
                'videos': videos_with_comments
            })

        # Legacy single JSON file
        else:
            with open(item_path, 'r', encoding='utf-8') as f:
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
