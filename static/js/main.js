/* =============================================================================
   YouTube Topic Modeling - Main JavaScript
   ============================================================================= */

// =============================================================================
// Tab Navigation
// =============================================================================

function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
    });

    document.getElementById('tab-' + tabName).classList.add('active');
    
    // Find and activate the clicked nav item
    const navItem = document.querySelector(`.nav-item[onclick*="${tabName}"]`);
    if (navItem) navItem.classList.add('active');

    if (tabName === 'data') {
        loadDataFiles();
    } else if (tabName === 'modeling') {
        loadModelingData();
    } else if (tabName === 'bootstrap') {
        loadBootstrapTopics();
    }
}

// =============================================================================
// Extraction
// =============================================================================

let channelData = null;
let channelsList = [];
let pollingInterval = null;

async function getChannelInfo() {
    const channelInput = document.getElementById('channelInput').value.trim();
    if (!channelInput) {
        alert('Please enter a channel name or ID');
        return;
    }

    // Parse multiple channels
    const channels = channelInput.split(',').map(c => c.trim()).filter(c => c);

    const searchBtn = document.getElementById('searchBtn');
    searchBtn.disabled = true;
    searchBtn.innerHTML = '<span class="spinner"></span> Searching...';

    try {
        if (channels.length === 1) {
            // Single channel - show detailed info
            const response = await fetch('/api/channel-info', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ channel: channels[0] })
            });

            const data = await response.json();

            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }

            channelData = data;
            channelsList = [channels[0]];
            document.getElementById('channelLabel').textContent = 'Channel';
            document.getElementById('channelName').textContent = data.channel_name;
            document.getElementById('videoLabel').textContent = 'Videos';
            document.getElementById('videoCount').textContent = data.video_count + ' videos';
        } else {
            // Multiple channels - show summary
            channelsList = channels;
            channelData = { multi: true, count: channels.length };
            document.getElementById('channelLabel').textContent = 'Channels';
            document.getElementById('channelName').textContent = channels.length + ' channels selected';
            document.getElementById('videoLabel').textContent = 'List';
            document.getElementById('videoCount').textContent = channels.join(', ');
        }

        document.getElementById('channelInfo').style.display = 'block';
        document.getElementById('scrapeCard').style.display = 'block';
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        searchBtn.disabled = false;
        searchBtn.innerHTML = 'Search';
    }
}

async function addToQueue() {
    const channelInput = document.getElementById('channelInput').value.trim();
    if (!channelInput) {
        alert('Please search for a channel first');
        return;
    }

    // Get options
    const limitInput = document.getElementById('videoLimit').value;
    const limit = limitInput ? parseInt(limitInput) : null;
    const skipExisting = document.getElementById('skipExisting').checked;
    const workers = parseInt(document.getElementById('workersSlider').value);

    try {
        const response = await fetch('/api/scrape-comments', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                channel: channelInput,
                limit: limit,
                skip_existing: skipExisting,
                workers: workers
            })
        });

        const data = await response.json();

        if (data.error) {
            alert('Error: ' + data.error);
        } else {
            // Hide result box for new extractions
            document.getElementById('resultBox').style.display = 'none';
            // Start polling for progress
            startPolling();
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

function startPolling() {
    if (pollingInterval) return;
    
    pollingInterval = setInterval(updateProgress, 1000);
    updateProgress(); // Initial call
}

function stopPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
}

async function updateProgress() {
    try {
        const response = await fetch('/api/extraction-status');
        const status = await response.json();
        
        const progressContainer = document.getElementById('progressContainer');
        const stopBtn = document.getElementById('stopBtn');
        
        // Update queue display (filter out running item, it's shown in progress)
        const queueItems = (status.queue || []).filter(q => q.status !== 'running');
        updateQueueDisplay(queueItems);
        
        if (status.active) {
            progressContainer.style.display = 'block';
            stopBtn.style.display = 'inline-flex';
            
            const progress = status.videos_total > 0 
                ? (status.videos_completed / status.videos_total) * 100 
                : 0;
            
            document.getElementById('progressChannel').textContent = status.current_channel || 'Starting...';
            document.getElementById('progressPercent').textContent = Math.round(progress) + '%';
            document.getElementById('progressFill').style.width = progress + '%';
            document.getElementById('progressText').textContent = 
                `${status.videos_completed}/${status.videos_total} videos processed`;
            document.getElementById('progressDetail').textContent = 
                status.current_video 
                    ? `Current: ${status.current_video} | ${status.comments_extracted.toLocaleString()} comments`
                    : `${status.comments_extracted.toLocaleString()} comments extracted`;
        } else {
            progressContainer.style.display = 'none';
            stopBtn.style.display = 'none';
            
            // Check if there are completed items in queue
            const completedItems = (status.queue || []).filter(q => q.status === 'completed');
            if (completedItems.length > 0) {
                showCompletionResult(completedItems[completedItems.length - 1]);
            }
            
            // Check if queue has pending items
            const pendingItems = (status.queue || []).filter(q => q.status === 'queued');
            if (pendingItems.length === 0 && completedItems.length === 0) {
                stopPolling();
            }
        }
    } catch (error) {
        console.error('Polling error:', error);
    }
}

function showCompletionResult(item) {
    const resultBox = document.getElementById('resultBox');
    const result = item.result;
    
    if (!result) return;
    
    resultBox.style.display = 'block';
    resultBox.classList.remove('error');
    
    if (result.error) {
        resultBox.classList.add('error');
        document.getElementById('resultTitle').textContent = 'Error';
        document.getElementById('resultText').textContent = result.error;
        document.getElementById('downloadLink').style.display = 'none';
    } else {
        document.getElementById('resultTitle').textContent = 
            result.stopped ? 'Extraction Stopped' : 'Extraction Complete';
        document.getElementById('resultText').textContent = 
            `${result.channel_name}: ${result.total_videos} videos, ${result.total_comments?.toLocaleString() || 0} comments`;
        
        if (result.filename) {
            const downloadLink = document.getElementById('downloadLink');
            downloadLink.href = '/api/download/' + result.filename;
            downloadLink.style.display = 'inline-block';
        }
    }
}

function updateQueueDisplay(queue) {
    const queueList = document.getElementById('queueList');
    
    if (!queue || queue.length === 0) {
        queueList.innerHTML = '<span style="color: var(--text-muted);">No extractions in queue</span>';
        return;
    }
    
    queueList.innerHTML = queue.map(item => {
        let statusIcon = '';
        let statusColor = '';
        switch(item.status) {
            case 'queued': statusIcon = '⏳'; statusColor = 'var(--text-muted)'; break;
            case 'completed': statusIcon = '✅'; statusColor = 'var(--success)'; break;
            case 'error': statusIcon = '❌'; statusColor = 'var(--error)'; break;
        }
        return `<div style="padding: 8px 0; border-bottom: 1px solid var(--border-color); display: flex; align-items: center; gap: 12px;">
            <span>${statusIcon}</span>
            <span style="flex: 1;">${item.channel}</span>
            <span style="color: ${statusColor}; font-size: 12px; text-transform: uppercase;">${item.status}</span>
        </div>`;
    }).join('');
}

async function stopExtraction() {
    try {
        const response = await fetch('/api/stop-extraction', { method: 'POST' });
        const data = await response.json();
        if (data.success) {
            document.getElementById('progressText').textContent = 'Stopping...';
        }
    } catch (error) {
        console.error('Stop error:', error);
    }
}

async function clearQueue() {
    try {
        await fetch('/api/clear-queue', { method: 'POST' });
        document.getElementById('resultBox').style.display = 'none';
        updateProgress();
    } catch (error) {
        console.error('Clear queue error:', error);
    }
}

// =============================================================================
// Data Tab
// =============================================================================

let filesData = [];
let currentFileData = null;

async function loadDataFiles() {
    try {
        const response = await fetch('/api/files-stats');
        const data = await response.json();

        filesData = data.files || [];
        renderDataTable();
        updateGlobalStats(data);
    } catch (error) {
        console.error('Error loading files:', error);
        // Fallback to simple file list
        try {
            const response = await fetch('/api/files');
            const data = await response.json();
            filesData = data.files || [];
            renderDataTable();
        } catch (e) {
            console.error('Fallback also failed:', e);
        }
    }
}

function updateGlobalStats(data) {
    document.getElementById('statChannels').textContent = data.total_channels || 0;
    document.getElementById('statVideos').textContent = (data.total_videos || 0).toLocaleString();
    document.getElementById('statComments').textContent = (data.total_comments || 0).toLocaleString();
    document.getElementById('statFiles').textContent = data.files?.length || 0;
}

function renderDataTable() {
    const tbody = document.getElementById('dataTableBody');
    const emptyState = document.getElementById('dataEmptyState');
    const table = document.getElementById('dataTable');

    if (!filesData || filesData.length === 0) {
        table.style.display = 'none';
        emptyState.style.display = 'block';
        return;
    }

    table.style.display = 'table';
    emptyState.style.display = 'none';

    tbody.innerHTML = filesData.map(file => `
        <tr>
            <td>
                <div class="channel-name">
                    <div class="channel-avatar">${(file.channel_name || file.folder || '?').charAt(0).toUpperCase()}</div>
                    <div>
                        <div>${file.channel_name || file.folder}</div>
                        ${file.subscriber_count ? `<div style="font-size: 12px; color: var(--text-muted);">${formatSubscribers(file.subscriber_count)} subscribers</div>` : ''}
                    </div>
                </div>
            </td>
            <td>${file.video_count || '-'}</td>
            <td>${file.comment_count ? file.comment_count.toLocaleString() : '-'}</td>
            <td>${file.last_updated ? formatDate(file.last_updated) : '-'}</td>
            <td>${file.size || '-'}</td>
            <td>
                <button class="btn btn-secondary btn-sm" onclick="viewFileDetail('${file.folder}')">
                    Insights
                </button>
            </td>
        </tr>
    `).join('');
}

async function viewFileDetail(filename) {
    try {
        const response = await fetch(`/api/file-detail/${filename}`);
        const data = await response.json();

        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }

        currentFileData = data;
        showFileDetail(data);
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

function showFileDetail(data) {
    document.getElementById('dataListView').style.display = 'none';
    document.getElementById('dataDetailView').classList.add('active');

    // Update header
    document.getElementById('detailChannelName').textContent = data.channel_name || 'Unknown Channel';
    document.getElementById('detailDate').textContent = data.last_updated ? `Last updated: ${formatDate(data.last_updated)}` : '';

    // Update stats
    const totalComments = data.total_comments || 0;
    const totalVideos = data.total_videos || 0;
    const avgComments = totalVideos > 0 ? Math.round(totalComments / totalVideos) : 0;
    const totalReplies = data.videos?.reduce((sum, v) =>
        sum + (v.comments?.filter(c => c.is_reply).length || 0), 0) || 0;

    document.getElementById('detailVideos').textContent = totalVideos.toLocaleString();
    document.getElementById('detailComments').textContent = totalComments.toLocaleString();
    document.getElementById('detailAvg').textContent = avgComments.toLocaleString();
    document.getElementById('detailReplies').textContent = totalReplies.toLocaleString();

    // Render charts
    renderVideoChart(data.videos || []);
    renderTimelineChart(data.videos || []);

    // Render video list
    renderVideoList(data.videos || []);
}

function hideFileDetail() {
    document.getElementById('dataListView').style.display = 'block';
    document.getElementById('dataDetailView').classList.remove('active');
}

function renderVideoChart(videos) {
    const sortedVideos = [...videos]
        .sort((a, b) => (b.comment_count || 0) - (a.comment_count || 0))
        .slice(0, 20);

    const data = [{
        x: sortedVideos.map(v => v.comment_count || 0),
        y: sortedVideos.map(v => truncateText(v.title, 40)),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: '#3b82f6'
        }
    }];

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#888', size: 11 },
        margin: { l: 200, r: 20, t: 10, b: 40 },
        xaxis: {
            gridcolor: '#2a2a2a',
            title: 'Comments'
        },
        yaxis: {
            autorange: 'reversed'
        }
    };

    Plotly.newPlot('chartVideos', data, layout, { responsive: true, displayModeBar: false });
}

function renderTimelineChart(videos) {
    // Aggregate comments by date
    const commentsByDate = {};

    videos.forEach(video => {
        if (!video.comments) return;
        video.comments.forEach(comment => {
            if (!comment.timestamp) return;
            const date = new Date(comment.timestamp * 1000).toISOString().split('T')[0];
            commentsByDate[date] = (commentsByDate[date] || 0) + 1;
        });
    });

    const dates = Object.keys(commentsByDate).sort();
    const counts = dates.map(d => commentsByDate[d]);

    if (dates.length === 0) {
        document.getElementById('chartTimeline').innerHTML =
            '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #555;">No temporal data available</div>';
        return;
    }

    const data = [{
        x: dates,
        y: counts,
        type: 'scatter',
        mode: 'lines',
        fill: 'tozeroy',
        line: { color: '#3b82f6', width: 2 },
        fillcolor: 'rgba(59, 130, 246, 0.1)'
    }];

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#888', size: 11 },
        margin: { l: 50, r: 20, t: 10, b: 40 },
        xaxis: {
            gridcolor: '#2a2a2a',
            type: 'date'
        },
        yaxis: {
            gridcolor: '#2a2a2a',
            title: 'Comments'
        }
    };

    Plotly.newPlot('chartTimeline', data, layout, { responsive: true, displayModeBar: false });
}

function renderVideoList(videos) {
    const sortedVideos = [...videos].sort((a, b) => (b.comment_count || 0) - (a.comment_count || 0));

    document.getElementById('videoList').innerHTML = sortedVideos.map(video => `
        <div class="video-item">
            <a href="${video.url}" target="_blank" class="video-title" title="${video.title}">
                ${video.title}
            </a>
            <span class="video-comments">${(video.comment_count || 0).toLocaleString()} comments</span>
        </div>
    `).join('');
}

// =============================================================================
// Utilities
// =============================================================================

function formatDate(dateStr) {
    try {
        const date = new Date(dateStr);
        return date.toLocaleDateString('en-US', {
            month: 'short',
            day: '2-digit',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch {
        return dateStr;
    }
}

function formatSubscribers(count) {
    if (!count) return '';
    if (count >= 1000000) {
        return (count / 1000000).toFixed(1) + 'M';
    } else if (count >= 1000) {
        return (count / 1000).toFixed(1) + 'K';
    }
    return count.toString();
}

function truncateText(text, maxLength) {
    if (!text) return '';
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
}

// =============================================================================
// Modeling Tab
// =============================================================================

let modelingData = null;
let stopwordsData = null;
const TOPICS_INITIAL_LIMIT = 10;
let showAllTopics = false;

async function loadModelingData() {
    try {
        // Load dataset stats
        const statsRes = await fetch('/api/modeling/stats');
        if (statsRes.ok) {
            const stats = await statsRes.json();
            document.getElementById('modelingComments').textContent = 
                (stats.total_comments || 0).toLocaleString();
        }

        // Load topics
        const topicsRes = await fetch('/api/modeling/topics');
        if (topicsRes.ok) {
            modelingData = await topicsRes.json();
            renderTopics(modelingData);
            document.getElementById('modelingTopics').textContent = modelingData.num_topics || '-';
            document.getElementById('modelingSample').textContent = 
                (modelingData.sample_size || 0).toLocaleString();
            document.getElementById('modelingFilename').textContent = modelingData.filename || '';
            document.getElementById('modelingDate').textContent = 
                modelingData.generated_at ? formatDate(modelingData.generated_at) : '-';
            
            if (modelingData.params) {
                document.getElementById('paramsJson').textContent = 
                    JSON.stringify(modelingData.params, null, 2);
            }
        }

        // Load stopwords
        const swRes = await fetch('/api/modeling/stopwords');
        if (swRes.ok) {
            stopwordsData = await swRes.json();
            renderStopwords(stopwordsData);
            document.getElementById('modelingStopwords').textContent = 
                (stopwordsData.detected_stopwords?.length || 0).toLocaleString();
        }

        // Load visualizations
        const vizRes = await fetch('/api/modeling/visualizations');
        if (vizRes.ok) {
            const vizData = await vizRes.json();
            renderVisualizations(vizData.files || []);
        }

    } catch (error) {
        console.error('Error loading modeling data:', error);
    }
}

function renderTopics(data) {
    const container = document.getElementById('topicsList');
    const topics = data.topics || [];

    if (topics.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">📊</div>
                <div class="empty-title">No topics yet</div>
                <div class="empty-text">Run <code>python modeling/extract_topics.py</code></div>
            </div>`;
        return;
    }

    const displayTopics = showAllTopics ? topics : topics.slice(0, TOPICS_INITIAL_LIMIT);
    const hasMore = topics.length > TOPICS_INITIAL_LIMIT;

    container.innerHTML = displayTopics.map((topic, idx) => {
        const silhouette = topic.silhouette || 0;
        const isFourreTout = silhouette < 0.1;
        const topWords = topic.top_words_centroid_mmr || topic.top_words || [];
        const comments = topic.example_comments_centroid_mmr || topic.example_comments_original || topic.example_comments || [];
        
        // Silhouette color
        let silColor = 'var(--success)';
        if (silhouette < 0.1) silColor = 'var(--error)';
        else if (silhouette < 0.3) silColor = 'var(--warning)';
        
        return `
            <div style="border: 1px solid var(--border-color); border-radius: 8px; padding: 16px; margin-bottom: 12px; ${isFourreTout ? 'border-left: 3px solid var(--warning);' : ''}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span style="font-weight: 600; font-size: 16px;">Topic ${topic.id}</span>
                        <span class="badge" style="background: var(--bg-secondary);">${topic.count} comments</span>
                        ${topic.generated_name ? `<span style="color: var(--accent);">${topic.generated_name}</span>` : ''}
                        ${isFourreTout ? '<span style="color: var(--warning);">⚠️ fourre-tout?</span>' : ''}
                    </div>
                    <div style="display: flex; gap: 16px; font-size: 12px; color: var(--text-muted);">
                        <span>sil: <span style="color: ${silColor}; font-weight: 500;">${silhouette.toFixed(3)}</span></span>
                        <span>var: ${(topic.variance || 0).toFixed(3)}</span>
                    </div>
                </div>
                <div style="margin-bottom: 12px;">
                    ${topWords.slice(0, 8).map(w => `<span style="display: inline-block; background: var(--bg-secondary); padding: 4px 10px; border-radius: 4px; margin: 2px; font-size: 13px;">${w}</span>`).join('')}
                </div>
                <details style="margin-top: 8px;">
                    <summary style="cursor: pointer; color: var(--text-muted); font-size: 13px;">Example comments (${comments.length})</summary>
                    <div style="margin-top: 8px; padding-left: 12px; border-left: 2px solid var(--border-color);">
                        ${comments.slice(0, 3).map(c => `<p style="font-size: 13px; color: var(--text-secondary); margin: 8px 0; line-height: 1.5;">"${truncateText(c, 200)}"</p>`).join('')}
                    </div>
                </details>
            </div>
        `;
    }).join('');

    // Add "Show more" button if needed
    if (hasMore) {
        container.innerHTML += `
            <div style="text-align: center; margin-top: 16px;">
                <button class="btn btn-secondary" onclick="toggleShowAllTopics()">
                    ${showAllTopics ? '▲ Show less' : `▼ Show all ${topics.length} topics`}
                </button>
            </div>
        `;
    }
}

function toggleShowAllTopics() {
    showAllTopics = !showAllTopics;
    if (modelingData) {
        renderTopics(modelingData);
    }
}

function renderStopwords(data) {
    // NLTK count
    const nltkCount = Object.values(data.sources?.nltk?.by_language || {}).reduce((a, b) => a + b, 0);
    document.getElementById('stopwordsNltk').textContent = `${nltkCount} words (${Object.keys(data.sources?.nltk?.by_language || {}).join(', ')})`;
    
    // Entropy-detected
    const entropyWords = data.sources?.entropy?.words || [];
    if (entropyWords.length > 0) {
        document.getElementById('stopwordsEntropy').innerHTML = entropyWords.slice(0, 50).map(w => 
            `<span style="display: inline-block; background: var(--bg-secondary); padding: 2px 8px; border-radius: 4px; margin: 2px; font-size: 12px;">${w}</span>`
        ).join('') + (entropyWords.length > 50 ? `<span style="color: var(--text-muted);"> +${entropyWords.length - 50} more</span>` : '');
    }
}

function showModelingSubtab(subtab) {
    // Hide all subtab contents
    document.querySelectorAll('.modeling-subtab-content').forEach(el => {
        el.style.display = 'none';
    });
    // Remove active from all buttons
    document.querySelectorAll('.modeling-subtab').forEach(btn => {
        btn.classList.remove('active');
    });
    // Show selected subtab
    document.getElementById('modeling-subtab-' + subtab).style.display = 'block';
    document.querySelector(`.modeling-subtab[data-subtab="${subtab}"]`).classList.add('active');
}

function renderVisualizations(files) {
    const tabsContainer = document.getElementById('vizTabs');
    const frameContainer = document.getElementById('vizFrame');
    
    if (!files || files.length === 0) {
        tabsContainer.innerHTML = '<span style="color: var(--text-muted);">No visualizations. Run <code>python modeling/visualize_topics.py</code></span>';
        return;
    }

    // Group by type and get latest of each
    const byType = {};
    files.forEach(f => {
        if (!byType[f.type]) byType[f.type] = f;
    });

    const types = Object.keys(byType);
    const icons = {
        'galaxy': '🌌',
        'hierarchy': '🌳',
        'heatmap': '🔥',
        'barchart': '📊',
        'topics_distance': '📍'
    };

    tabsContainer.innerHTML = types.map(type => {
        const icon = icons[type] || '📄';
        return `<button class="btn btn-secondary btn-sm viz-tab" data-viz="${type}" onclick="loadVisualization('${byType[type].path}', '${type}')">
            ${icon} ${type}
        </button>`;
    }).join('');

    // Load first visualization by default
    if (types.length > 0) {
        loadVisualization(byType[types[0]].path, types[0]);
    }
}

function loadVisualization(path, type) {
    const frameContainer = document.getElementById('vizFrame');
    
    // Update active tab
    document.querySelectorAll('.viz-tab').forEach(btn => btn.classList.remove('active'));
    document.querySelector(`.viz-tab[data-viz="${type}"]`)?.classList.add('active');
    
    // Load in iframe
    frameContainer.innerHTML = `<iframe src="${path}" style="width: 100%; height: 600px; border: none; border-radius: 8px;"></iframe>`;
}

// =============================================================================
// Initialization
// =============================================================================

async function initWorkersSlider() {
    try {
        const response = await fetch('/api/system-info');
        const info = await response.json();

        const slider = document.getElementById('workersSlider');
        slider.max = info.max_workers;
        slider.value = info.default_workers;
        document.getElementById('workersValue').textContent = info.default_workers;
    } catch (error) {
        console.error('Failed to load system info:', error);
    }
}

document.addEventListener('DOMContentLoaded', async function() {
    // Initialize workers slider
    initWorkersSlider();

    // Enter key support
    document.getElementById('channelInput')?.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            getChannelInfo();
        }
    });

    try {
        const response = await fetch('/api/extraction-status');
        const status = await response.json();

        // If there's an active extraction or queue items, start polling
        if (status.active || (status.queue && status.queue.some(q => q.status === 'queued' || q.status === 'running'))) {
            document.getElementById('scrapeCard').style.display = 'block';
            startPolling();
        }

        // Update queue display
        updateQueueDisplay(status.queue || []);
    } catch (error) {
        console.error('Initial status check failed:', error);
    }
});

// =============================================================================
// Bootstrap Tab - Topic Naming
// =============================================================================

let bootstrapTopics = [];
let groundTruth = {};
let currentNamingTopic = null;

async function loadBootstrapTopics() {
    try {
        // Load topics
        const topicsRes = await fetch('/api/modeling/topics');
        if (topicsRes.ok) {
            const data = await topicsRes.json();
            bootstrapTopics = data.topics || [];
        }

        // Load existing ground truth
        const gtRes = await fetch('/api/bootstrap/ground-truth');
        if (gtRes.ok) {
            groundTruth = await gtRes.json();
        } else {
            groundTruth = {};
        }

        updateNamingProgress();
        renderGroundTruthStats();

        if (bootstrapTopics.length > 0) {
            document.getElementById('namingEmpty').style.display = 'none';
            document.getElementById('namingContent').style.display = 'block';
            showRandomUnnamedTopic();
        } else {
            document.getElementById('namingEmpty').innerHTML = `
                <div class="empty-icon">📊</div>
                <div class="empty-title">No topics available</div>
                <div class="empty-text">Run extract_topics.py first</div>
            `;
        }
    } catch (error) {
        console.error('Error loading bootstrap data:', error);
    }
}

function updateNamingProgress() {
    const named = Object.keys(groundTruth).length;
    const total = bootstrapTopics.length;
    document.getElementById('namingProgress').textContent = `${named}/${total} named`;
}

function showRandomUnnamedTopic() {
    // Find topics not yet in ground truth
    const unnamed = bootstrapTopics.filter(t => !groundTruth[t.id]);
    
    if (unnamed.length === 0) {
        // All named! Show a random one for review
        showRandomTopic();
        return;
    }

    // Pick random unnamed
    const topic = unnamed[Math.floor(Math.random() * unnamed.length)];
    displayTopicForNaming(topic);
}

function showRandomTopic() {
    if (bootstrapTopics.length === 0) return;
    const topic = bootstrapTopics[Math.floor(Math.random() * bootstrapTopics.length)];
    displayTopicForNaming(topic);
}

function displayTopicForNaming(topic) {
    currentNamingTopic = topic;

    document.getElementById('namingTopicId').textContent = topic.id;
    document.getElementById('namingTopicCount').textContent = `${topic.count} comments`;

    // Top words
    const words = topic.top_words_centroid_mmr || topic.top_words || [];
    document.getElementById('namingTopWords').innerHTML = words.slice(0, 15).map(w =>
        `<span style="display: inline-block; background: var(--bg-secondary); padding: 4px 10px; border-radius: 4px; margin: 2px; font-size: 13px;">${w}</span>`
    ).join('');

    // Example comments
    const comments = topic.example_comments_centroid_mmr || topic.example_comments_original || [];
    document.getElementById('namingComments').innerHTML = comments.slice(0, 10).map((c, i) =>
        `<p style="margin: 8px 0; padding-bottom: 8px; border-bottom: 1px solid var(--border-color); font-size: 13px; color: var(--text-secondary); line-height: 1.5;">
            <span style="color: var(--text-muted);">${i + 1}.</span> ${c}
        </p>`
    ).join('');

    // Pre-fill with existing name if any
    const existingName = groundTruth[topic.id] || topic.generated_name || '';
    document.getElementById('namingInput').value = existingName;
    document.getElementById('namingInput').focus();
}

async function saveAndNextTopic() {
    if (!currentNamingTopic) return;

    const name = document.getElementById('namingInput').value.trim();
    if (!name) {
        alert('Please enter a name for the topic');
        return;
    }

    // Save to ground truth
    groundTruth[currentNamingTopic.id] = name;

    try {
        await fetch('/api/bootstrap/ground-truth', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ topic_id: currentNamingTopic.id, name: name })
        });
    } catch (error) {
        console.error('Error saving ground truth:', error);
    }

    updateNamingProgress();
    renderGroundTruthStats();
    showRandomUnnamedTopic();
}

function skipTopic() {
    showRandomUnnamedTopic();
}

function renderGroundTruthStats() {
    const container = document.getElementById('groundTruthStats');
    const entries = Object.entries(groundTruth);

    if (entries.length === 0) {
        container.innerHTML = 'No ground truth data yet. Start naming topics above!';
        return;
    }

    container.innerHTML = `
        <p style="margin-bottom: 12px;"><strong>${entries.length}</strong> topics named</p>
        <div style="max-height: 200px; overflow-y: auto;">
            ${entries.slice(-10).reverse().map(([id, name]) =>
                `<div style="padding: 6px 0; border-bottom: 1px solid var(--border-color); display: flex; justify-content: space-between;">
                    <span style="color: var(--text-muted);">Topic ${id}</span>
                    <span>${name}</span>
                </div>`
            ).join('')}
            ${entries.length > 10 ? `<p style="color: var(--text-muted); margin-top: 8px;">... and ${entries.length - 10} more</p>` : ''}
        </div>
    `;
}

function downloadGroundTruth() {
    const dataStr = JSON.stringify(groundTruth, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'naming_ground_truth.json';
    a.click();
    URL.revokeObjectURL(url);
}

function showBootstrapSubtab(subtab) {
    document.querySelectorAll('.bootstrap-subtab-content').forEach(el => {
        el.style.display = 'none';
    });
    document.querySelectorAll('.bootstrap-subtab').forEach(btn => {
        btn.classList.remove('active');
    });
    document.getElementById('bootstrap-subtab-' + subtab).style.display = 'block';
    document.querySelector(`.bootstrap-subtab[data-subtab="${subtab}"]`)?.classList.add('active');
}

