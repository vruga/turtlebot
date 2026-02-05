/**
 * Agricultural Disease Detection Dashboard
 * Real-time updates and control interface
 */

// State
let isConnected = false;
let eventSource = null;
let pendingSpray = null;
let sprayLog = [];

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initEventSource();
    fetchInitialState();
    checkCameraFeed();

    // Keyboard shortcut for capture (Spacebar)
    document.addEventListener('keydown', function(e) {
        if (e.code === 'Space' && e.target.tagName !== 'INPUT') {
            e.preventDefault();
            triggerCapture();
        }
    });
});

// Server-Sent Events for real-time updates
function initEventSource() {
    eventSource = new EventSource('/events');

    eventSource.onopen = function() {
        isConnected = true;
        updateConnectionStatus(true);
    };

    eventSource.onerror = function() {
        isConnected = false;
        updateConnectionStatus(false);

        // Reconnect after delay
        setTimeout(function() {
            if (!isConnected) {
                initEventSource();
            }
        }, 5000);
    };

    eventSource.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            handleEvent(data);
        } catch (e) {
            console.error('Failed to parse event:', e);
        }
    };
}

// Handle incoming events
function handleEvent(event) {
    updateLastUpdate();

    switch (event.type) {
        case 'detection':
            updateDetection(event.data);
            break;
        case 'recommendation':
            updateRecommendation(event.data);
            break;
        case 'spray_status':
            updateSprayStatus(event.data);
            break;
        case 'system_status':
            updateSystemStatus(event.data);
            break;
    }
}

// Fetch initial state
function fetchInitialState() {
    fetch('/api/state')
        .then(response => response.json())
        .then(data => {
            if (data.latest_detection) {
                updateDetection(data.latest_detection);
            }
            if (data.latest_recommendation) {
                updateRecommendation(data.latest_recommendation);
            }
            if (data.spray_status) {
                updateSprayStatus(data.spray_status);
            }
            if (data.system_status) {
                updateSystemStatus(data.system_status);
            }
            if (data.detection_history) {
                data.detection_history.forEach(d => addToHistory(d));
            }
        })
        .catch(err => console.error('Failed to fetch state:', err));
}

// Update detection display
function updateDetection(detection) {
    const container = document.getElementById('detection-result');
    const confidenceFill = document.getElementById('confidence-fill');
    const confidenceValue = document.getElementById('confidence-value');

    if (detection.error) {
        container.innerHTML = `
            <div class="detection-display">
                <div class="detection-disease" style="color: var(--error-color)">Error</div>
                <p>${detection.message || 'Detection failed'}</p>
            </div>
        `;
        return;
    }

    const disease = formatDiseaseName(detection.disease_name);
    const confidence = (detection.confidence * 100).toFixed(1);
    const severity = detection.severity || 'unknown';

    container.innerHTML = `
        <div class="detection-display">
            <div class="detection-disease">${disease}</div>
            <span class="detection-severity ${severity}">${severity}</span>
            <div class="detection-action">
                ${detection.should_spray
                    ? `Spray applied: ${detection.spray_duration}ms`
                    : 'No spray needed'}
            </div>
        </div>
    `;

    // Update confidence meter
    confidenceFill.style.width = `${confidence}%`;
    confidenceValue.textContent = `${confidence}%`;

    // Add to history
    addToHistory(detection);

    // Stop capture animation
    document.getElementById('capture-btn').classList.remove('capturing');
}

// Update recommendation display
function updateRecommendation(recommendation) {
    const container = document.getElementById('recommendation-content');
    const source = document.getElementById('recommendation-source');

    container.innerHTML = `
        <div class="recommendation-text">${recommendation.recommendation}</div>
    `;

    const sourceText = recommendation.source === 'api' ? 'Claude AI'
        : recommendation.source === 'cache' ? 'Cached'
        : 'Local Database';
    source.textContent = `Source: ${sourceText}`;
}

// Update spray status
function updateSprayStatus(status) {
    const statusText = document.getElementById('spray-status-text');
    const emergencyBtn = document.getElementById('emergency-btn');

    if (status.emergency_stopped) {
        statusText.textContent = 'EMERGENCY STOPPED';
        statusText.style.color = 'var(--error-color)';
        emergencyBtn.textContent = 'RESUME';
        emergencyBtn.classList.add('stopped');
    } else if (status.event === 'awaiting_confirmation') {
        showConfirmationModal(status);
    } else if (status.event === 'spray_complete') {
        statusText.textContent = 'Spray Complete';
        statusText.style.color = 'var(--status-ok)';
        addSprayLog(status, true);
    } else if (status.event === 'spray_failed') {
        statusText.textContent = 'Spray Failed';
        statusText.style.color = 'var(--error-color)';
        addSprayLog(status, false);
    } else {
        statusText.textContent = status.connected ? 'Ready' : 'Disconnected';
        statusText.style.color = status.connected ? 'var(--text-color)' : 'var(--error-color)';
        emergencyBtn.textContent = 'EMERGENCY STOP';
        emergencyBtn.classList.remove('stopped');
    }
}

// Update system status indicators
function updateSystemStatus(status) {
    const components = ['camera', 'model', 'esp32', 'llm'];

    components.forEach(component => {
        const element = document.getElementById(`status-${component}`);
        if (element) {
            element.classList.remove('ok', 'warning', 'error');
            const state = status[component] || 'unknown';
            if (state === 'ok') {
                element.classList.add('ok');
            } else if (state === 'warning') {
                element.classList.add('warning');
            } else if (state === 'error') {
                element.classList.add('error');
            }
        }
    });
}

// Add detection to history
function addToHistory(detection) {
    const list = document.getElementById('history-list');

    // Remove empty message
    const empty = list.querySelector('.history-empty');
    if (empty) empty.remove();

    // Create history item
    const item = document.createElement('div');
    item.className = 'history-item';

    const confidence = (detection.confidence * 100).toFixed(0);
    const confidenceClass = confidence >= 90 ? 'high' : confidence >= 80 ? 'medium' : 'low';
    const time = new Date(detection.timestamp).toLocaleTimeString();

    item.innerHTML = `
        <img class="history-thumb" src="/api/latest-frame" alt="Capture">
        <div class="history-info">
            <div class="history-disease">${formatDiseaseName(detection.disease_name)}</div>
            <div class="history-time">${time}</div>
        </div>
        <span class="history-confidence ${confidenceClass}">${confidence}%</span>
    `;

    // Insert at top
    list.insertBefore(item, list.firstChild);

    // Limit history items
    while (list.children.length > 10) {
        list.removeChild(list.lastChild);
    }
}

// Add spray event to log
function addSprayLog(event, success) {
    const list = document.getElementById('spray-log');

    // Remove empty message
    const empty = list.querySelector('.log-empty');
    if (empty) empty.remove();

    const item = document.createElement('div');
    item.className = 'log-item';

    const time = new Date().toLocaleTimeString();
    const action = success
        ? `Sprayed ${event.duration_ms || 0}ms`
        : 'Spray failed';

    item.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-action ${success ? 'success' : 'failed'}">${action}</span>
    `;

    list.insertBefore(item, list.firstChild);

    // Limit log items
    while (list.children.length > 10) {
        list.removeChild(list.lastChild);
    }
}

// Trigger frame capture
function triggerCapture() {
    const btn = document.getElementById('capture-btn');
    btn.classList.add('capturing');

    fetch('/api/capture', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (!data.success) {
                btn.classList.remove('capturing');
                alert(data.message || 'Capture failed');
            }
        })
        .catch(err => {
            btn.classList.remove('capturing');
            console.error('Capture failed:', err);
        });
}

// Manual spray
function manualSpray(duration) {
    fetch('/api/spray', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: `SPRAY:${duration}` })
    })
        .then(response => response.json())
        .then(data => {
            if (!data.success) {
                alert(data.message || 'Spray command failed');
            }
        })
        .catch(err => console.error('Spray failed:', err));
}

// Emergency stop
function emergencyStop() {
    const btn = document.getElementById('emergency-btn');

    if (btn.classList.contains('stopped')) {
        // Resume
        fetch('/api/resume', { method: 'POST' })
            .then(response => response.json())
            .catch(err => console.error('Resume failed:', err));
    } else {
        // Stop
        fetch('/api/emergency-stop', { method: 'POST' })
            .then(response => response.json())
            .catch(err => console.error('Emergency stop failed:', err));
    }
}

// Show confirmation modal
function showConfirmationModal(status) {
    pendingSpray = status;
    const modal = document.getElementById('confirmation-modal');
    const message = document.getElementById('confirmation-message');

    message.textContent = `Spray ${status.duration || 0}ms for ${formatDiseaseName(status.disease || 'unknown')}?`;
    modal.classList.add('visible');
}

// Handle spray confirmation
function confirmSpray(confirmed) {
    const modal = document.getElementById('confirmation-modal');
    modal.classList.remove('visible');

    fetch('/api/spray/confirm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confirm: confirmed })
    })
        .then(response => response.json())
        .catch(err => console.error('Confirmation failed:', err));

    pendingSpray = null;
}

// Check camera feed
function checkCameraFeed() {
    const img = document.getElementById('camera-stream');
    const overlay = document.getElementById('camera-overlay');

    img.onerror = function() {
        overlay.classList.add('visible');
    };

    img.onload = function() {
        overlay.classList.remove('visible');
    };
}

// Update connection status
function updateConnectionStatus(connected) {
    const element = document.getElementById('connection-status');
    element.textContent = connected ? 'Connected' : 'Disconnected';
    element.className = connected ? 'connected' : 'disconnected';
}

// Update last update time
function updateLastUpdate() {
    const element = document.getElementById('last-update');
    element.textContent = `Updated: ${new Date().toLocaleTimeString()}`;
}

// Format disease name for display
function formatDiseaseName(name) {
    if (!name) return 'Unknown';
    return name
        .replace(/_/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase());
}
