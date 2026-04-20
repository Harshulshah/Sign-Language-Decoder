/**
 * Sign Language Decoder — Frontend JavaScript (Production-Level)
 * 
 * Features:
 * - Detection state rendering (IDLE/DETECTING/LOCKED/FORMING)
 * - Gesture trail visualization with gradient
 * - Hold-to-confirm progress ring
 * - Adaptive frame rate (slow when idle)
 * - Auto-speak on word completion
 * - FPS counter
 * - Web Speech API TTS
 */

// ─── State ──────────────────────────────────────────────────
const state = {
    cameraActive: false,
    modelReady: false,
    pollingInterval: null,
    frameInterval: null,
    recentSigns: [],
    startTime: null,
    currentMode: 'auto',
    libraryData: null,
    activeCategory: 'all',
    cameraMode: 'client',
    stream: null,
    ttsServerAvailable: false,
    lastSpokenText: '',
    lastSpeakTime: 0,
    detectionState: 'idle',
    handDetected: false,
    frameRate: 150,        // Current frame interval (ms)
    idleFrameRate: 500,    // Slower rate when no hand
    activeFrameRate: 150,  // Normal rate when hand present
    trailPoints: [],       // Gesture trail buffer
    lastSentence: '',      // Track sentence changes for auto-speak
    signCount: 0,
};

// ─── DOM Elements ───────────────────────────────────────────
const dom = {
    videoFeed: document.getElementById("videoFeed"),
    videoFeedMjpeg: document.getElementById("videoFeedMjpeg"),
    videoPlaceholder: document.getElementById("videoPlaceholder"),
    captureCanvas: document.getElementById("captureCanvas"),
    liveBadge: document.getElementById("liveBadge"),
    predictionOverlay: document.getElementById("predictionOverlay"),
    predictionLabel: document.getElementById("predictionLabel"),
    predictionConf: document.getElementById("predictionConf"),
    predictionCategory: document.getElementById("predictionCategory"),
    confidenceBarFill: document.getElementById("confidenceBarFill"),
    sentenceDisplay: document.getElementById("sentenceDisplay"),
    partialWord: document.getElementById("partialWord"),
    statPrediction: document.getElementById("statPrediction"),
    statConfidence: document.getElementById("statConfidence"),
    statSigns: document.getElementById("statSigns"),
    statTime: document.getElementById("statTime"),
    statFps: document.getElementById("statFps"),
    statState: document.getElementById("statState"),
    btnStartCamera: document.getElementById("btnStartCamera"),
    btnStopCamera: document.getElementById("btnStopCamera"),
    btnReset: document.getElementById("btnReset"),
    btnSpeak: document.getElementById("btnSpeak"),
    btnBackspace: document.getElementById("btnBackspace"),
    btnSpace: document.getElementById("btnSpace"),
    btnSpeakSentence: document.getElementById("btnSpeakSentence"),
    btnClear: document.getElementById("btnClear"),
    btnLibrary: document.getElementById("btnLibrary"),
    btnAddSign: document.getElementById("btnAddSign"),
    autoSpeakToggle: document.getElementById("autoSpeakToggle"),
    statusBadge: document.getElementById("statusBadge"),
    statusText: document.getElementById("statusText"),
    recentSignsContainer: document.getElementById("recentSigns"),
    modelStatusIcon: document.getElementById("modelStatusIcon"),
    modelStatusLabel: document.getElementById("modelStatusLabel"),
    modelStatusDetail: document.getElementById("modelStatusDetail"),
    toastContainer: document.getElementById("toastContainer"),
    suggestionChips: document.getElementById("suggestionChips"),
    modeButtons: document.getElementById("modeButtons"),
    libraryModal: document.getElementById("libraryModal"),
    libraryClose: document.getElementById("libraryClose"),
    librarySearch: document.getElementById("librarySearch"),
    libraryTabs: document.getElementById("libraryTabs"),
    libraryGrid: document.getElementById("libraryGrid"),
    addSignModal: document.getElementById("addSignModal"),
    addSignClose: document.getElementById("addSignClose"),
    btnSaveSign: document.getElementById("btnSaveSign"),
    newSignLabel: document.getElementById("newSignLabel"),
    newSignDescription: document.getElementById("newSignDescription"),
    newSignCategory: document.getElementById("newSignCategory"),
    detectionStateBadge: document.getElementById("detectionStateBadge"),
    stateIcon: document.getElementById("stateIcon"),
    stateText: document.getElementById("stateText"),
    holdRingFill: document.getElementById("holdRingFill"),
    holdProgressContainer: document.getElementById("holdProgressContainer"),
    trailCanvas: document.getElementById("trailCanvas"),
};

// ─── API Helpers ────────────────────────────────────────────
async function apiCall(endpoint, method = "GET", body = null) {
    try {
        const options = { method, headers: { "Content-Type": "application/json" } };
        if (body) options.body = JSON.stringify(body);
        const response = await fetch(`/api${endpoint}`, options);
        return await response.json();
    } catch (error) {
        console.error(`API Error [${endpoint}]:`, error);
        return null;
    }
}

// ─── Toast Notifications ────────────────────────────────────
function showToast(message, type = "info", duration = 3000) {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    dom.toastContainer.appendChild(toast);
    setTimeout(() => {
        toast.style.animation = "toastOut 0.4s ease-in forwards";
        setTimeout(() => toast.remove(), 400);
    }, duration);
}

// ─── Web Speech API TTS ─────────────────────────────────────
function speakText(text) {
    if (!text || !text.trim()) return;
    const now = Date.now();
    if (text === state.lastSpokenText && now - state.lastSpeakTime < 2000) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;
    utterance.lang = 'en-US';
    window.speechSynthesis.speak(utterance);
    state.lastSpokenText = text;
    state.lastSpeakTime = now;
}

// ═══════════════════════════════════════════════════════════════
//   DETECTION STATE RENDERING
// ═══════════════════════════════════════════════════════════════

const STATE_CONFIG = {
    idle:      { icon: '🔴', text: 'No Hand',    class: 'state-idle',      color: '#ff3b5c' },
    detecting: { icon: '🟡', text: 'Detecting',  class: 'state-detecting', color: '#ffc107' },
    locked:    { icon: '🟢', text: 'Locked ✓',   class: 'state-locked',    color: '#00e676' },
    forming:   { icon: '🔵', text: 'Forming Word', class: 'state-forming', color: '#448aff' },
};

function updateDetectionState(newState) {
    if (newState === state.detectionState) return;
    state.detectionState = newState;

    const config = STATE_CONFIG[newState] || STATE_CONFIG.idle;
    
    dom.detectionStateBadge.className = `detection-state-badge ${config.class}`;
    dom.stateIcon.textContent = config.icon;
    dom.stateText.textContent = config.text;
    dom.statState.textContent = config.text;
    dom.statState.style.color = config.color;
}

function updateHoldProgress(progress) {
    if (!dom.holdRingFill) return;
    const circumference = 2 * Math.PI * 26; // r=26
    const offset = circumference * (1 - progress);
    dom.holdRingFill.style.strokeDasharray = circumference;
    dom.holdRingFill.style.strokeDashoffset = offset;

    // Show/hide based on state
    if (progress > 0 && progress < 1 && state.detectionState === 'detecting') {
        dom.holdProgressContainer.classList.add('visible');
    } else {
        dom.holdProgressContainer.classList.remove('visible');
    }
}

// ═══════════════════════════════════════════════════════════════
//   GESTURE TRAIL VISUALIZATION
// ═══════════════════════════════════════════════════════════════

function drawGestureTrail(trailPoints) {
    const cvs = dom.trailCanvas;
    if (!cvs || !trailPoints || trailPoints.length < 2) {
        if (cvs) cvs.getContext("2d").clearRect(0, 0, cvs.width, cvs.height);
        return;
    }

    const ctx = cvs.getContext("2d");
    ctx.clearRect(0, 0, cvs.width, cvs.height);

    const len = trailPoints.length;

    for (let i = 1; i < len; i++) {
        const prev = trailPoints[i - 1];
        const curr = trailPoints[i];
        const progress = i / len; // 0 = oldest, 1 = newest

        // Multi-color gradient: cyan → magenta
        const r = Math.round(0 + progress * 255);
        const g = Math.round(255 - progress * 100);
        const b = Math.round(200 + progress * 55);
        const alpha = 0.15 + progress * 0.85; // Fade older points

        ctx.beginPath();
        ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
        ctx.lineWidth = 1.5 + progress * 4; // Thicker at tip
        ctx.lineCap = 'round';
        ctx.moveTo(prev[0] * cvs.width, prev[1] * cvs.height);
        ctx.lineTo(curr[0] * cvs.width, curr[1] * cvs.height);
        ctx.stroke();
    }

    // Draw glowing dot at fingertip (latest point)
    const tip = trailPoints[len - 1];
    const tipX = tip[0] * cvs.width;
    const tipY = tip[1] * cvs.height;

    // Outer glow
    const glow = ctx.createRadialGradient(tipX, tipY, 0, tipX, tipY, 12);
    glow.addColorStop(0, 'rgba(0, 255, 200, 0.9)');
    glow.addColorStop(0.5, 'rgba(0, 255, 200, 0.3)');
    glow.addColorStop(1, 'rgba(0, 255, 200, 0)');
    ctx.beginPath();
    ctx.fillStyle = glow;
    ctx.arc(tipX, tipY, 12, 0, 2 * Math.PI);
    ctx.fill();

    // Inner dot
    ctx.beginPath();
    ctx.fillStyle = '#00FFC8';
    ctx.arc(tipX, tipY, 4, 0, 2 * Math.PI);
    ctx.fill();
}

// ═══════════════════════════════════════════════════════════════
//   CAMERA: Client-side capture via getUserMedia
// ═══════════════════════════════════════════════════════════════

async function startClientCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: "user" }
        });
        state.stream = stream;
        dom.videoFeed.srcObject = stream;
        dom.videoFeed.style.display = "block";
        dom.videoFeedMjpeg.style.display = "none";
        dom.videoPlaceholder.classList.add("hidden");
        await new Promise(resolve => { dom.videoFeed.onloadedmetadata = resolve; });
        dom.captureCanvas.width = 640;
        dom.captureCanvas.height = 480;
        startFrameCapture();
        return true;
    } catch (err) {
        console.error("getUserMedia failed:", err);
        showToast("Camera access denied. Please allow camera.", "error");
        return false;
    }
}

function stopClientCamera() {
    if (state.stream) {
        state.stream.getTracks().forEach(track => track.stop());
        state.stream = null;
    }
    dom.videoFeed.srcObject = null;
    stopFrameCapture();
}

function startFrameCapture() {
    if (state.frameInterval) clearInterval(state.frameInterval);
    state.frameRate = state.activeFrameRate;
    state.frameInterval = setInterval(captureAndSendFrame, state.frameRate);
}

function stopFrameCapture() {
    if (state.frameInterval) {
        clearInterval(state.frameInterval);
        state.frameInterval = null;
    }
}

// Adaptive frame rate: slow down when no hand detected
function adjustFrameRate(handDetected) {
    const targetRate = handDetected ? state.activeFrameRate : state.idleFrameRate;
    if (targetRate !== state.frameRate) {
        state.frameRate = targetRate;
        if (state.frameInterval) {
            clearInterval(state.frameInterval);
            state.frameInterval = setInterval(captureAndSendFrame, state.frameRate);
        }
    }
}

let frameInFlight = false;

async function captureAndSendFrame() {
    if (!state.cameraActive || frameInFlight) return;
    const video = dom.videoFeed;
    if (!video.videoWidth) return;

    const ctx = dom.captureCanvas.getContext("2d");
    ctx.drawImage(video, 0, 0, 640, 480);
    const frameData = dom.captureCanvas.toDataURL("image/jpeg", 0.7);

    frameInFlight = true;
    try {
        const data = await apiCall("/process_frame", "POST", { frame: frameData });
        if (data && !data.error) {
            updatePredictionUI(data);
            adjustFrameRate(data.hand_detected);
        }
    } catch (e) { /* Silently fail */ }
    frameInFlight = false;
}

// ═══════════════════════════════════════════════════════════════
//   CAMERA: Local mode (MJPEG stream fallback)
// ═══════════════════════════════════════════════════════════════

async function startLocalCamera() {
    const result = await apiCall("/start_camera", "POST");
    if (result && result.status === "ok") {
        dom.videoFeedMjpeg.src = "/api/video_feed?" + Date.now();
        dom.videoFeedMjpeg.style.display = "block";
        dom.videoFeed.style.display = "none";
        dom.videoPlaceholder.classList.add("hidden");
        startPolling();
        return true;
    }
    return false;
}

function stopLocalCamera() {
    apiCall("/stop_camera", "POST");
    dom.videoFeedMjpeg.src = "";
    dom.videoFeedMjpeg.style.display = "none";
    stopPolling();
}

// ─── Unified Camera Start/Stop ──────────────────────────────

async function startCamera() {
    dom.btnStartCamera.disabled = true;
    dom.btnStartCamera.innerHTML = '<span class="spinner"></span> Starting...';
    let success = false;

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        state.cameraMode = 'client';
        success = await startClientCamera();
    }
    if (!success) {
        state.cameraMode = 'local';
        success = await startLocalCamera();
    }

    if (success) {
        state.cameraActive = true;
        state.startTime = Date.now();
        dom.liveBadge.classList.add("visible");
        dom.predictionOverlay.classList.add("visible");
        dom.statusBadge.classList.add("active");
        dom.statusText.textContent = state.cameraMode === 'client' ? "Camera Active (Browser)" : "Camera Active (Local)";
        dom.btnStartCamera.style.display = "none";
        dom.btnStopCamera.style.display = "inline-flex";
        showToast(`Camera started — show your signs!`, "success");
    } else {
        showToast("Failed to start camera. Check permissions.", "error");
    }

    dom.btnStartCamera.disabled = false;
    dom.btnStartCamera.innerHTML = '<span class="icon">📷</span> Start Camera';
}

async function stopCamera() {
    state.cameraActive = false;
    if (state.cameraMode === 'client') stopClientCamera();
    else stopLocalCamera();

    dom.videoFeed.style.display = "none";
    dom.videoFeedMjpeg.style.display = "none";
    dom.videoPlaceholder.classList.remove("hidden");
    dom.liveBadge.classList.remove("visible");
    dom.predictionOverlay.classList.remove("visible");
    dom.statusBadge.classList.remove("active");
    dom.statusText.textContent = "Camera Inactive";
    dom.btnStartCamera.style.display = "inline-flex";
    dom.btnStopCamera.style.display = "none";

    // Clear canvases
    ["overlayCanvas", "trailCanvas"].forEach(id => {
        const cvs = document.getElementById(id);
        if (cvs) cvs.getContext("2d").clearRect(0, 0, cvs.width, cvs.height);
    });

    updateDetectionState('idle');
    updateHoldProgress(0);
    showToast("Camera stopped", "info");
}

// ─── Local Mode Polling ─────────────────────────────────────
function startPolling() {
    if (state.pollingInterval) clearInterval(state.pollingInterval);
    state.pollingInterval = setInterval(async () => {
        if (!state.cameraActive) return;
        const data = await apiCall("/predict");
        if (data) updatePredictionUI(data);
    }, 250);
}

function stopPolling() {
    if (state.pollingInterval) {
        clearInterval(state.pollingInterval);
        state.pollingInterval = null;
    }
}

// ─── Session Timer ──────────────────────────────────────────
setInterval(() => {
    if (state.startTime && state.cameraActive) {
        const elapsed = Math.floor((Date.now() - state.startTime) / 1000);
        const mins = Math.floor(elapsed / 60);
        const secs = elapsed % 60;
        dom.statTime.textContent = `${mins}:${secs.toString().padStart(2, "0")}`;
    }
}, 1000);

// ═══════════════════════════════════════════════════════════════
//   UPDATE UI
// ═══════════════════════════════════════════════════════════════

let lastPrediction = "";
let animatedConfidence = 0;

function updatePredictionUI(data) {
    const { prediction, confidence, category, sentence, partial_word,
            model_ready, mode, suggestions, tts_server, landmarks,
            detection_state, hold_progress, trail_points, speak_trigger, fps } = data;

    if (tts_server !== undefined) state.ttsServerAvailable = tts_server;

    // Draw landmarks
    if (landmarks !== undefined) drawLandmarks(landmarks);

    // Draw gesture trail
    if (trail_points && trail_points.length > 1) {
        drawGestureTrail(trail_points);
    } else if (dom.trailCanvas) {
        dom.trailCanvas.getContext("2d").clearRect(0, 0, dom.trailCanvas.width, dom.trailCanvas.height);
    }

    // Update detection state
    if (detection_state) updateDetectionState(detection_state);

    // Update hold progress
    if (hold_progress !== undefined) updateHoldProgress(hold_progress);

    // FPS
    if (fps !== undefined && dom.statFps) {
        dom.statFps.textContent = fps > 0 ? `${fps}` : '—';
    }

    // Model status
    state.modelReady = model_ready;
    if (model_ready) {
        dom.modelStatusIcon.textContent = "✅";
        dom.modelStatusLabel.textContent = "Model Ready";
        dom.modelStatusDetail.textContent = "Rule-based + Motion detection active";
    }

    // Prediction overlay
    if (prediction) {
        dom.predictionLabel.textContent = prediction;
        dom.predictionConf.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
        dom.predictionCategory.textContent = category ? `[${category}]` : '';

        // Smooth confidence bar animation
        animatedConfidence += (confidence - animatedConfidence) * 0.3;
        dom.confidenceBarFill.style.width = `${animatedConfidence * 100}%`;

        // Color-code confidence
        if (confidence >= 0.8) {
            dom.confidenceBarFill.className = 'confidence-bar-fill high';
        } else if (confidence >= 0.5) {
            dom.confidenceBarFill.className = 'confidence-bar-fill medium';
        } else {
            dom.confidenceBarFill.className = 'confidence-bar-fill low';
        }

        dom.statPrediction.textContent = prediction;
        dom.statConfidence.textContent = `${(confidence * 100).toFixed(0)}%`;

        if (confidence >= 0.8) dom.statConfidence.className = "stat-value success";
        else if (confidence >= 0.5) dom.statConfidence.className = "stat-value warning";
        else dom.statConfidence.className = "stat-value";

        if (prediction !== lastPrediction) {
            dom.predictionLabel.classList.remove("detect-pop");
            void dom.predictionLabel.offsetWidth;
            dom.predictionLabel.classList.add("detect-pop");
            lastPrediction = prediction;
        }

        if (state.recentSigns.length === 0 || state.recentSigns[0].label !== prediction) {
            state.recentSigns.unshift({ label: prediction, category: category || '' });
            if (state.recentSigns.length > 12) state.recentSigns.pop();
            updateRecentSigns();
        }
    } else {
        dom.predictionLabel.textContent = "—";
        dom.predictionConf.textContent = "Waiting for gesture...";
        dom.predictionCategory.textContent = "";
        animatedConfidence *= 0.9; // Smooth decay
        dom.confidenceBarFill.style.width = `${animatedConfidence * 100}%`;
        dom.statPrediction.textContent = "—";
    }

    updateSentenceDisplay(sentence);

    if (partial_word) {
        dom.partialWord.textContent = `Spelling: ${partial_word}_`;
        dom.partialWord.classList.add('active');
    } else {
        dom.partialWord.textContent = "";
        dom.partialWord.classList.remove('active');
    }

    // Count actual signs
    if (sentence) {
        const chars = sentence.replace(/\s/g, "").length;
        dom.statSigns.textContent = chars;
    } else {
        dom.statSigns.textContent = "0";
    }

    if (suggestions) updateSuggestions(suggestions);

    // Auto-speak trigger from server
    if (speak_trigger && dom.autoSpeakToggle.checked) {
        speakText(speak_trigger);
    }

    // Track sentence changes for client-side auto-speak
    if (sentence !== state.lastSentence) {
        // Check if a word was just completed (space was added)
        if (dom.autoSpeakToggle.checked && sentence && sentence.endsWith(" ") && 
            sentence.length > (state.lastSentence || '').length) {
            const words = sentence.trim().split(/\s+/);
            const lastWord = words[words.length - 1];
            if (lastWord && lastWord.length > 1 && !state.ttsServerAvailable) {
                speakText(lastWord);
            }
        }
        state.lastSentence = sentence || '';
    }
}

function updateSentenceDisplay(sentence) {
    if (sentence && sentence.length > 0) {
        // Highlight words with spans
        const words = sentence.split(' ');
        const html = words.map((w, i) => {
            if (!w) return ' ';
            const isLast = i === words.length - 1;
            return `<span class="sentence-word ${isLast ? 'latest' : ''}">${escapeHtml(w)}</span>`;
        }).join(' ') + '<span class="cursor-blink"></span>';
        dom.sentenceDisplay.innerHTML = html;
        dom.sentenceDisplay.classList.remove("empty");
        // Auto-scroll
        dom.sentenceDisplay.scrollTop = dom.sentenceDisplay.scrollHeight;
    } else {
        dom.sentenceDisplay.innerHTML = '<span class="cursor-blink"></span>';
        dom.sentenceDisplay.classList.add("empty");
    }
}

function updateRecentSigns() {
    dom.recentSignsContainer.innerHTML = state.recentSigns
        .map(s => {
            const isWord = s.label.length > 1 && !['SPACE','DELETE'].includes(s.label);
            return `<span class="sign-chip ${isWord ? 'word' : ''}">${escapeHtml(s.label)}</span>`;
        })
        .join("");
}

// ─── Suggestions ────────────────────────────────────────────
function updateSuggestions(suggestions) {
    const chips = [];
    if (suggestions.completions) {
        suggestions.completions.slice(0, 4).forEach(word => {
            chips.push(`<span class="suggestion-chip" onclick="insertCompletion('${escapeAttr(word)}')">${escapeHtml(word)}</span>`);
        });
    }
    if (suggestions.corrections) {
        suggestions.corrections.slice(0, 2).forEach(word => {
            chips.push(`<span class="suggestion-chip correction" onclick="insertCompletion('${escapeAttr(word)}')" title="Did you mean?">✏️ ${escapeHtml(word)}</span>`);
        });
    }
    if (suggestions.next_words) {
        suggestions.next_words.slice(0, 3).forEach(word => {
            chips.push(`<span class="suggestion-chip phrase" onclick="insertWord('${escapeAttr(word)}')" title="Next word">→ ${escapeHtml(word)}</span>`);
        });
    }
    dom.suggestionChips.innerHTML = chips.length > 0
        ? chips.join("")
        : '<span class="suggestion-placeholder">Show a sign to see suggestions...</span>';
}

// ─── Landmarks Drawing ─────────────────────────────────────
const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]
];

function drawLandmarks(landmarks) {
    const cvs = document.getElementById("overlayCanvas");
    if (!cvs) return;
    const ctx = cvs.getContext("2d");
    ctx.clearRect(0, 0, cvs.width, cvs.height);
    if (!landmarks) return;

    for (let h = 0; h < 2; h++) {
        let offset = h * 63;
        let isZero = true;
        for (let i = 0; i < 63; i++) {
            if (landmarks[offset + i] !== 0) { isZero = false; break; }
        }
        if (isZero) continue;

        // Draw connections with gradient
        ctx.lineWidth = 2.5;
        HAND_CONNECTIONS.forEach(([i, j]) => {
            const x1 = landmarks[offset + i*3] * cvs.width;
            const y1 = landmarks[offset + i*3 + 1] * cvs.height;
            const x2 = landmarks[offset + j*3] * cvs.width;
            const y2 = landmarks[offset + j*3 + 1] * cvs.height;

            const gradient = ctx.createLinearGradient(x1, y1, x2, y2);
            gradient.addColorStop(0, 'rgba(0, 255, 200, 0.9)');
            gradient.addColorStop(1, 'rgba(0, 200, 255, 0.9)');
            ctx.strokeStyle = gradient;

            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
        });

        // Draw landmark points
        for (let i = 0; i < 21; i++) {
            const x = landmarks[offset + i*3] * cvs.width;
            const y = landmarks[offset + i*3 + 1] * cvs.height;

            // Fingertips (4,8,12,16,20) get larger, colored dots
            const isTip = [4, 8, 12, 16, 20].includes(i);

            ctx.beginPath();
            if (isTip) {
                ctx.fillStyle = '#FF3366';
                ctx.arc(x, y, 6, 0, 2*Math.PI);
                ctx.fill();
                // Glow
                ctx.beginPath();
                ctx.fillStyle = 'rgba(255, 51, 102, 0.3)';
                ctx.arc(x, y, 10, 0, 2*Math.PI);
                ctx.fill();
            } else {
                ctx.fillStyle = '#00FFC8';
                ctx.arc(x, y, 3.5, 0, 2*Math.PI);
                ctx.fill();
            }
        }
    }
}

// ─── Word Insertion ─────────────────────────────────────────
async function insertCompletion(word) {
    await apiCall("/complete_word", "POST", { word });
    showToast(`Completed: ${word}`, "success");
}

async function insertWord(word) {
    await apiCall("/insert_word", "POST", { word });
    showToast(`Added: ${word}`, "success");
}

// ─── Actions ────────────────────────────────────────────────
async function resetSentence() {
    await apiCall("/reset", "POST");
    state.recentSigns = [];
    state.lastSentence = '';
    updateRecentSigns();
    dom.sentenceDisplay.innerHTML = '<span class="cursor-blink"></span>';
    dom.sentenceDisplay.classList.add("empty");
    dom.partialWord.textContent = "";
    dom.statSigns.textContent = "0";
    dom.suggestionChips.innerHTML = '<span class="suggestion-placeholder">Show a sign to see suggestions...</span>';
    animatedConfidence = 0;
    showToast("Sentence cleared", "info");
}

async function speakSentence() {
    const result = await apiCall("/speak", "POST");
    if (result && result.status === "ok") {
        if (!result.tts_server) speakText(result.message);
        showToast("Speaking...", "success");
    } else {
        const sentence = dom.sentenceDisplay.textContent.replace('​', '').trim();
        if (sentence) speakText(sentence);
    }
}

async function addSpace() { await apiCall("/add_char", "POST", { char: " " }); }
async function doBackspace() { await apiCall("/backspace", "POST"); }

async function setMode(mode) {
    const result = await apiCall("/mode", "POST", { mode });
    if (result) {
        if (result.status === 'locked') {
            showToast(`Mode locked — wait a moment`, "warning");
            return;
        }
        state.currentMode = result.mode;
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === result.mode);
        });
        showToast(`Mode: ${result.mode.charAt(0).toUpperCase() + result.mode.slice(1)}`, "info");
    }
}

async function toggleAutoSpeak() {
    const result = await apiCall("/auto_speak", "POST");
    if (result) {
        dom.autoSpeakToggle.checked = result.auto_speak;
        showToast(`Auto-speak: ${result.auto_speak ? 'ON' : 'OFF'}`, "info");
    }
}

// ─── Library Browser ────────────────────────────────────────
async function openLibrary() {
    dom.libraryModal.classList.add("visible");
    if (!state.libraryData) {
        const data = await apiCall("/library");
        if (data) {
            state.libraryData = data;
            buildLibraryTabs(data.categories);
            renderLibraryGrid(data.signs);
        }
    }
}

function closeLibrary() { dom.libraryModal.classList.remove("visible"); }

function buildLibraryTabs(categories) {
    let html = '<button class="tab active" data-cat="all">All</button>';
    categories.forEach(cat => {
        html += `<button class="tab" data-cat="${cat.key}">${cat.icon} ${cat.label} (${cat.count})</button>`;
    });
    dom.libraryTabs.innerHTML = html;
    dom.libraryTabs.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            dom.libraryTabs.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            state.activeCategory = tab.dataset.cat;
            filterLibrary();
        });
    });
}

function renderLibraryGrid(signs) {
    if (!signs || signs.length === 0) {
        dom.libraryGrid.innerHTML = '<div class="loading-text">No signs found</div>';
        return;
    }
    dom.libraryGrid.innerHTML = signs.map(sign => `
        <div class="library-card">
            <div class="sign-name">${escapeHtml(sign.label)}</div>
            <div class="sign-type ${sign.type === 'dynamic' ? 'dynamic' : ''}">${sign.type}</div>
            <div class="sign-desc">${escapeHtml(sign.description)}</div>
            <div class="sign-cat">${sign.category_icon} ${escapeHtml(sign.category_label)}</div>
        </div>
    `).join("");
}

function filterLibrary() {
    if (!state.libraryData) return;
    const query = dom.librarySearch.value.toLowerCase().trim();
    const cat = state.activeCategory;
    let signs = state.libraryData.signs;
    if (cat !== 'all') signs = signs.filter(s => s.category === cat);
    if (query) signs = signs.filter(s => s.label.toLowerCase().includes(query) || s.description.toLowerCase().includes(query));
    renderLibraryGrid(signs);
}

// ─── Add Sign ───────────────────────────────────────────────
function openAddSign() { dom.addSignModal.classList.add("visible"); }
function closeAddSign() { dom.addSignModal.classList.remove("visible"); }

async function saveNewSign() {
    const label = dom.newSignLabel.value.trim();
    const description = dom.newSignDescription.value.trim();
    const category = dom.newSignCategory.value;
    if (!label) { showToast("Please enter a sign label", "warning"); return; }
    const result = await apiCall("/add_sign", "POST", { label, description, category });
    if (result && result.status === "ok") {
        showToast(`Sign "${label}" saved!`, "success");
        closeAddSign();
        dom.newSignLabel.value = "";
        dom.newSignDescription.value = "";
        state.libraryData = null;
    } else {
        showToast(result?.message || "Failed to save sign", "error");
    }
}

// ─── Utilities ──────────────────────────────────────────────
function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}
function escapeAttr(text) { return text.replace(/'/g, "\\'").replace(/"/g, '\\"'); }

// ─── Keyboard Shortcuts ─────────────────────────────────────
document.addEventListener("keydown", (e) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
    switch (e.key.toLowerCase()) {
        case "s": if (!state.cameraActive) startCamera(); break;
        case "q": if (state.cameraActive) stopCamera(); break;
        case "r": resetSentence(); break;
        case " ": e.preventDefault(); addSpace(); break;
        case "backspace": e.preventDefault(); doBackspace(); break;
        case "enter": speakSentence(); break;
        case "escape": closeLibrary(); closeAddSign(); break;
    }
});

// ─── Event Listeners ────────────────────────────────────────
dom.btnStartCamera.addEventListener("click", startCamera);
dom.btnStopCamera.addEventListener("click", stopCamera);
dom.btnReset.addEventListener("click", resetSentence);
dom.btnSpeak.addEventListener("click", speakSentence);
dom.btnBackspace.addEventListener("click", doBackspace);
dom.btnSpace.addEventListener("click", addSpace);
dom.btnSpeakSentence.addEventListener("click", speakSentence);
dom.btnClear.addEventListener("click", resetSentence);
dom.btnLibrary.addEventListener("click", openLibrary);
dom.btnAddSign.addEventListener("click", openAddSign);
dom.autoSpeakToggle.addEventListener("change", toggleAutoSpeak);
dom.modeButtons.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => setMode(btn.dataset.mode));
});
dom.libraryClose.addEventListener("click", closeLibrary);
dom.libraryModal.addEventListener("click", (e) => { if (e.target === dom.libraryModal) closeLibrary(); });
dom.librarySearch.addEventListener("input", filterLibrary);
dom.addSignClose.addEventListener("click", closeAddSign);
dom.addSignModal.addEventListener("click", (e) => { if (e.target === dom.addSignModal) closeAddSign(); });
dom.btnSaveSign.addEventListener("click", saveNewSign);

// ─── Init ───────────────────────────────────────────────────
(async function init() {
    const status = await apiCall("/status");
    if (status) {
        state.modelReady = status.model_ready;
        state.currentMode = status.mode || 'auto';
        state.ttsServerAvailable = status.tts_server || false;
        if (status.model_ready) {
            dom.modelStatusIcon.textContent = "✅";
            dom.modelStatusLabel.textContent = "Model Ready";
            dom.modelStatusDetail.textContent = `136+ signs • ${status.mode || 'auto'} mode`;
        }
    }
})();
