/**
 * Sign Language Decoder — Frontend JavaScript
 * Handles camera, predictions, word suggestions, library browser,
 * custom signs, mode switching, and all UI interactions.
 */

// ─── State ──────────────────────────────────────────────────
const state = {
    cameraActive: false,
    modelReady: false,
    pollingInterval: null,
    recentSigns: [],
    startTime: null,
    currentMode: 'auto',
    libraryData: null,
    activeCategory: 'all',
};

// ─── DOM Elements ───────────────────────────────────────────
const dom = {
    videoFeed: document.getElementById("videoFeed"),
    videoPlaceholder: document.getElementById("videoPlaceholder"),
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
    // Modals
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
};

// ─── API Helpers ────────────────────────────────────────────
async function apiCall(endpoint, method = "GET", body = null) {
    try {
        const options = {
            method,
            headers: { "Content-Type": "application/json" },
        };
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

// ─── Camera Controls ────────────────────────────────────────
async function startCamera() {
    dom.btnStartCamera.disabled = true;
    dom.btnStartCamera.innerHTML = '<span class="spinner"></span> Starting...';

    const result = await apiCall("/start_camera", "POST");

    if (result && result.status === "ok") {
        state.cameraActive = true;
        state.startTime = Date.now();

        dom.videoFeed.src = "/api/video_feed?" + Date.now();
        dom.videoFeed.style.display = "block";
        dom.videoPlaceholder.classList.add("hidden");
        dom.liveBadge.classList.add("visible");
        dom.predictionOverlay.classList.add("visible");

        dom.statusBadge.classList.add("active");
        dom.statusText.textContent = "Camera Active";

        startPolling();

        dom.btnStartCamera.style.display = "none";
        dom.btnStopCamera.style.display = "inline-flex";

        showToast("Camera started — show your signs!", "success");
    } else {
        showToast("Failed to start camera. Check webcam.", "error");
    }

    dom.btnStartCamera.disabled = false;
    dom.btnStartCamera.innerHTML = '<span class="icon">📷</span> Start Camera';
}

async function stopCamera() {
    const result = await apiCall("/stop_camera", "POST");
    if (result && result.status === "ok") {
        state.cameraActive = false;
        dom.videoFeed.style.display = "none";
        dom.videoFeed.src = "";
        dom.videoPlaceholder.classList.remove("hidden");
        dom.liveBadge.classList.remove("visible");
        dom.predictionOverlay.classList.remove("visible");

        dom.statusBadge.classList.remove("active");
        dom.statusText.textContent = "Camera Inactive";

        stopPolling();

        dom.btnStartCamera.style.display = "inline-flex";
        dom.btnStopCamera.style.display = "none";

        showToast("Camera stopped", "info");
    }
}

// ─── Prediction Polling ─────────────────────────────────────
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

let lastPrediction = "";

function updatePredictionUI(data) {
    const { prediction, confidence, category, sentence, partial_word, model_ready, mode, suggestions } = data;

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
        dom.confidenceBarFill.style.width = `${confidence * 100}%`;

        dom.statPrediction.textContent = prediction;
        dom.statConfidence.textContent = `${(confidence * 100).toFixed(0)}%`;

        if (confidence >= 0.8) {
            dom.statConfidence.className = "stat-value success";
        } else if (confidence >= 0.5) {
            dom.statConfidence.className = "stat-value warning";
        } else {
            dom.statConfidence.className = "stat-value";
        }

        // Pop animation on new detection
        if (prediction !== lastPrediction) {
            dom.predictionLabel.classList.remove("detect-pop");
            void dom.predictionLabel.offsetWidth;
            dom.predictionLabel.classList.add("detect-pop");
            lastPrediction = prediction;
        }

        // Track recent signs
        if (state.recentSigns.length === 0 || state.recentSigns[0].label !== prediction) {
            state.recentSigns.unshift({ label: prediction, category: category || '' });
            if (state.recentSigns.length > 12) state.recentSigns.pop();
            updateRecentSigns();
        }
    } else {
        dom.predictionLabel.textContent = "—";
        dom.predictionConf.textContent = "Waiting for gesture...";
        dom.predictionCategory.textContent = "";
        dom.confidenceBarFill.style.width = "0%";
        dom.statPrediction.textContent = "—";
    }

    // Sentence display
    updateSentenceDisplay(sentence);

    // Partial word indicator
    if (partial_word) {
        dom.partialWord.textContent = `Spelling: ${partial_word}_`;
    } else {
        dom.partialWord.textContent = "";
    }

    // Sign count
    dom.statSigns.textContent = sentence ? sentence.replace(/\s/g, "").length : "0";

    // Session time
    if (state.startTime) {
        const elapsed = Math.floor((Date.now() - state.startTime) / 1000);
        const mins = Math.floor(elapsed / 60);
        const secs = elapsed % 60;
        dom.statTime.textContent = `${mins}:${secs.toString().padStart(2, "0")}`;
    }

    // Update suggestions
    if (suggestions) {
        updateSuggestions(suggestions);
    }
}

function updateSentenceDisplay(sentence) {
    if (sentence && sentence.length > 0) {
        dom.sentenceDisplay.innerHTML = escapeHtml(sentence) + '<span class="cursor-blink"></span>';
        dom.sentenceDisplay.classList.remove("empty");
    } else {
        dom.sentenceDisplay.innerHTML = '<span class="cursor-blink"></span>';
        dom.sentenceDisplay.classList.add("empty");
    }
}

function updateRecentSigns() {
    dom.recentSignsContainer.innerHTML = state.recentSigns
        .map(s => {
            const isWord = s.label.length > 1 && !['SPACE','DELETE'].includes(s.label);
            const cls = isWord ? 'sign-chip word' : 'sign-chip';
            return `<span class="${cls}">${escapeHtml(s.label)}</span>`;
        })
        .join("");
}

// ─── Suggestions ────────────────────────────────────────────
function updateSuggestions(suggestions) {
    const chips = [];

    // Word completions
    if (suggestions.completions && suggestions.completions.length > 0) {
        suggestions.completions.slice(0, 4).forEach(word => {
            chips.push(`<span class="suggestion-chip" onclick="insertCompletion('${escapeAttr(word)}')">${escapeHtml(word)}</span>`);
        });
    }

    // Spell corrections
    if (suggestions.corrections && suggestions.corrections.length > 0) {
        suggestions.corrections.slice(0, 2).forEach(word => {
            chips.push(`<span class="suggestion-chip correction" onclick="insertCompletion('${escapeAttr(word)}')" title="Did you mean?">✏️ ${escapeHtml(word)}</span>`);
        });
    }

    // Next-word suggestions
    if (suggestions.next_words && suggestions.next_words.length > 0) {
        suggestions.next_words.slice(0, 3).forEach(word => {
            chips.push(`<span class="suggestion-chip phrase" onclick="insertWord('${escapeAttr(word)}')" title="Next word">→ ${escapeHtml(word)}</span>`);
        });
    }

    if (chips.length > 0) {
        dom.suggestionChips.innerHTML = chips.join("");
    } else {
        dom.suggestionChips.innerHTML = '<span class="suggestion-placeholder">Show a sign to see suggestions...</span>';
    }
}

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
    updateRecentSigns();
    dom.sentenceDisplay.innerHTML = '<span class="cursor-blink"></span>';
    dom.sentenceDisplay.classList.add("empty");
    dom.partialWord.textContent = "";
    dom.statSigns.textContent = "0";
    dom.suggestionChips.innerHTML = '<span class="suggestion-placeholder">Show a sign to see suggestions...</span>';
    showToast("Sentence cleared", "info");
}

async function speakSentence() {
    dom.btnSpeak.disabled = true;
    dom.btnSpeak.innerHTML = '<span class="spinner"></span> Speaking...';

    const result = await apiCall("/speak", "POST");
    if (result && result.status === "ok") {
        showToast("Speaking output...", "success");
    } else {
        showToast(result?.message || "Nothing to speak", "error");
    }

    setTimeout(() => {
        dom.btnSpeak.disabled = false;
        dom.btnSpeak.innerHTML = '<span class="icon">🔊</span> Speak';
    }, 1500);
}

async function addSpace() { await apiCall("/add_char", "POST", { char: " " }); }
async function doBackspace() { await apiCall("/backspace", "POST"); }

// ─── Mode Switching ─────────────────────────────────────────
async function setMode(mode) {
    const result = await apiCall("/mode", "POST", { mode });
    if (result) {
        state.currentMode = result.mode;
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === result.mode);
        });
        showToast(`Mode: ${result.mode.charAt(0).toUpperCase() + result.mode.slice(1)}`, "info");
    }
}

// ─── Auto-Speak Toggle ─────────────────────────────────────
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

function closeLibrary() {
    dom.libraryModal.classList.remove("visible");
}

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

    if (cat !== 'all') {
        signs = signs.filter(s => s.category === cat);
    }
    if (query) {
        signs = signs.filter(s =>
            s.label.toLowerCase().includes(query) ||
            s.description.toLowerCase().includes(query)
        );
    }

    renderLibraryGrid(signs);
}

// ─── Add Sign ───────────────────────────────────────────────
function openAddSign() {
    dom.addSignModal.classList.add("visible");
}

function closeAddSign() {
    dom.addSignModal.classList.remove("visible");
}

async function saveNewSign() {
    const label = dom.newSignLabel.value.trim();
    const description = dom.newSignDescription.value.trim();
    const category = dom.newSignCategory.value;

    if (!label) {
        showToast("Please enter a sign label", "warning");
        return;
    }

    const result = await apiCall("/add_sign", "POST", { label, description, category });
    if (result && result.status === "ok") {
        showToast(`Sign "${label}" saved!`, "success");
        closeAddSign();
        dom.newSignLabel.value = "";
        dom.newSignDescription.value = "";
        state.libraryData = null; // Force refresh
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

function escapeAttr(text) {
    return text.replace(/'/g, "\\'").replace(/"/g, '\\"');
}

// ─── Keyboard Shortcuts ─────────────────────────────────────
document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;

    switch (e.key.toLowerCase()) {
        case "s":
            if (!state.cameraActive) startCamera();
            break;
        case "q":
            if (state.cameraActive) stopCamera();
            break;
        case "r":
            resetSentence();
            break;
        case " ":
            e.preventDefault();
            addSpace();
            break;
        case "backspace":
            e.preventDefault();
            doBackspace();
            break;
        case "enter":
            speakSentence();
            break;
        case "escape":
            closeLibrary();
            closeAddSign();
            break;
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

// Mode buttons
dom.modeButtons.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => setMode(btn.dataset.mode));
});

// Library modal
dom.libraryClose.addEventListener("click", closeLibrary);
dom.libraryModal.addEventListener("click", (e) => {
    if (e.target === dom.libraryModal) closeLibrary();
});
dom.librarySearch.addEventListener("input", filterLibrary);

// Add sign modal
dom.addSignClose.addEventListener("click", closeAddSign);
dom.addSignModal.addEventListener("click", (e) => {
    if (e.target === dom.addSignModal) closeAddSign();
});
dom.btnSaveSign.addEventListener("click", saveNewSign);

// ─── Initial Status Check ───────────────────────────────────
(async function init() {
    const status = await apiCall("/status");
    if (status) {
        state.modelReady = status.model_ready;
        state.currentMode = status.mode || 'auto';
        if (status.model_ready) {
            dom.modelStatusIcon.textContent = "✅";
            dom.modelStatusLabel.textContent = "Model Ready";
            dom.modelStatusDetail.textContent = `136+ signs • ${status.mode || 'auto'} mode`;
        }
    }
})();
