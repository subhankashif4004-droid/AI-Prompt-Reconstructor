/**
 * 🚀 AI Prompt Reconstructor & Enhancer
 * Frontend Logic - Clean & Modern
 */


// DOM ELEMENTS
const youtubeUrl = document.getElementById('youtubeUrl');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const loadingSection = document.getElementById('loadingSection');
const errorSection = document.getElementById('errorSection');
const resultsSection = document.getElementById('resultsSection');
const errorMessage = document.getElementById('errorMessage');
const loadingText = document.getElementById('loadingText');
const geminiApiKey = document.getElementById('geminiApiKey');
const verifyApiKeyBtn = document.getElementById('verifyApiKeyBtn');
const apiStatusIcon = document.getElementById('apiStatusIcon');


// EVENT LISTENERS
analyzeBtn.addEventListener('click', startAnalysis);
clearBtn.addEventListener('click', () => {
    youtubeUrl.value = '';
    youtubeUrl.focus();
});

youtubeUrl.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') startAnalysis();
});

verifyApiKeyBtn.addEventListener('click', validateApiKey);

geminiApiKey.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') validateApiKey();
});


// MAIN ANALYSIS FUNCTION
async function startAnalysis() {
    const url = youtubeUrl.value.trim();
    if (!url) {
        shakeInput();
        return;
    }

    // Reset UI
    hideAll();
    showLoading();
    setButtonLoading(true);

    try {
        // Animate loading steps
        animateLoadingSteps();

        const response = await fetch('/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                url: url,
                api_key: geminiApiKey.value.trim()
            })
        });

        const data = await response.json();

        if (data.success) {
            hideLoading();
            displayResults(data);
        } else {
            hideLoading();
            showError(data.error);
        }
    } catch (error) {
        hideLoading();
        showError('Network error. Please check your connection and try again.');
        console.error('Analysis error:', error);
    } finally {
        setButtonLoading(false);
    }
}

// ============================================
// DISPLAY RESULTS
// ============================================
function displayResults(data) {
    resultsSection.style.display = 'flex';

    // Video Info
    const videoId = data.video_id;
    document.getElementById('videoThumbnail').src = 
        `https://img.youtube.com/vi/${videoId}/mqdefault.jpg`;
    document.getElementById('videoThumbnail').parentElement.onclick = () => {
        window.open(`https://www.youtube.com/watch?v=${videoId}`, '_blank');
    };

    document.getElementById('transcriptLength').textContent = 
        formatNumber(data.transcript_length);
    document.getElementById('segmentCount').textContent = 
        data.segments || '—';
    document.getElementById('contentType').textContent = 
        data.analysis.content_type || '—';
    document.getElementById('complexity').textContent = 
        data.analysis.complexity || '—';
    document.getElementById('aiModeResult').textContent = data.ai_mode;
    document.getElementById('transcriptPreview').textContent = 
        data.transcript_preview;

    // Topics
    const topicsList = document.getElementById('topicsList');
    topicsList.innerHTML = '';
    const topics = data.analysis.main_topics || [];
    topics.forEach(topic => {
        const tag = document.createElement('span');
        tag.className = 'topic-tag';
        tag.textContent = typeof topic === 'string' ? topic : topic.name;
        topicsList.appendChild(tag);
    });

    // Reconstructed Prompts
    const promptsContainer = document.getElementById('reconstructedPrompts');
    promptsContainer.innerHTML = '';
    document.getElementById('promptCount').textContent = 
        data.reconstructed_prompts.length;

    data.reconstructed_prompts.forEach((prompt, index) => {
        promptsContainer.appendChild(createPromptCard(prompt, index));
    });

    // Enhanced Prompts
    const enhancedContainer = document.getElementById('enhancedPrompts');
    enhancedContainer.innerHTML = '';

    data.enhanced_prompts.forEach((prompt, index) => {
        enhancedContainer.appendChild(createEnhancedCard(prompt, index));
    });

    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 200);
}

// ============================================
// CREATE PROMPT CARD
// ============================================
function createPromptCard(prompt, index) {
    const card = document.createElement('div');
    card.className = 'prompt-item';
    card.style.animationDelay = `${index * 0.1}s`;

    const confidence = prompt.confidence || 80;
    const id = `prompt-${index}`;

    card.innerHTML = `
        <div class="prompt-item-header">
            <span class="prompt-type">${prompt.type}</span>
            <div class="confidence-bar">
                <span class="confidence-label">Confidence</span>
                <div class="confidence-track">
                    <div class="confidence-fill" 
                         style="width: 0%" 
                         data-width="${confidence}%"></div>
                </div>
                <span class="confidence-value">${confidence}%</span>
            </div>
        </div>
        <div class="prompt-text">${escapeHtml(prompt.prompt)}</div>
        <div class="prompt-actions">
            <button class="copy-btn" onclick="copyText(this, '${id}')">
                <i class="fas fa-copy"></i> Copy Prompt
            </button>
        </div>
        <textarea id="${id}" style="display:none">${prompt.prompt}</textarea>
    `;

    // Animate confidence bar
    setTimeout(() => {
        const fill = card.querySelector('.confidence-fill');
        fill.style.width = fill.dataset.width;
    }, 300 + index * 150);

    return card;
}

// ============================================
// CREATE ENHANCED CARD
// ============================================
function createEnhancedCard(prompt, index) {
    const card = document.createElement('div');
    card.className = 'enhanced-item';
    card.style.animationDelay = `${index * 0.1}s`;

    const id = `enhanced-${index}`;
    const score = prompt.improvement_score || 95;

    card.innerHTML = `
        <div class="enhanced-header">
            <span class="enhanced-type">${prompt.type} ⚡</span>
            <span class="improvement-score">
                <i class="fas fa-arrow-up"></i> ${score}% Quality Score
            </span>
        </div>
        <div class="original-prompt">
            <span class="original-label">Original Prompt</span>
            ${escapeHtml(prompt.original)}
        </div>
        <div class="enhanced-prompt-text">${escapeHtml(prompt.enhanced)}</div>
        <div class="prompt-actions">
            <button class="copy-btn" onclick="copyText(this, '${id}')">
                <i class="fas fa-copy"></i> Copy Enhanced Prompt
            </button>
            <button class="copy-btn" onclick="copyText(this, '${id}-orig')">
                <i class="fas fa-file-alt"></i> Copy Original
            </button>
        </div>
        <textarea id="${id}" style="display:none">${prompt.enhanced}</textarea>
        <textarea id="${id}-orig" style="display:none">${prompt.original}</textarea>
    `;

    return card;
}

// ============================================
// UTILITY FUNCTIONS
// ============================================
function copyText(button, textId) {
    const textarea = document.getElementById(textId);
    if (!textarea) return;

    navigator.clipboard.writeText(textarea.value).then(() => {
        const originalHTML = button.innerHTML;
        button.innerHTML = '<i class="fas fa-check"></i> Copied!';
        button.classList.add('copied');

        setTimeout(() => {
            button.innerHTML = originalHTML;
            button.classList.remove('copied');
        }, 2000);
    }).catch(() => {
        // Fallback
        textarea.style.display = 'block';
        textarea.select();
        document.execCommand('copy');
        textarea.style.display = 'none';

        const originalHTML = button.innerHTML;
        button.innerHTML = '<i class="fas fa-check"></i> Copied!';
        button.classList.add('copied');

        setTimeout(() => {
            button.innerHTML = originalHTML;
            button.classList.remove('copied');
        }, 2000);
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatNumber(num) {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toString();
}

function shakeInput() {
    youtubeUrl.style.animation = 'shake 0.5s ease-in-out';
    youtubeUrl.style.borderColor = 'var(--danger)';
    setTimeout(() => {
        youtubeUrl.style.animation = '';
        youtubeUrl.style.borderColor = '';
    }, 500);
}

// ============================================
// UI STATE MANAGEMENT
// ============================================
function hideAll() {
    loadingSection.style.display = 'none';
    errorSection.style.display = 'none';
    resultsSection.style.display = 'none';
}

function showLoading() {
    loadingSection.style.display = 'block';
    // Reset steps
    document.querySelectorAll('.loading-steps .step').forEach(step => {
        step.classList.remove('active', 'done');
    });
    document.getElementById('step1').classList.add('active');
}

function hideLoading() {
    loadingSection.style.display = 'none';
}

function showError(message) {
    errorSection.style.display = 'block';
    errorMessage.textContent = message;
}

function setButtonLoading(loading) {
    analyzeBtn.disabled = loading;
    const content = analyzeBtn.querySelector('.btn-content');
    const loadingEl = analyzeBtn.querySelector('.btn-loading');

    if (loading) {
        content.style.display = 'none';
        loadingEl.style.display = 'flex';
    } else {
        content.style.display = 'flex';
        loadingEl.style.display = 'none';
    }
}

function resetUI() {
    hideAll();
    youtubeUrl.focus();
}

// ============================================
// API KEY VALIDATION
// ============================================
async function validateApiKey() {
    const key = geminiApiKey.value.trim();
    if (!key) return;

    // UI Feedback
    verifyApiKeyBtn.disabled = true;
    verifyApiKeyBtn.classList.add('loading');
    verifyApiKeyBtn.innerHTML = '<div class="spinner" style="width:12px; height:12px"></div> Validating...';
    apiStatusIcon.className = 'api-status-icon'; // Reset

    try {
        const response = await fetch('/validate_api_key', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ api_key: key })
        });

        const data = await response.json();

        if (data.success) {
            // Valid Key
            apiStatusIcon.innerHTML = '<i class="fas fa-check-circle"></i>';
            apiStatusIcon.className = 'api-status-icon valid';
            geminiApiKey.parentElement.classList.add('locked');
            localStorage.setItem('gemini_api_key', key);
            localStorage.setItem('gemini_api_verified', 'true');
            
            // Update the badge
            const badge = document.getElementById('aiModeBadge');
            if (badge) badge.textContent = 'GEMINI Engine';
        } else {
            // Invalid Key
            apiStatusIcon.innerHTML = '<i class="fas fa-times-circle"></i>';
            apiStatusIcon.className = 'api-status-icon invalid';
            shakeElement(geminiApiKey);
            alert(data.error);
        }
    } catch (error) {
        console.error('Validation error:', error);
        alert('Network error while validating API key.');
    } finally {
        verifyApiKeyBtn.disabled = false;
        verifyApiKeyBtn.classList.remove('loading');
        verifyApiKeyBtn.innerHTML = '<i class="fas fa-check-circle"></i> Verify';
    }
}

function shakeElement(el) {
    el.style.animation = 'shake 0.5s ease-in-out';
    el.style.borderColor = 'var(--danger)';
    setTimeout(() => {
        el.style.animation = '';
        el.style.borderColor = '';
    }, 500);
}

// ============================================
// LOADING STEP ANIMATION
// ============================================
function animateLoadingSteps() {
    const steps = ['step1', 'step2', 'step3', 'step4'];
    const messages = [
        'Fetching transcript from YouTube...',
        'Analyzing content with AI...',
        'Reconstructing probable prompts...',
        'Enhancing prompts to pro quality...'
    ];

    let currentStep = 0;

    function advanceStep() {
        if (currentStep > 0) {
            document.getElementById(steps[currentStep - 1]).classList.remove('active');
            document.getElementById(steps[currentStep - 1]).classList.add('done');
        }

        if (currentStep < steps.length) {
            document.getElementById(steps[currentStep]).classList.add('active');
            loadingText.textContent = messages[currentStep];
            currentStep++;
            setTimeout(advanceStep, 1200);
        }
    }

    advanceStep();
}

// ============================================
// CSS ANIMATION FOR SHAKE
// ============================================
const style = document.createElement('style');
style.textContent = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        20% { transform: translateX(-8px); }
        40% { transform: translateX(8px); }
        60% { transform: translateX(-4px); }
        80% { transform: translateX(4px); }
    }
`;
document.head.appendChild(style);

// ============================================
// KEYBOARD SHORTCUT
// ============================================
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to analyze
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        startAnalysis();
    }
    // Escape to reset
    if (e.key === 'Escape') {
        resetUI();
    }
});

// ============================================
// AUTO-FOCUS & STORAGE ON LOAD
// ============================================
window.addEventListener('load', () => {
    // Load API Key from storage
    const savedKey = localStorage.getItem('gemini_api_key');
    const isVerified = localStorage.getItem('gemini_api_verified') === 'true';

    if (savedKey) {
        geminiApiKey.value = savedKey;
        if (isVerified) {
            geminiApiKey.parentElement.classList.add('locked');
            apiStatusIcon.innerHTML = '<i class="fas fa-check-circle"></i>';
            apiStatusIcon.className = 'api-status-icon valid';
        }
    }

    setTimeout(() => youtubeUrl.focus(), 500);
});

// Save API Key on change (reset verification)
geminiApiKey.addEventListener('input', () => {
    geminiApiKey.parentElement.classList.remove('locked');
    apiStatusIcon.className = 'api-status-icon';
    localStorage.setItem('gemini_api_verified', 'false');
});