// API Configuration
const API_URL = 'http://localhost:5000';  // Change this to your deployed API URL

// Tab Switching
function switchTab(tab) {
    // Remove active class from all tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });

    // Add active class to selected tab
    event.target.classList.add('active');
    document.getElementById(tab + '-tab').classList.add('active');

    // Hide results when switching tabs
    hideResults();
}

// Set URL from example
function setURL(url) {
    document.getElementById('url-input').value = url;
}

// Show/Hide Loading State
function setLoading(type, isLoading) {
    const btnText = document.getElementById(type + '-btn-text');
    const loading = document.getElementById(type + '-loading');
    const btn = event.target;

    if (isLoading) {
        btnText.classList.add('hidden');
        loading.classList.remove('hidden');
        btn.disabled = true;
    } else {
        btnText.classList.remove('hidden');
        loading.classList.add('hidden');
        btn.disabled = false;
    }
}

// Show Error
function showError(message) {
    const errorDiv = document.getElementById('error-message');
    errorDiv.textContent = '❌ ' + message;
    errorDiv.classList.remove('hidden');
    
    // Hide after 5 seconds
    setTimeout(() => {
        errorDiv.classList.add('hidden');
    }, 5000);
}

// Hide Results
function hideResults() {
    document.getElementById('results').classList.add('hidden');
    document.getElementById('error-message').classList.add('hidden');
}

// Display Results
function displayResults(data) {
    console.log("=== displayResults called ===");
    console.log("Data received:", data);
    console.log("Results div:", document.getElementById('results'));
    const resultsDiv = document.getElementById('results');
    const sentimentBadge = document.getElementById('sentiment-badge');
    const sentimentIcon = document.getElementById('sentiment-icon');
    const sentimentLabel = document.getElementById('sentiment-label');
    const sentimentConfidence = document.getElementById('sentiment-confidence');

    // Get sentiment
    const sentiment = data.sentiment.toLowerCase();
    
    // Update badge
    sentimentBadge.className = 'sentiment-badge ' + sentiment;
    
    // Update icon
    const icons = {
        'positive': '✅',
        'neutral': '➖',
        'negative': '❌'
    };
    sentimentIcon.textContent = icons[sentiment] || '📊';
    
    // Update label and confidence
    sentimentLabel.textContent = data.sentiment;
    sentimentConfidence.textContent = `Confidence: ${data.confidence}%`;

    // Update probabilities
    updateProbabilities(data.probabilities);

    // Update summary if available
    if (data.summary) {
        document.getElementById('summary-section').classList.remove('hidden');
        document.getElementById('summary-text').textContent = data.summary;
        document.getElementById('word-count').textContent = data.word_count || 'N/A';
    } else {
        document.getElementById('summary-section').classList.add('hidden');
    }

    // Show results
    resultsDiv.classList.remove('hidden');
    
    // Scroll to results
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Update Probability Bars
function updateProbabilities(probs) {
    // Convert to percentages
    const positive = (probs.positive * 100).toFixed(1);
    const neutral = (probs.neutral * 100).toFixed(1);
    const negative = (probs.negative * 100).toFixed(1);

    // Update bars
    document.getElementById('prob-positive').style.width = positive + '%';
    document.getElementById('prob-neutral').style.width = neutral + '%';
    document.getElementById('prob-negative').style.width = negative + '%';

    // Update text
    document.getElementById('prob-positive-text').textContent = positive + '%';
    document.getElementById('prob-neutral-text').textContent = neutral + '%';
    document.getElementById('prob-negative-text').textContent = negative + '%';
}

// Analyze URL
async function analyzeURL() {
    const urlInput = document.getElementById('url-input');
    const url = urlInput.value.trim();

    // Validate input
    if (!url) {
        showError('Please enter a URL');
        return;
    }

    if (!url.startsWith('http://') && !url.startsWith('https://')) {
        showError('Please enter a valid URL (must start with http:// or https://)');
        return;
    }

    hideResults();

    try {
        // Show loading
        const analyzeBtn = document.querySelector('#url-tab .analyze-btn');
        const btnText = document.getElementById('url-btn-text');
        const loading = document.getElementById('url-loading');
        
        btnText.classList.add('hidden');
        loading.classList.remove('hidden');
        analyzeBtn.disabled = true;

        // Make API call
        const response = await fetch(`${API_URL}/analyze_url`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: url })
        });

        const data = await response.json();

        // Hide loading
        btnText.classList.remove('hidden');
        loading.classList.add('hidden');
        analyzeBtn.disabled = false;

        if (response.ok && data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Failed to analyze URL');
        }

    } catch (error) {
        // Hide loading
        const analyzeBtn = document.querySelector('#url-tab .analyze-btn');
        const btnText = document.getElementById('url-btn-text');
        const loading = document.getElementById('url-loading');
        
        btnText.classList.remove('hidden');
        loading.classList.add('hidden');
        analyzeBtn.disabled = false;

        console.error('Error:', error);
        showError('Network error. Please make sure the API server is running.');
    }
}

// Analyze Text
async function analyzeText() {
    const textInput = document.getElementById('text-input');
    const text = textInput.value.trim();

    // Validate input
    if (!text) {
        showError('Please enter some text');
        return;
    }

    if (text.length < 10) {
        showError('Text is too short (minimum 10 characters)');
        return;
    }

    hideResults();

    try {
        // Show loading
        const analyzeBtn = document.querySelector('#text-tab .analyze-btn');
        const btnText = document.getElementById('text-btn-text');
        const loading = document.getElementById('text-loading');
        
        btnText.classList.add('hidden');
        loading.classList.remove('hidden');
        analyzeBtn.disabled = true;

        // Make API call
        const response = await fetch(`${API_URL}/analyze_text`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        // Hide loading
        btnText.classList.remove('hidden');
        loading.classList.add('hidden');
        analyzeBtn.disabled = false;

        if (response.ok && data.success) {
            displayResults(data);
        } else {
            showError(data.error || 'Failed to analyze text');
        }

    } catch (error) {
        // Hide loading
        const analyzeBtn = document.querySelector('#text-tab .analyze-btn');
        const btnText = document.getElementById('text-btn-text');
        const loading = document.getElementById('text-loading');
        
        btnText.classList.remove('hidden');
        loading.classList.add('hidden');
        analyzeBtn.disabled = false;

        console.error('Error:', error);
        showError('Network error. Please make sure the API server is running.');
    }
}

// Allow Enter key to submit
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('url-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            analyzeURL();
        }
    });
});