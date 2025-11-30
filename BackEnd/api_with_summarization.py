from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import pickle
import json
import os
import re

app = Flask(__name__)
CORS(app)

print("="*60)
print("STARTING FINANCIAL SENTIMENT ANALYSIS API")
print("="*60)

# ============================================
# LOAD SENTIMENT ANALYSIS MODEL (LSTM)
# ============================================

print("\n[1/2] Loading Sentiment Analysis Model...")

try:
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    print("   ✅ Config loaded")

    # Load tokenizer
    tokenizer = None
    
    if os.path.exists('tokenizer.pkl'):
        try:
            with open('tokenizer.pkl', 'rb') as f:
                tokenizer = pickle.load(f)
            print("   ✅ Tokenizer loaded from pickle")
        except (ModuleNotFoundError, AttributeError) as e:
            print(f"   ⚠️  Pickle failed, trying config...")
    
    if tokenizer is None and os.path.exists('tokenizer_config.json'):
        with open('tokenizer_config.json', 'r') as f:
            tokenizer_config = json.load(f)
        
        tokenizer = Tokenizer(
            num_words=tokenizer_config.get('num_words'),
            oov_token=tokenizer_config.get('oov_token', '<unk>'),
            filters=tokenizer_config.get('filters', ''),
            lower=tokenizer_config.get('lower', True)
        )
        tokenizer.word_index = tokenizer_config['word_index']
        print("   ✅ Tokenizer loaded from config")
    
    if tokenizer is None:
        tokenizer = Tokenizer(
            num_words=config.get('vocab_size', 15000),
            oov_token='<unk>',
            lower=True,
            filters=''
        )
        print("   ⚠️  Using new tokenizer")

    # Load LSTM model
    sentiment_model = tf.keras.models.load_model('sentiment_model.h5')
    print("   ✅ LSTM model loaded")
    
    SENTIMENT_MODEL_LOADED = True
    print("   ✅ SENTIMENT ANALYSIS: READY")

except Exception as e:
    print(f"   ❌ Error: {e}")
    SENTIMENT_MODEL_LOADED = False
    print("   ⚠️  SENTIMENT ANALYSIS: DEMO MODE")

# ============================================
# LOAD SUMMARIZATION MODEL (PEGASUS)
# ============================================

print("\n[2/2] Loading Summarization Model (Pegasus)...")

try:
    #model_name = "google/pegasus-cnn_dailymail"
    model_name = "sshleifer/distilbart-cnn-12-6"
    pegasus_tokenizer = BartTokenizer.from_pretrained(model_name)
    pegasus_model = BartForConditionalGeneration.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"   Device: {device}")
    print("   Note: First run downloads ~2GB (takes 5-10 min)")
    print("   Downloading model...")
    
    #pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)
    #pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    
    SUMMARIZATION_LOADED = True
    print("   ✅ Pegasus model loaded")
    print("   ✅ SUMMARIZATION: READY (AI-powered)")
    
except Exception as e:
    print(f"   ⚠️  Error: {e}")
    SUMMARIZATION_LOADED = False
    print("   ⚠️  SUMMARIZATION: Simple truncation mode")

# ============================================
# HELPER FUNCTIONS
# ============================================

def extract_main_content(soup):
    """Extract main article content from webpage"""
    
    # Remove unwanted elements
    for element in soup(["script", "style", "nav", "footer", "header", "aside", 
                         "iframe", "noscript", "svg", "form"]):
        element.decompose()
    
    main_content = None
    
    # Try common article selectors
    article_selectors = [
        'article', '[role="main"]', '.article-body', '.article-content',
        '.post-content', '.entry-content', '.story-body', '.content-body',
        'main', '#main-content', '.main-content'
    ]
    
    for selector in article_selectors:
        main_content = soup.select_one(selector)
        if main_content:
            break
    
    if not main_content:
        main_content = soup.find('body')
    
    if not main_content:
        return None
    
    # Extract text from paragraphs
    paragraphs = main_content.find_all('p')
    
    if paragraphs:
        text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
    else:
        text = main_content.get_text()
    
    # Clean up
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = text.strip()
    
    return text


def fetch_url_content(url):
    """Fetch and extract clean text content from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        text = extract_main_content(soup)
        
        if not text:
            raise Exception("Could not extract text from webpage")
        
        return text[:5000]
        
    except Exception as e:
        raise Exception(f"Error fetching URL: {str(e)}")


def summarize_text(text, max_length=120):
    """Generate summary using Pegasus AI or fallback to truncation"""
    try:
        if not SUMMARIZATION_LOADED:
            # Fallback: simple truncation
            words = text.split()[:150]
            return ' '.join(words) + '...'
        
        # AI-powered summary with Pegasus
        tokens = pegasus_tokenizer(
            [text],
            truncation=True,
            max_length=1024,
            padding="longest",
            return_tensors="pt"
        ).to(device)
        
        summary_ids = pegasus_model.generate(
            **tokens,
            max_length=max_length,
            num_beams=5,
            early_stopping=True
        )
        
        summary = pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summary = summary.replace('<n>', ' ')
        summary = summary.replace('\n', ' ')
        summary = ' '.join(summary.split())
        summary = summary.strip()
        return summary
        
    except Exception as e:
        print(f"⚠️ Summarization error: {e}")
        # Fallback
        words = text.split()[:150]
        return ' '.join(words) + '...'


def analyze_sentiment(text):
    """Analyze sentiment using LSTM model"""
    if not SENTIMENT_MODEL_LOADED:
        # Demo mode
        import random
        sentiments = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
        sentiment = random.choice(sentiments)
        score = random.uniform(0.6, 0.99)
        
        return {
            'label': sentiment,
            'score': score,
            'probabilities': {
                'negative': random.uniform(0.1, 0.4),
                'neutral': random.uniform(0.1, 0.4),
                'positive': random.uniform(0.1, 0.4)
            }
        }
    
    # Preprocess text
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        sequence,
        maxlen=config['max_len'],
        padding='post',
        truncating='post'
    )
    
    # Predict
    predictions = sentiment_model.predict(padded, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])
    label = config['label_mapping'][str(predicted_class)]
    
    return {
        'label': label.upper(),
        'score': confidence,
        'probabilities': {
            'negative': float(predictions[0][0]),
            'neutral': float(predictions[0][1]),
            'positive': float(predictions[0][2])
        }
    }

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/')
def home():
    return jsonify({
        'message': 'Financial Sentiment Analysis API',
        'version': '2.0',
        'features': {
            'sentiment_analysis': SENTIMENT_MODEL_LOADED,
            'ai_summarization': SUMMARIZATION_LOADED
        },
        'endpoints': {
            '/analyze_url': 'POST - Analyze URL with AI summary',
            '/analyze_text': 'POST - Analyze raw text',
            '/health': 'GET - Health check'
        }
    })


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'sentiment_model': SENTIMENT_MODEL_LOADED,
        'summarization_model': SUMMARIZATION_LOADED
    })


@app.route('/analyze_url', methods=['POST'])
def analyze_url():
    """Analyze sentiment from URL with AI-generated summary"""
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({'error': 'URL is required'}), 400
        
        url = data['url']
        print(f"\n📥 Analyzing: {url}")
        
        # Fetch content
        print("   Fetching webpage...")
        text = fetch_url_content(url)
        
        if not text or len(text) < 50:
            return jsonify({'error': 'Could not extract text from URL'}), 400
        
        print(f"   Extracted {len(text)} characters")
        
        # Generate AI summary
        print("   Generating summary...")
        summary = summarize_text(text, max_length=120)
        print(f"   Summary: {summary[:80]}...")
        
        # Analyze sentiment
        print("   Analyzing sentiment...")
        result = analyze_sentiment(text)
        print(f"   Result: {result['label']} ({result['score']:.2%})")
        
        return jsonify({
            'success': True,
            'sentiment': result['label'],
            'confidence': round(result['score'] * 100, 2),
            'probabilities': result['probabilities'],
            'summary': summary,
            'word_count': len(text.split()),
            'summarization_type': 'AI-powered' if SUMMARIZATION_LOADED else 'Simple truncation'
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    """Analyze sentiment from raw text"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
        
        text = data['text']
        
        if len(text) < 10:
            return jsonify({'error': 'Text too short (minimum 10 characters)'}), 400
        
        print(f"\n📝 Analyzing text ({len(text)} chars)")
        
        # Analyze sentiment
        result = analyze_sentiment(text)
        
        response = {
            'success': True,
            'sentiment': result['label'],
            'confidence': round(result['score'] * 100, 2),
            'probabilities': result['probabilities'],
            'text_length': len(text)
        }
        
        # Optionally summarize if text is long
        if len(text) > 500:
            summary = summarize_text(text, max_length=100)
            response['summary'] = summary
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================
# STARTUP
# ============================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("API SERVER STATUS")
    print("="*60)
    
    if SENTIMENT_MODEL_LOADED:
        print("✅ Sentiment Analysis: READY")
        print(f"   Model: BiLSTM + GloVe")
        print(f"   Vocab: {config.get('vocab_size')}")
        print(f"   Max Length: {config.get('max_len')}")
    else:
        print("⚠️  Sentiment Analysis: DEMO MODE")
    
    if SUMMARIZATION_LOADED:
        print("✅ AI Summarization: READY")
        print(f"   Model: Pegasus (Google)")
        print(f"   Device: {device}")
    else:
        print("⚠️  AI Summarization: FALLBACK MODE")
    
    print("="*60)
    print("🚀 Starting Flask server...")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)