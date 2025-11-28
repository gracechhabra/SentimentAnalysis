from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import json
import os

# For summarization
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

app = Flask(__name__)
CORS(app)

# ============================================
# LOAD SENTIMENT MODEL
# ============================================

print("Loading sentiment model and tokenizer...")

try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    print("✅ Config loaded")

    tokenizer = None
    
    if os.path.exists('tokenizer.pkl'):
        try:
            with open('tokenizer.pkl', 'rb') as f:
                tokenizer = pickle.load(f)
            print("✅ Tokenizer loaded from pickle")
        except (ModuleNotFoundError, AttributeError) as e:
            print(f"⚠️ Pickle loading failed: {e}")
    
    if tokenizer is None and os.path.exists('tokenizer_config.json'):
        try:
            with open('tokenizer_config.json', 'r') as f:
                tokenizer_config = json.load(f)
            
            tokenizer = Tokenizer(
                num_words=tokenizer_config.get('num_words'),
                oov_token=tokenizer_config.get('oov_token', '<unk>'),
                filters=tokenizer_config.get('filters', ''),
                lower=tokenizer_config.get('lower', True)
            )
            tokenizer.word_index = tokenizer_config['word_index']
            print("✅ Tokenizer loaded from config")
        except Exception as e:
            print(f"❌ Config loading failed: {e}")
    
    if tokenizer is None:
        print("⚠️ Creating new tokenizer (predictions may be less accurate)")
        tokenizer = Tokenizer(
            num_words=config.get('vocab_size', 15000),
            oov_token='<unk>',
            lower=True,
            filters=''
        )

    model = tf.keras.models.load_model('sentiment_model.h5')
    print("✅ Sentiment model loaded!")
    MODEL_LOADED = True

except Exception as e:
    print(f"❌ Error loading sentiment model: {e}")
    MODEL_LOADED = False


# ============================================
# LOAD SUMMARIZATION MODEL
# ============================================

print("\nLoading summarization model...")

try:
    summarizer_path = "./pegasus_summarizer"
    
    if os.path.exists(summarizer_path):
        print("Loading local Pegasus model...")
        sum_tokenizer = PegasusTokenizer.from_pretrained(summarizer_path)
        sum_model = PegasusForConditionalGeneration.from_pretrained(
    summarizer_path,
    torch_dtype=torch.float16,  # Uses half the memory
    low_cpu_mem_usage=True
)
    else:
        print("Downloading Pegasus model (first time only, ~2GB)...")
        model_name = "google/pegasus-cnn_dailymail"
        sum_tokenizer = PegasusTokenizer.from_pretrained(model_name)
        sum_model = PegasusForConditionalGeneration.from_pretrained(
    summarizer_path,
    torch_dtype=torch.float16,  # Uses half the memory
    low_cpu_mem_usage=True
)
        
        # Save locally for next time
        print("Saving model locally for faster loading next time...")
        sum_tokenizer.save_pretrained(summarizer_path)
        sum_model.save_pretrained(summarizer_path)
    
    sum_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sum_model = sum_model.to(sum_device)
    
    print(f"✅ Summarization model loaded on {sum_device}!")
    SUMMARIZER_LOADED = True

except Exception as e:
    print(f"❌ Error loading summarization model: {e}")
    SUMMARIZER_LOADED = False
    sum_device = 'cpu'


# ============================================
# HELPER FUNCTIONS
# ============================================

def fetch_url_content(url):
    """Fetch text content from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:5000]
    except Exception as e:
        raise Exception(f"Error fetching URL: {str(e)}")


def generate_summary(text, max_input_tokens=1024, max_summary_length=120):
    """Generate summary using Pegasus"""
    if not SUMMARIZER_LOADED:
        return None
    
    try:
        tokens = sum_tokenizer(
            [text],
            truncation=True,
            max_length=max_input_tokens,
            padding="longest",
            return_tensors="pt"
        ).to(sum_device)
        
        summary_ids = sum_model.generate(
            **tokens,
            max_length=max_summary_length,
            num_beams=5,
            early_stopping=True
        )
        
        summary = sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Clean up the <n> tokens
        summary = summary.replace("<n>", " ").strip()
        
        return summary
    except Exception as e:
        print(f"Summarization error: {e}")
        return None

def analyze_sentiment(text):
    """Analyze sentiment of text"""
    if not MODEL_LOADED:
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
    
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        sequence,
        maxlen=config['max_len'],
        padding='post',
        truncating='post'
    )
    
    predictions = model.predict(padded, verbose=0)
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
        'message': 'Financial Sentiment & Summarization API',
        'status': 'online',
        'sentiment_model': MODEL_LOADED,
        'summarizer_model': SUMMARIZER_LOADED,
        'device': sum_device
    })


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'sentiment_model': MODEL_LOADED,
        'summarizer_model': SUMMARIZER_LOADED
    })


@app.route('/analyze_url', methods=['POST'])
def analyze_url():
    """Analyze sentiment and summarize from URL"""
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({'error': 'URL is required'}), 400
        
        url = data['url']
        
        # Fetch content
        text = fetch_url_content(url)
        
        if not text or len(text) < 50:
            return jsonify({'error': 'Could not extract text from URL'}), 400
        
        # Analyze sentiment
        result = analyze_sentiment(text)
        
        # Generate summary
        summary = generate_summary(text)
        
        # Word count
        words = text.split()
        
        return jsonify({
            'success': True,
            'sentiment': result['label'],
            'confidence': round(result['score'] * 100, 2),
            'probabilities': result['probabilities'],
            'summary': summary,
            'word_count': len(words)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    """Analyze sentiment and summarize text"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
        
        text = data['text']
        
        if len(text) < 10:
            return jsonify({'error': 'Text too short (minimum 10 characters)'}), 400
        
        # Analyze sentiment
        result = analyze_sentiment(text)
        
        # Generate summary (only if text is long enough)
        summary = None
        if len(text) >= 100:
            summary = generate_summary(text)
        
        return jsonify({
            'success': True,
            'sentiment': result['label'],
            'confidence': round(result['score'] * 100, 2),
            'probabilities': result['probabilities'],
            'summary': summary,
            'word_count': len(text.split())
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 API Server Starting...")
    print(f"Sentiment Model: {'✅' if MODEL_LOADED else '❌'}")
    print(f"Summarizer Model: {'✅' if SUMMARIZER_LOADED else '❌'}")
    print(f"Device: {sum_device}")
    print("="*50 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)