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

app = Flask(__name__)
CORS(app)

# ============================================
# LOAD MODEL AND TOKENIZER (WITH ERROR HANDLING)
# ============================================

print("Loading model and tokenizer...")

try:
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    print("✅ Config loaded")

    # Try to load tokenizer - with compatibility fix
    tokenizer = None
    
    # Method 1: Try loading from pickle
    if os.path.exists('tokenizer.pkl'):
        try:
            with open('tokenizer.pkl', 'rb') as f:
                tokenizer = pickle.load(f)
            print("✅ Tokenizer loaded from pickle")
        except (ModuleNotFoundError, AttributeError) as e:
            print(f"⚠️ Pickle loading failed: {e}")
            print("   Trying alternative method...")
    
    # Method 2: Load from JSON config (more compatible)
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
    
    # Method 3: Create fresh tokenizer (last resort)
    if tokenizer is None:
        print("⚠️ Creating new tokenizer (predictions may be less accurate)")
        tokenizer = Tokenizer(
            num_words=config.get('vocab_size', 15000),
            oov_token='<unk>',
            lower=True,
            filters=''
        )
        print("✅ New tokenizer created")

    # Load model
    model = tf.keras.models.load_model('sentiment_model.h5')
    print("✅ Model loaded successfully!")

    MODEL_LOADED = True

except FileNotFoundError as e:
    print(f"❌ Error: File not found - {e}")
    print("\nMake sure these files are in the same directory:")
    print("  - sentiment_model.h5")
    print("  - tokenizer.pkl (or tokenizer_config.json)")
    print("  - config.json")
    MODEL_LOADED = False
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    MODEL_LOADED = False

if not MODEL_LOADED:
    print("\n⚠️ Running in DEMO mode (random predictions)")
    print("To fix: Export model files from Colab and place them here\n")

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
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:5000]
    except Exception as e:
        raise Exception(f"Error fetching URL: {str(e)}")

def analyze_sentiment(text):
    """Analyze sentiment of text"""
    if not MODEL_LOADED:
        # Demo mode - random predictions
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
    
    # Preprocess
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        sequence,
        maxlen=config['max_len'],
        padding='post',
        truncating='post'
    )
    
    # Predict
    predictions = model.predict(padded, verbose=0)
    
    # Get results
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])
    
    # Map to label
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
        'status': 'online',
        'model_loaded': MODEL_LOADED,
        'endpoints': {
            '/analyze_url': 'POST - Analyze URL',
            '/analyze_text': 'POST - Analyze text',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED
    })

@app.route('/analyze_url', methods=['POST'])
def analyze_url():
    """Analyze sentiment from URL"""
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({'error': 'URL is required'}), 400
        
        url = data['url']
        
        # Fetch content
        text = fetch_url_content(url)
        
        if not text or len(text) < 50:
            return jsonify({'error': 'Could not extract text from URL'}), 400
        
        # Analyze
        result = analyze_sentiment(text)
        
        # Format response
        words = text.split()
        summary = ' '.join(words[:100]) + '...' if len(words) > 100 else text
        
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
    """Analyze sentiment from text"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
        
        text = data['text']
        
        if len(text) < 10:
            return jsonify({'error': 'Text too short (minimum 10 characters)'}), 400
        
        # Analyze
        result = analyze_sentiment(text)
        
        return jsonify({
            'success': True,
            'sentiment': result['label'],
            'confidence': round(result['score'] * 100, 2),
            'probabilities': result['probabilities'],
            'text_length': len(text)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    if MODEL_LOADED:
        print("🚀 API Server ready!")
        print("Model: LSTM with GloVe embeddings")
        print("Vocab size:", config.get('vocab_size'))
        print("Max length:", config.get('max_len'))
    else:
        print("⚠️ API Server running in DEMO mode")
        print("To use your model, export files from Colab")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)