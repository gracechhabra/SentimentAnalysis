# Sentiment Analysis

A web application for analyzing sentiment in text using machine learning.

## Setup Instructions

### 1. Create a Virtual Environment
```bash
python -m venv venv
```

### 2. Activate the Virtual Environment
**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the API Server
Navigate to the BackEnd folder and run the following command: 
To navigate to the BackEnd folder, use the following command:
```bash
cd BackEnd
```

Then write the following command: 
```bash
python simple_api.py
```
If running the summarization model with sentiment analysis, run by going into the BackEnd folder:
```bash
python api_with_summarization.py
```

### 5. Run the Frontend
  1. Download the live server on VS code
  2. Go to index.html in FronEnd folder and click Go Live. This will open a website

## Usage
Once both the API server and frontend are running, open your browser to `http://localhost:3000` (or the appropriate port).

## Technologies Used
- Python
- LSTM with Glove embeddings
- Peagsus model


Note: The peagsus model will take 5-10 minutes to download on the first run. 

