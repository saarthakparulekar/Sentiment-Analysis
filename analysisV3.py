import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import textstat
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Load your CSV file
df = pd.read_csv("combined_transcriptions.csv")

# Load the tokenizer and sequence classification model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Define the label mapping for depth of knowledge
label_mapping = {
    'LABEL_0': 'Beginner',
    'LABEL_1': 'Intermediate',
    'LABEL_2': 'Advanced'
}

# Function to analyze the depth of knowledge
def analyze_depth_of_knowledge(text):
    if isinstance(text, float) or pd.isna(text):
        text = ""
    if text == "":
        return 'Unknown'
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    
    label = f'LABEL_{predictions.item()}'
    return label_mapping.get(label, 'Unknown')

# Function to analyze fluency based on filler word count
def analyze_fluency(text):
    if isinstance(text, float) or pd.isna(text):
        text = ""
    words = text.split()
    filler_count = sum(1 for word in words if word.lower() in ['um', 'uh', 'ah', 'like', 'you know'])
    if len(words) == 0:
        return 'Low Fluency'
    elif filler_count / len(words) > 0.05:
        return 'Low Fluency'
    else:
        return 'High Fluency'

# Function to analyze readability
def analyze_readability(text):
    if isinstance(text, float) or pd.isna(text):
        text = ""
    return textstat.flesch_reading_ease(text)

# Function to analyze engagement based on sentiment
def analyze_engagement(text):
    if isinstance(text, float) or pd.isna(text):
        text = ""
    if text == "":
        return 'Low Engagement'

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)

    # Assuming positive sentiment is at index 1
    positive_score = probabilities[0][1].item()  # Get the positive sentiment score

    # Classify engagement based on the positive score
    if positive_score > 0.5:
        return 'High Engagement'
    elif positive_score >= 0.4:
        return 'Moderate Engagement'
    else:
        return 'Low Engagement'

# Function to analyze speech pace (words per minute)
def analyze_speech_pace(text, duration=None):
    if isinstance(text, float) or pd.isna(text) or (duration is not None and pd.isna(duration)):
        return "Unknown"
    
    words = text.split()
    words_count = len(words)

    # Set default duration to 5 minutes (300 seconds) if the duration is not provided
    if pd.isna(duration) or duration is None:
        duration = 60  # default duration in seconds (5 minutes)
    
    # Calculate Words Per Minute (WPM)
    wpm = (words_count / duration) * 60  # duration is in seconds, converting to minutes
    
    # Classify pace
    if wpm > 170:  # Fast speech
        return "Fast"
    elif 120 <= wpm <= 169:  # Moderate speech
        return "Moderate"
    else:  # Slow speech
        return "Slow"

# Apply readability analysis to each transcription
df['Readability'] = df['Transcription'].apply(analyze_readability)

# Classify clarity based on readability score
def classify_clarity(score):
    if score > 70:
        return 'Clear (Easy to understand)'
    elif score > 50:
        return 'Moderate (Understandable)'
    else:
        return 'Difficult (Complex language)'

def analyze_emotional_tone(text):
    scores = sia.polarity_scores(text)
    return 'Positive' if scores['compound'] >= 0.5 else 'Negative' if scores['compound'] <= -0.5 else 'Neutral'

def analyze_vocabulary_richness(text):
    words = text.split()
    unique_words = set(words)
    ttr = len(unique_words) / len(words) if words else 0
    
    # Classify vocabulary richness based on TTR thresholds
    if ttr > 0.7:
        return 'High'
    elif ttr > 0.4:
        return 'Medium'
    else:
        return 'Low'

# Apply fluency analysis to each transcription
df['Fluency'] = df['Transcription'].apply(analyze_fluency)

# Apply clarity classification
df['Clarity'] = df['Readability'].apply(classify_clarity)

# Apply depth of knowledge analysis to each transcription
df['Depth of Knowledge'] = df['Transcription'].apply(analyze_depth_of_knowledge)

# Apply engagement analysis to each transcription
df['Engagement'] = df['Transcription'].apply(analyze_engagement)

# Apply speech pace analysis to each transcription
# If the 'Duration' column is missing, a default duration of 300 seconds (5 minutes) will be used
df['Speech Pace'] = df.apply(lambda row: analyze_speech_pace(row['Transcription'], row.get('Duration')), axis=1)

df['Emotional Tone'] = df['Transcription'].apply(analyze_emotional_tone)

df['Vocabulary Richness'] = df['Transcription'].apply(analyze_vocabulary_richness)


# Export to CSV for Power BI visualization
df.to_csv("analysis_results_updated.csv", index=False)
print(df.head())
