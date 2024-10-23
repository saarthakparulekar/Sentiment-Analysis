import pandas as pd
from transformers import pipeline
import textstat

# Load the CSV file
df = pd.read_csv("combined_transcriptions.csv")

# Initialize the text classifier (NLP model) and sentiment analyzer
classifier = pipeline("text-classification", model="distilbert-base-uncased")
sentiment_analyzer = pipeline("sentiment-analysis")

# Define a mapping for the integer labels returned by the model
label_mapping = {
    0: 'Beginner',
    1: 'Intermediate',
    2: 'Advanced'
}

# Function to analyze depth of knowledge
def analyze_depth_of_knowledge(text):
    results = classifier(text)
    label = int(results[0]['label'])  # Convert label to integer
    return label_mapping.get(label, 'Unknown')

# Function to measure fluency
filler_words = ['um', 'uh', 'ah', 'like', 'you know']
def analyze_fluency(text):
    words = text.split()
    filler_count = sum(1 for word in words if word.lower() in filler_words)
    if filler_count / len(words) > 0.05:
        return 'Low Fluency'
    else:
        return 'High Fluency'

# Function to calculate words per minute
# def calculate_wpm(transcription, duration_in_seconds):
#     word_count = len(transcription.split())
#     duration_in_minutes = duration_in_seconds / 60
#     return word_count / duration_in_minutes

# def classify_rate_of_speech(wpm):
#     if wpm < 120:
#         return 'Slow'
#     elif wpm < 180:
#         return 'Moderate'
#     else:
#         return 'Fast'

# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment = sentiment_analyzer(text)[0]
    return sentiment['label']

# Function to calculate readability
def analyze_readability(text):
    return textstat.flesch_reading_ease(text)

def classify_clarity(score):
    if score > 70:
        return 'Clear (Easy to understand)'
    elif score > 50:
        return 'Moderate (Understandable)'
    else:
        return 'Difficult (Complex language)'

# Apply all analysis functions to the dataframe
df['Depth of Knowledge'] = df['Transcription'].apply(analyze_depth_of_knowledge)
df['Fluency'] = df['Transcription'].apply(analyze_fluency)
# df['WPM'] = df.apply(lambda row: calculate_wpm(row['transcription'], row['duration']), axis=1)
# df['Rate of Speech'] = df['WPM'].apply(classify_rate_of_speech)
df['Sentiment'] = df['Transcription'].apply(analyze_sentiment)
df['Readability'] = df['Transcription'].apply(analyze_readability)
df['Clarity'] = df['Readability'].apply(classify_clarity)

# Display the dataframe with all new columns
print(df.head())

# Save the results for Power BI visualization
df.to_csv("analysis_results_extended.csv", index=False)
