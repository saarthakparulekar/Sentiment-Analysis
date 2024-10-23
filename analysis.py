import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import textstat

# Load your CSV file
df = pd.read_csv("combined_transcriptions.csv")

# Load the CodeBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base")

# Define the label mapping (assuming LABEL_1, LABEL_2, etc. represent different depths)
label_mapping = {
    'LABEL_0': 'Beginner',
    'LABEL_1': 'Intermediate',
    'LABEL_2': 'Advanced'
}

# Function to analyze the depth of knowledge
def analyze_python_depth(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract label from the logits (e.g., the highest scoring label)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    
    # Convert the label to human-readable depth level using the mapping
    label = f'LABEL_{predictions.item()}'
    return label_mapping.get(label, 'Unknown')

# Define a list of common filler words
filler_words = ['um', 'uh', 'ah', 'like', 'you know']

# Function to measure fluency based on filler word count
def analyze_fluency(text):
    words = text.split()
    filler_count = sum(1 for word in words if word.lower() in filler_words)
    # Return fluency rating based on filler word percentage
    if filler_count / len(words) > 0.05:
        return 'Low Fluency'
    else:
        return 'High Fluency'

def analyze_readability(text):
    return textstat.flesch_reading_ease(text)

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


# Apply fluency analysis to each transcription
df['Fluency'] = df['Transcription'].apply(analyze_fluency)

# Apply clarity classification
df['Clarity'] = df['Readability'].apply(classify_clarity)

# Apply the analysis to each transcription and map to meaningful depth levels
df['Depth of Knowledge'] = df['Transcription'].apply(analyze_python_depth)

# Export to CSV for Power BI visualization
df.to_csv("analysis_results4.csv", index=False)
print(df.head())

