import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load your CSV file
df = pd.read_csv("output_transcriptions.csv")

# Load the CodeBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base")

# Define the label mapping (assuming LABEL_1, LABEL_2, etc. represent different depths)
label_mapping = {
    'LABEL_1': 'Beginner',
    'LABEL_2': 'Intermediate',
    'LABEL_3': 'Advanced'
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

# Apply the analysis to each transcription and map to meaningful depth levels
df['Depth of Knowledge'] = df['Transcription'].apply(analyze_python_depth)

# Export to CSV for Power BI visualization
df.to_csv("analysis_results.csv", index=False)
print(df.head())

