import pandas as pd
import re
import language_tool_python
from spellchecker import SpellChecker
import nltk
from nltk.corpus import words
from nltk.metrics import edit_distance

# Download required NLTK data
nltk.download('words')

# Initialize language tool and spell checker
tool = language_tool_python.LanguageTool('en-US')
spell = SpellChecker()
word_list = set(words.words())

# Step 1: Load CSV data
def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df

# Step 2: Clean text (removing unwanted characters)
def clean_text(text):
    # Remove unwanted characters, punctuation, and multiple spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()

# Step 3: Grammar correction using LanguageTool
def correct_grammar(text):
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text

# Step 4: Spelling correction using SpellChecker
def correct_spelling(text):
    words = text.split()
    corrected_text = [spell.correction(word) if word in spell.unknown(words) else word for word in words]
    return ' '.join(corrected_text)

# Step 5: Phonetic check using NLTK (basic enunciation check)
def check_enunciation(text):
    tokens = text.split()
    corrected_tokens = []
    for token in tokens:
        if token not in word_list:
            # Find closest match using edit distance
            closest_word = min(word_list, key=lambda w: edit_distance(token, w))
            corrected_tokens.append(closest_word)
        else:
            corrected_tokens.append(token)
    return ' '.join(corrected_tokens)

# Step 6: Preprocess the data
def preprocess_lecture_transcripts(file_path, output_csv="processed_transcripts.csv"):
    # Load the CSV file
    df = load_csv(file_path)
    
    # Replace 'transcript' with 'transcription'
    df['cleaned_transcript'] = df['Transcription'].apply(clean_text)
    print("s1")
    # Apply grammar correction
    df['corrected_transcript'] = df['cleaned_transcript'].apply(correct_grammar)
    print("s2")

    # Save the preprocessed data into a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Preprocessed data saved to {output_csv}")
    # Apply spelling correction
    # df['spelled_transcript'] = df['corrected_transcript'].apply(correct_spelling)
    # print("s3")
    # Apply enunciation check (phonetic similarity)
    # df['final_transcript'] = df['spelled_transcript'].apply(check_enunciation)
    
    

# Example usage
if __name__ == "__main__":
    # Path to the CSV file with lecture transcripts
    input_csv = 'output_transcriptions.csv'
    preprocess_lecture_transcripts(input_csv, output_csv="processed_output_transcriptions.csv")

# import pandas as pd

# df = pd.read_csv('output_transcriptions.csv')  # Replace with your actual file path
# print(df)
# # Display the first 5 rows of the CSV
# # print(df.head())

# # If you want to see all columns or a large portion of the file:

