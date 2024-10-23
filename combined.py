import pandas as pd

# File path to your CSV
csv_file = 'output_transcriptions.csv'

# Create an empty list to store the transcriptions from each chunk
transcriptions_list = []

# Define the chunk size
chunk_size = 1000  # Adjust as needed

# Read the CSV file in chunks, focusing only on the 'transcriptions' column
for i, chunk in enumerate(pd.read_csv(csv_file, chunksize=chunk_size, usecols=['Transcription'])):
    print(f"Chunk {i + 1} - Rows: {len(chunk)}")  # Optional: Check number of rows in each chunk
    transcriptions_list.append(chunk['Transcription'])

# Concatenate all transcriptions into a single Series
combined_transcriptions = pd.concat(transcriptions_list, axis=0, ignore_index=True)

# Optionally, save the combined transcriptions to a new CSV file
combined_transcriptions.to_csv('combined_transcriptions.csv', index=False)

# Display a preview of the combined transcriptions
print("Combined Transcriptions (first 5 rows):")
print(combined_transcriptions.head())


