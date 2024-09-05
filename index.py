import os
import csv
import speech_recognition as sr
from pydub import AudioSegment

def transcribe_audio(chunk_filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(chunk_filename) as source:
        audio = recognizer.record(source)
        return recognizer.recognize_google(audio)

def get_large_audio_transcription_fixed_interval(path, minutes=0.5, output_csv="transcriptions.csv"):
    """Splitting the large audio file into fixed interval chunks
    and apply speech recognition on each of these chunks"""
    sound = AudioSegment.from_file(path)  
    chunk_length_ms = int(1000 * 60 * minutes)  # 30 seconds
    chunks = [sound[i:i + chunk_length_ms] for i in range(0, len(sound), chunk_length_ms)]
    
    folder_name = "audio-fixed-chunks"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    
    whole_text = ""
    
    # Open the CSV file to write the results
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Chunk Filename", "Transcription"])
        
        for i, audio_chunk in enumerate(chunks, start=1):
            chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
            audio_chunk.export(chunk_filename, format="wav")
            try:
                text = transcribe_audio(chunk_filename)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
                text = ""  # Handle errors by storing an empty string
            else:
                text = f"{text.capitalize()}."
                print(chunk_filename, ":", text)
                whole_text += text
            
            # Write to CSV
            writer.writerow([chunk_filename, text])
    
    return whole_text

transcription = get_large_audio_transcription_fixed_interval("audio.wav", minutes=0.5, output_csv="output_transcriptions.csv")
