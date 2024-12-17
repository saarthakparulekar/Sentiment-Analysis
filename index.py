import os
import csv
import speech_recognition as sr
from pydub import AudioSegment

def transcribe_audio(chunk_filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(chunk_filename) as source:
        audio = recognizer.record(source)
        return recognizer.recognize_google(audio)

def get_large_audio_transcription_fixed_interval(path, minutes=1, output_csv="transcriptions.csv"):
    """Splitting the large audio file into fixed interval chunks
    and applying speech recognition on each of these chunks."""
    sound = AudioSegment.from_file(path)
    chunk_length_ms = int(1000 * 60 * minutes)  # Convert minutes to milliseconds
    chunks = [sound[i:i + chunk_length_ms] for i in range(0, len(sound), chunk_length_ms)]

    folder_name = "audio-chunks"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)

    whole_text = ""

    # Open the CSV file to write the transcriptions
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Chunk Filename", "Transcription"])  # CSV headers

        for i, audio_chunk in enumerate(chunks, start=1):
            chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
            audio_chunk.export(chunk_filename, format="wav")  # Export chunk to .wav file
            
            try:
                text = transcribe_audio(chunk_filename)
            except sr.UnknownValueError:
                text = "[Unintelligible]"
            except sr.RequestError as e:
                text = f"[Service Error: {e}]"
            
            # Print to console
            print(chunk_filename, ":", text)

            # Write the chunk filename and transcription to the CSV
            writer.writerow([chunk_filename, text])

            # Append to full transcription
            whole_text += text + " "

    return whole_text

# Run the transcription process and save to CSV
audio_file = "audio.wav"  # Your audio file path
output_csv = "output_transcriptions.csv"  # Output CSV file path
transcription = get_large_audio_transcription_fixed_interval(audio_file, minutes=1, output_csv=output_csv)
print("Full Transcription:\n", transcription)
