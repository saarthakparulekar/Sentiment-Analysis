# Qualitative Analysis of Educational Video Lectures

This repository contains Python scripts designed to analyze the qualitative aspects of educational video lectures. The workflow includes audio-to-text conversion, transcription management, analysis, and generating final ratings. 

## Features
The project assesses video lectures based on the following qualitative aspects:
- **Depth of Knowledge**
- **Readability**
- **Fluency**
- **Speech Pace**
- **Engagement**
- **Emotional Tone**
- **Vocabulary Richness**

## File Structure

1. **`index.py`**  
   Converts audio files (in `.wav` format) into text transcriptions and saves the output in a `.csv` format.

2. **`combine.py`**  
   Combines the individual transcriptions into a single file for further processing.

3. **`analysis.py`**  
   Analyzes the combined transcription file to evaluate the qualitative aspects of the lectures.

4. **`ratings.py`**  
   Aggregates the analysis results and generates final qualitative ratings for the lecture.

## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- Required Python libraries 

### Installation
1. Clone the repository

2. Navigate to the project directory

3. Install the dependencies

### Usage

1. Place your `.wav` audio files in the `audio/` directory.

2. Run `index.py` to convert audio files to text:
   This will generate a CSV file containing the transcriptions.

3. Combine the transcriptions using `combine.py`

4. Analyze the combined transcription using `analysis.py`:
   This script evaluates the qualitative aspects listed above.

5. Generate the final ratings using `ratings.py`

6. Visualize the ratings and other aspects in PowerBI
   
  
### Output
- Transcription CSV file (from `index.py`)
- Combined transcription CSV file (from `combine.py`)
- Analysis results (from `analysis.py`)
- Final ratings (from `ratings.py`)

## Contributing
Feel free to open issues or submit pull requests for any improvements or additional features.

## License
This project is licensed under the [MIT License](LICENSE).

##Sample PowerBI Visualization.

![PowerBI Screenshot](https://github.com/user-attachments/assets/4ade8b43-097b-4043-9223-ba6a85cc973b)
![PowerBI Screenshot](https://github.com/user-attachments/assets/45a102ea-4183-4392-a0f3-5f1da742bbe8)
![PowerBI Screenshot](https://github.com/user-attachments/assets/7e887970-1588-4054-8b03-7666096084a2)

