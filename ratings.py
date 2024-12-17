import pandas as pd

# Load your dataset
df = pd.read_csv("analysis_results_pck1.csv")

# Define score mappings for each metric
score_mapping = {
    "Clarity": {"Clear (Easy to understand)": 3, "Moderate (Understandable)": 2, "Difficult (Complex language)": 1},
    "Fluency": {"High Fluency": 3, "Low Fluency": 1},
    "Engagement": {"High Engagement": 3, "Moderate Engagement": 2, "Low Engagement": 1},
    "Speech Pace": {"Moderate": 3, "Fast": 2, "Slow": 1},
    "Depth of Knowledge": {"Advanced": 3, "Intermediate": 2, "Beginner": 1},
    "Vocabulary Richness": {"High": 3, "Medium": 2, "Low": 1}
}

# Initialize a dictionary to store total scores and counts for each metric
total_scores = {metric: 0 for metric in score_mapping}
counts = {metric: 0 for metric in score_mapping}

# Calculate total score and count for each metric across all chunks
for metric, mapping in score_mapping.items():
    for value, score in mapping.items():
        metric_counts = df[metric].value_counts()
        total_scores[metric] += metric_counts.get(value, 0) * score
        counts[metric] += metric_counts.get(value, 0)

# Calculate average score (as a percentage) for each metric
final_ratings = {}
for metric in total_scores:
    if counts[metric] > 0:
        final_ratings[metric] = (total_scores[metric] / (counts[metric] * 3)) * 100
    else:
        final_ratings[metric] = "No data"

# Convert final_ratings dictionary to a DataFrame
final_ratings_df = pd.DataFrame(list(final_ratings.items()), columns=["Metric", "Rating (Out of 100)"])

# Export to CSV
final_ratings_df.to_csv("overall_ratings_pck1.csv", index=False)

print("Exported overall ratings to overall_ratings.csv")
print(final_ratings_df)
