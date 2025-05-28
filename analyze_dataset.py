import json
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

# Read the JSON file
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Analyze the data
def analyze_dataset(data):
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)
    
    # Get category distribution
    category_counts = df['category'].value_counts()
    
    print("\nCategory Distribution:")
    print(category_counts)
    print(f"\nTotal number of categories: {len(category_counts)}")
    print(f"Total number of articles: {len(df)}")
    
    # Basic statistics about headlines
    df['headline_length'] = df['headline'].str.len()
    print("\nHeadline Length Statistics:")
    print(df['headline_length'].describe())
    
    # Plot category distribution
    plt.figure(figsize=(12, 6))
    category_counts.plot(kind='bar')
    plt.title('Distribution of News Categories')
    plt.xlabel('Category')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('category_distribution.png')
    plt.close()

if __name__ == "__main__":
    data = load_data('News_Category_Dataset_v3.json')
    analyze_dataset(data) 