# This file will scrape news data from Google News RSS Feeds since it is free and legally allowed - https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en
# Going to use that one instead since it isnt limited to just tech news. -> Broader

import pandas as pd
import feedparser
import schedule
import time
from datetime import datetime
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from scipy import sparse
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import ne_chunk, pos_tag, word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# TODO: I need to add another parameter in articles for category to help with ML
def scrape_news():
    print(f"Running scraper at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rss_url = "https://news.google.com/rss/search?q=technology&hl=en-US&gl=US&ceid=US:en"
    # rss_url = "https://news.google.com/rss/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx1YlY4U0FtVnVHZ0pWVXlnQVAB?hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)

    articles = []
    for entry in feed.entries:
        articles.append({
            'title': entry.title,
            'link': entry.link,
            'published': entry.published
        })

    df = pd.DataFrame(articles)

    # Save to timestamped CSV
    filename = f"news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} articles to {filename}\n")
    return filename

# TODO: Train the csv data using my model in improved_model.py to predict the categories
# of the articles and then use that to personalize the news digest - i will need to summarize
# the articles using a chatgpt api and then spit that out to the user.
# Function to preprocess the titles
def preprocess_data(df):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return ' '.join(words)
    
    df['cleaned_title'] = df['title'].apply(clean_text)
    return df

# Function to extract additional features
def extract_additional_features(df):
    # Text length features
    df['headline_length'] = df['title'].str.len()
    df['description_length'] = df['title'].str.len()  # No description, so use title
    df['headline_word_count'] = df['title'].str.split().str.len()
    df['description_word_count'] = df['title'].str.split().str.len()

    # Sentiment features
    sia = SentimentIntensityAnalyzer()
    sentiments = df['title'].apply(lambda x: sia.polarity_scores(x) if isinstance(x, str) else {'neg':0,'neu':0,'pos':0,'compound':0})
    df['headline_neg'] = sentiments.apply(lambda x: x['neg'])
    df['headline_pos'] = sentiments.apply(lambda x: x['pos'])
    df['desc_neg'] = sentiments.apply(lambda x: x['neg'])
    df['desc_pos'] = sentiments.apply(lambda x: x['pos'])

    # Named entity features
    def extract_entities(text):
        if not isinstance(text, str):
            return 0
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            named_entities = ne_chunk(pos_tags)
            return len([chunk for chunk in named_entities if hasattr(chunk, 'label')])
        except:
            return 0
    df['headline_entities'] = df['title'].apply(extract_entities)
    df['description_entities'] = df['title'].apply(extract_entities)

    # Return as numpy array
    return df[[
        'headline_length', 'description_length',
        'headline_word_count', 'description_word_count',
        'headline_neg', 'headline_pos',
        'desc_neg', 'desc_pos',
        'headline_entities', 'description_entities'
    ]].values

# Function to perform feature extraction
def extract_features(df):
    with open('tfidf_vectorizer.pickle', 'rb') as f:
        vectorizer = pickle.load(f)
    print(f"[DEBUG] Vectorizer expects {len(vectorizer.get_feature_names_out())} features.")
    tfidf_features = vectorizer.transform(df['cleaned_title'])
    print(f"[DEBUG] Transformed TF-IDF features shape: {tfidf_features.shape}")
    additional_features = extract_additional_features(df)
    additional_features_sparse = sparse.csr_matrix(additional_features)
    all_features = sparse.hstack([tfidf_features, additional_features_sparse])
    print(f"[DEBUG] All features shape (should match model): {all_features.shape}")
    return all_features

# Function to load the model and predict categories
def predict_categories(features):
    with open('random_forest_model.pickle', 'rb') as f:
        model = pickle.load(f)
    predictions = model.predict(features)
    return predictions

# TODO: Need to automate this now

# schedule.every().day.at("09:00").do(scrape_news)

    

# TODO: I am going to make a function to store the preferences of users 
# Alternate idea, make one that stores what type of news articles you click and personalize based on that
def userPreferences():
    database = []
    print("What types of tech news categories would you like to view summaries of?")
    print("Choices include 'AI', 'Quantum Computing', 'Chip Design' or Company names typed in individually please")
    print("Please type Done when you have selected all your preferences :)")
    
    while True:
        names = input()
        if names == "Done":
            break
        database.append(names)
    
    return database

# TODO: Need to preprocess the data before feeding it to training function i.e. make everything lowercase, remove punctuation, split data into 80/20 for train/test

# TODO: I need another function to train the machine learning model to fetch the data based on preferences - will make a supervised model
'''
def build_model(my_learning_rate, num_features):
  """Create and compile a simple linear regression model."""
  # Describe the topography of the model.
  # The topography of a simple linear regression model
  # is a single node in a single layer.
  inputs = keras.Input(shape=(num_features,))
  outputs = keras.layers.Dense(units=1)(inputs)
  model = keras.Model(inputs=inputs, outputs=outputs)

  # Compile the model topography into code that Keras can efficiently
  # execute. Configure training to minimize the model's mean squared error.
  model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=my_learning_rate),
                loss="mean_squared_error",
                metrics=[keras.metrics.RootMeanSquaredError()])

  return model
'''

'''
def train_model(model, features, label, epochs, batch_size):
  """Train the model by feeding it data."""

  # Feed the model the feature and the label.
  # The model will train for the specified number of epochs.
  history = model.fit(x=features,
                      y=label,
                      batch_size=batch_size,
                      epochs=epochs)

  # Gather the trained model's weight and bias.
  trained_weight = model.get_weights()[0]
  trained_bias = model.get_weights()[1]

  # The list of epochs is stored separately from the rest of history.
  epochs = history.epoch

  # Isolate the error for each epoch.
  hist = pd.DataFrame(history.history)

  # To track the progression of training, we're going to take a snapshot
  # of the model's root mean squared error at each epoch.
  rmse = hist["root_mean_squared_error"]

  return trained_weight, trained_bias, epochs, rmse
'''

# TODO: A function to test the model and see the initial and after training accuracy - for resume arc purposes
'''
def run_experiment(df, feature_names, label_name, learning_rate, epochs, batch_size):

  print('INFO: starting training experiment with features={} and label={}\n'.format(feature_names, label_name))

  num_features = len(feature_names)

  features = df.loc[:, feature_names].values
  label = df[label_name].values

  model = build_model(learning_rate, num_features)
  model_output = train_model(model, features, label, epochs, batch_size)

  print('\nSUCCESS: training experiment complete\n')
  print('{}'.format(model_info(feature_names, label_name, model_output)))
  make_plots(df, feature_names, label_name, model_output)

  return model

print("SUCCESS: defining linear regression functions complete.")
'''
# TODO: I need a function to summarize the fetched data for the digest - can use a chatgpt api for this

# TODO: I need a function to ouput the summarized articles - simple print statement or could even be done inside of summarize function itself

# TODO: Integrate a voice assistance - use google assistant or siri - whichever one is free and easiest to integrate

# TODO: Utilize cloud storage - aws lambda or s3


# Define keyword lists for each category
CATEGORY_KEYWORDS = {
    'TECH': [
        'tech', 'technology', 'quantum', 'ai', 'artificial intelligence', 'machine learning', 'computer',
        'software', 'hardware', 'chip', 'semiconductor', 'gadget', 'device', 'internet', 'cybersecurity',
        'cloud', 'data', 'startup', 'app', 'application', 'robotics', 'automation', 'blockchain', 'crypto',
        'it', 'vr', 'ar', 'virtual reality', 'augmented reality', 'mobile', 'smartphone', 'electronics',
        'innovation', 'science'
    ],
    'SPORTS': [
        'sports', 'football', 'soccer', 'cricket', 'basketball', 'tennis', 'olympics', 'athlete', 'match',
        'game', 'tournament', 'league', 'score', 'goal', 'win', 'loss', 'championship'
    ],
    'POLITICS': [
        'politics', 'election', 'government', 'senate', 'congress', 'parliament', 'president', 'prime minister',
        'policy', 'law', 'vote', 'campaign', 'political', 'democrat', 'republican', 'conservative', 'liberal'
    ],
    'BUSINESS': [
        'business', 'finance', 'stock', 'market', 'economy', 'company', 'corporate', 'earnings', 'profit',
        'loss', 'investment', 'investor', 'share', 'merger', 'acquisition', 'ipo', 'ceo', 'startup', 'entrepreneur'
    ],
    'ENTERTAINMENT': [
        'entertainment', 'movie', 'film', 'music', 'concert', 'celebrity', 'actor', 'actress', 'singer', 'band',
        'album', 'show', 'tv', 'series', 'award', 'festival', 'hollywood', 'bollywood'
    ],
    'SCIENCE': [
        'science', 'research', 'study', 'scientist', 'experiment', 'discovery', 'laboratory', 'physics',
        'chemistry', 'biology', 'astronomy', 'space', 'nasa', 'scientific', 'academic', 'journal'
    ]
}

CATEGORY_PRIORITY = ['TECH', 'SPORTS', 'POLITICS', 'BUSINESS', 'ENTERTAINMENT', 'SCIENCE']

def rule_based_categories(text):
    text = text.lower()
    matched = []
    for category in CATEGORY_PRIORITY:
        for keyword in CATEGORY_KEYWORDS[category]:
            if keyword in text:
                matched.append(category)
                break  # Only need one keyword per category
    return matched


# Steps after this - I can now ask users on what they like to give them personalised news OR I can track the articles they click
# The second one may have issues with cookies I think but would be cooler, the first one could be a quicker one for users to see results

if __name__ == "__main__":
    filename = scrape_news()
    df = pd.read_csv(filename)
    df = preprocess_data(df)
    features = extract_features(df)
    # Rule-based pre-filter for all categories (multi-label)
    predicted_categories = []
    for idx, row in df.iterrows():
        cats = rule_based_categories(row['cleaned_title'])
        if cats:
            predicted_categories.append(','.join(cats))
        else:
            predicted_categories.append(None)  # Placeholder for classifier
    # Use classifier only for articles not matched by rules
    to_classify_idx = [i for i, cat in enumerate(predicted_categories) if cat is None]
    if to_classify_idx:
        sub_features = features[to_classify_idx]
        sub_predictions = predict_categories(sub_features)
        for i, pred in zip(to_classify_idx, sub_predictions):
            predicted_categories[i] = pred
    df['predicted_category'] = predicted_categories
    df.to_csv(f"news_with_categories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
    print("Categories assigned and saved to CSV.")