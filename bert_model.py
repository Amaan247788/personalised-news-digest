import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from scipy import sparse
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import ne_chunk, pos_tag
import re
import json
import pickle

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Define category groups
CATEGORY_GROUPS = {
    'NEWS': ['U.S. NEWS', 'WORLD NEWS', 'THE WORLDPOST', 'WORLDPOST', 'IMPACT'],
    'LIFESTYLE': ['HEALTHY LIVING', 'GREEN', 'WELLNESS', 'STYLE & BEAUTY', 'STYLE'],
    'ENTERTAINMENT': ['ENTERTAINMENT', 'COMEDY', 'ARTS', 'ARTS & CULTURE', 'CULTURE & ARTS'],
    'TECH': ['TECH', 'SCIENCE'],
    'BUSINESS': ['BUSINESS', 'MONEY'],
    'SPORTS': ['SPORTS'],
    'POLITICS': ['POLITICS'],
    'OTHER': ['WEIRD NEWS', 'GOOD NEWS', 'FIFTY']
}

def get_category_group(category):
    for group, categories in CATEGORY_GROUPS.items():
        if category in categories:
            return group
    return 'OTHER'

def load_data(file_path, sample_size=0.12):
    print("Loading data...")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    # Sample the data
    df = df.sample(frac=sample_size, random_state=42)
    print(f"Loaded {len(df)} samples (using {sample_size*100}% of data)")
    return df

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def preprocess_text(text):
    try:
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stop_words]
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error processing text: {text[:50]}... Error: {str(e)}")
        return text

def extract_text_length_features(df):
    """Extract text length related features"""
    df['headline_length'] = df['headline'].str.len()
    df['description_length'] = df['short_description'].str.len()
    df['headline_word_count'] = df['headline'].str.split().str.len()
    df['description_word_count'] = df['short_description'].str.split().str.len()
    return df

def extract_sentiment_features(df):
    """Extract sentiment analysis features"""
    sia = SentimentIntensityAnalyzer()
    
    def get_sentiment_scores(text):
        if not isinstance(text, str):
            return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
        return sia.polarity_scores(text)
    
    # Get sentiment scores for headlines
    headline_sentiments = df['headline'].apply(get_sentiment_scores)
    df['headline_neg'] = headline_sentiments.apply(lambda x: x['neg'])
    df['headline_neu'] = headline_sentiments.apply(lambda x: x['neu'])
    df['headline_pos'] = headline_sentiments.apply(lambda x: x['pos'])
    df['headline_compound'] = headline_sentiments.apply(lambda x: x['compound'])
    
    # Get sentiment scores for descriptions
    desc_sentiments = df['short_description'].apply(get_sentiment_scores)
    df['desc_neg'] = desc_sentiments.apply(lambda x: x['neg'])
    df['desc_neu'] = desc_sentiments.apply(lambda x: x['neu'])
    df['desc_pos'] = desc_sentiments.apply(lambda x: x['pos'])
    df['desc_compound'] = desc_sentiments.apply(lambda x: x['compound'])
    
    return df

def extract_named_entities(text):
    """Extract named entities from text"""
    if not isinstance(text, str):
        return 0
    try:
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        named_entities = ne_chunk(pos_tags)
        return len([chunk for chunk in named_entities if hasattr(chunk, 'label')])
    except:
        return 0

def prepare_data(df):
    print("Preprocessing text...")
    # Clean and preprocess headlines
    print("Cleaning headlines...")
    df['cleaned_headline'] = df['headline'].apply(clean_text)
    print("Preprocessing headlines...")
    df['processed_headline'] = df['cleaned_headline'].apply(preprocess_text)
    
    # Clean and preprocess short descriptions
    print("Cleaning short descriptions...")
    df['cleaned_description'] = df['short_description'].apply(clean_text)
    print("Preprocessing short descriptions...")
    df['processed_description'] = df['cleaned_description'].apply(preprocess_text)
    
    # Extract additional features
    print("Extracting text length features...")
    df = extract_text_length_features(df)
    
    print("Extracting sentiment features...")
    df = extract_sentiment_features(df)
    
    print("Extracting named entity features...")
    df['headline_entities'] = df['headline'].apply(extract_named_entities)
    df['description_entities'] = df['short_description'].apply(extract_named_entities)
    
    # Combine headline and description
    df['combined_text'] = df['processed_headline'] + ' ' + df['processed_description']
    
    # Add category group
    df['category_group'] = df['category'].apply(get_category_group)
    
    return df

def balance_dataset(X, y):
    print("\nBalancing dataset...")
    # Get class distribution before balancing
    print("Class distribution before balancing:")
    print(pd.Series(y).value_counts())
    
    # Apply SMOTE to balance the dataset
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    # Get class distribution after balancing
    print("\nClass distribution after balancing:")
    print(pd.Series(y_balanced).value_counts())
    
    return X_balanced, y_balanced

def train_specialized_classifier(X, y, category):
    """Train a specialized classifier for a specific category"""
    # Create a binary classification problem
    y_binary = (y == category).astype(int)
    
    # Balance the dataset
    X_balanced, y_balanced = balance_dataset(X, y_binary)
    
    # Train classifier
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    classifier.fit(X_balanced, y_balanced)
    
    return classifier

def train_model():
    # Load and preprocess data
    df = load_data('News_Category_Dataset_v3.json', sample_size=0.12)
    df = prepare_data(df)
    
    # Create TF-IDF vectorizer with optimized parameters
    print("Creating TF-IDF vectors...")
    vectorizer = TfidfVectorizer(
        max_features=8010,
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.90,
        sublinear_tf=True,
        use_idf=True,
        smooth_idf=True
    )
    X_text = vectorizer.fit_transform(df['combined_text'])
    
    # Combine text features with additional features
    additional_features = df[[
        'headline_length', 'description_length',
        'headline_word_count', 'description_word_count',
        'headline_neg', 'headline_pos',
        'desc_neg', 'desc_pos',
        'headline_entities', 'description_entities'
    ]].values
    
    # Convert additional features to sparse matrix
    additional_features_sparse = sparse.csr_matrix(additional_features)
    
    # Combine sparse matrices
    X = sparse.hstack([X_text, additional_features_sparse])
    y = df['category']
    y_group = df['category_group']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    _, _, y_train_group, y_test_group = train_test_split(X, y_group, test_size=0.2, random_state=42, stratify=y_group)
    
    # Train main classifier for category groups
    print("\nTraining main classifier for category groups...")
    main_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=42
    )
    main_classifier.fit(X_train, y_train_group)
    
    # Train specialized classifiers for challenging categories
    print("\nTraining specialized classifiers for challenging categories...")
    specialized_classifiers = {}
    for category in ['U.S. NEWS', 'IMPACT', 'HEALTHY LIVING', 'GREEN', 'WEIRD NEWS', 'GOOD NEWS']:
        print(f"Training specialized classifier for {category}...")
        specialized_classifiers[category] = train_specialized_classifier(X_train, y_train, category)
    
    # Evaluate the model
    print("\nEvaluating model...")
    y_pred_group = main_classifier.predict(X_test)
    y_pred = y_test.copy()
    
    # Use specialized classifiers for challenging categories
    for category, classifier in specialized_classifiers.items():
        # Get predictions for the category
        category_mask = (y_pred_group == get_category_group(category))
        if category_mask.any():
            category_probs = classifier.predict_proba(X_test[category_mask])[:, 1]
            # Update predictions where specialized classifier is confident
            confident_mask = category_probs > 0.6
            y_pred[category_mask] = np.where(confident_mask, category, y_pred[category_mask])
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Print per-class accuracy
    print("\nPer-class Accuracy:")
    for i, class_name in enumerate(np.unique(y_test)):
        class_correct = np.sum((y_test == class_name) & (y_pred == class_name))
        class_total = np.sum(y_test == class_name)
        accuracy = class_correct / class_total if class_total > 0 else 0
        print(f"{class_name}: {accuracy:.2%} accuracy ({class_correct}/{class_total})")
    
    # Save the models and vectorizer
    print("\nSaving models and vectorizer...")
    with open('main_classifier.pickle', 'wb') as f:
        pickle.dump(main_classifier, f)
    with open('specialized_classifiers.pickle', 'wb') as f:
        pickle.dump(specialized_classifiers, f)
    with open('tfidf_vectorizer.pickle', 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"[DEBUG] Saved vectorizer with {len(vectorizer.get_feature_names_out())} features.")

    with open('random_forest_model.pickle', 'wb') as f:
        pickle.dump(main_classifier, f)
    print("Retraining complete. Old files replaced with updated ones.")

if __name__ == "__main__":
    train_model() 