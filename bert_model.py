import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from text_preprocessing import preprocess_text
import nltk

def load_and_preprocess_data(file_path):
    print("Loading data...")
    df = pd.read_json(file_path, lines=True)
    
    # Sample only 10% of the data for faster training
    df = df.sample(frac=0.1, random_state=42)
    print(f"Using {len(df)} samples for training")
    
    # Combine headline and description
    df['text'] = df['headline'] + " " + df['short_description']
    
    # Clean and preprocess text
    print("Preprocessing text...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Remove any empty texts after preprocessing
    df = df[df['processed_text'].str.strip().astype(bool)]
    
    return df

def prepare_data(df):
    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['category'])
    label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    
    return df['processed_text'].values, df['label'].values, label_map

def train_baseline_models(X_train, X_test, y_train, y_test):
    print("\nTraining baseline models...")
    
    # TF-IDF Vectorization
    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train Multinomial Naive Bayes
    print("\nTraining Multinomial Naive Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    nb_pred = nb_model.predict(X_test_tfidf)
    nb_accuracy = accuracy_score(y_test, nb_pred)
    print(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")
    
    # Train Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)
    lr_pred = lr_model.predict(X_test_tfidf)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_tfidf, y_train)
    rf_pred = rf_model.predict(X_test_tfidf)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    
    return nb_accuracy, lr_accuracy, rf_accuracy

class HubLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_url, **kwargs):
        super(HubLayer, self).__init__(**kwargs)
        self.embedding_url = embedding_url
        self.hub_layer = hub.KerasLayer(embedding_url, trainable=False)

    def call(self, inputs):
        return self.hub_layer(inputs)

def create_model(num_classes):
    embedding = "https://tfhub.dev/google/nnlm-en-dim128/2"
    hub_layer = HubLayer(embedding, name="embedding")
    
    inputs = tf.keras.Input(shape=(), dtype=tf.string, name="text")
    x = hub_layer(inputs)
    x = Dense(64)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    x = Dense(32)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_dataset(texts, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((texts, labels))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def train_model():
    # Load and preprocess data
    df = load_and_preprocess_data('News_Category_Dataset_v3.json')
    
    # Prepare data
    X, y, label_map = prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train baseline models
    nb_accuracy, lr_accuracy, rf_accuracy = train_baseline_models(X_train, X_test, y_train, y_test)
    
    # Create model
    model = create_model(num_classes=len(label_map))
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=1,
        min_lr=0.0001
    )
    
    # Create TensorFlow datasets
    train_dataset = create_dataset(X_train, y_train, batch_size=64)
    test_dataset = create_dataset(X_test, y_test, batch_size=64)
    
    # Train model
    print("\nTraining neural network model...")
    history = model.fit(
        train_dataset,
        epochs=3,
        validation_data=test_dataset,
        callbacks=[early_stopping, reduce_lr]
    )
    
    # Evaluate model
    loss, accuracy = model.evaluate(test_dataset)
    print(f"\nNeural Network Test Accuracy: {accuracy:.4f}")
    
    # Print comparison
    print("\nModel Comparison:")
    print(f"Multinomial Naive Bayes: {nb_accuracy:.4f}")
    print(f"Logistic Regression: {lr_accuracy:.4f}")
    print(f"Random Forest: {rf_accuracy:.4f}")
    print(f"Neural Network: {accuracy:.4f}")
    
    # Save model
    model.save('news_category_model.keras')
    
    # Save label map
    import pickle
    with open('label_map.pickle', 'wb') as handle:
        pickle.dump(label_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    train_model() 