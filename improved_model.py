import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from text_preprocessing import preprocess_text

# Load and preprocess data
def load_and_preprocess_data(file_path):
    print("Loading data...")
    df = pd.read_json(file_path, lines=True)
    
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
    
    # Tokenize and pad sequences
    tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['processed_text'])
    
    sequences = tokenizer.texts_to_sequences(df['processed_text'])
    padded_sequences = pad_sequences(sequences, maxlen=150, padding='post', truncating='post')
    
    return padded_sequences, df['label'].values, tokenizer, label_map

def create_model(vocab_size, num_classes):
    model = Sequential([
        # Embedding layer
        Embedding(input_dim=vocab_size, output_dim=100, input_length=150),
        
        # Bidirectional LSTM layers
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        
        # Dense layers
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model():
    # Load and preprocess data
    df = load_and_preprocess_data('News_Category_Dataset_v3.json')
    
    # Prepare data for training
    X, y, tokenizer, label_map = prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and compile model
    model = create_model(vocab_size=20000, num_classes=len(label_map))
    
    # Use a lower learning rate
    optimizer = Adam(learning_rate=0.001)
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=0.0001
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr]
    )
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {accuracy:.2f}")
    
    # Save model and tokenizer
    model.save('news_category_model.h5')
    
    # Save tokenizer and label map for later use
    import pickle
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('label_map.pickle', 'wb') as handle:
        pickle.dump(label_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    train_model() 