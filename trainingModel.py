#I am going to be training a machine learning model here based on kaggle data that contains 210,294 records between 2012 and 2022.
'''Each json record contains the following attributes:
category: Category article belongs to
headline: Headline of the article
authors: Person authored the article
link: Link to the post
short_description: Short description of the article
date: Date the article was published'''
# I want to then use this model to predict the category an article belongs to in a different dataset of mine. Then users will give me their
# preferred categories and I will use the model to suggest about 5 articles to them with links, and summarize the articles for the user.
# Good plan so far!

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense

#Parsing the organised Kaggle dataset using pandas
df = pd.read_json("News_Category_Dataset_v3.json", lines = True)

df[['category', 'headline', 'short_description']].head()

# Clean text data
df['text'] = df['headline'] + " " + df['short_description']
df.dropna(subset=['text', 'category'], inplace=True)
df['text'] = df['text'].astype(str)
df = df[df['text'].str.strip().astype(bool)]  # Remove empty/whitespace-only strings

#Encoding all the labels as numbers similar to enums in c++
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['category'])

#Making a hashmap for fast lookup
label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# Turning the words into numbers for my ML model and padding is just for formatting purposes
tokenizer = Tokenizer(num_words=10000, oov_token = "<OOV>")
tokenizer.fit_on_texts(df['text'])

sequences = tokenizer.texts_to_sequences(df['text'])

padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

X = padded_sequences
y = df['label'].values

# Split data into 80% training and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Embedding(input_dim=10000, output_dim=16, input_length=100),
    GlobalAveragePooling1D(),
    Dense(32, activation='relu'),
    Dense(len(label_map), activation='softmax')  # multi-class classification
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=5,
    validation_data=(X_test, y_test),
    batch_size=128
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}") #First test accuracy is 0.49 - alot left to improve on


