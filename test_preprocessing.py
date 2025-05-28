from text_preprocessing import clean_text, remove_stopwords, lemmatize_text, preprocess_text

# Test text
test_text = """
The quick brown foxes are running over the hills! 
Check out https://example.com for more info.
The weather is better today than yesterday.
"""

print("Original text:")
print(test_text)
print("\n" + "="*50 + "\n")

# Test clean_text
print("After cleaning:")
cleaned_text = clean_text(test_text)
print(cleaned_text)
print("\n" + "="*50 + "\n")

# Test remove_stopwords
print("After removing stopwords:")
text_without_stopwords = remove_stopwords(cleaned_text)
print(text_without_stopwords)
print("\n" + "="*50 + "\n")

# Test lemmatize_text
print("After lemmatization:")
lemmatized_text = lemmatize_text(text_without_stopwords)
print(lemmatized_text)
print("\n" + "="*50 + "\n")

# Test complete preprocessing
print("After complete preprocessing:")
final_text = preprocess_text(test_text)
print(final_text) 