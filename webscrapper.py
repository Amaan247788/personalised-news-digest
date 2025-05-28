# This file will scrape news data from Google News RSS Feeds since it is free and legally allowed - https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en
# Going to use that one instead since it isnt limited to just tech news. -> Broader

import pandas as pd
import feedparser
import schedule
import time
from datetime import datetime

# TODO: I need to add another parameter in articles for category to help with ML
def scrape_news():
    print(f"Running scraper at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rss_url = "https://news.google.com/rss/search?q=technology&hl=en-US&gl=US&ceid=US:en"
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

# Need to automate this now

schedule.every().day.at("09:00").do(scrape_news)

# TODO: I am going to make a function to store the preferences of users 
# Alternate idea, make one that stores what type of news articles you click and personalize based on that
def userPreferences():
    database = []
    print("What types of tech news categories would you like to view summaries of?")
    print("Choices include 'AI', 'Quantum Computing', 'Chip Design' or Company names typed in individually please")
    print("Please type Done when you have selected all your preferences :)")
    names = input()
    do {
        database.append(names)
        names = input()
    } while (names != "Done")
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





# Steps after this - I can now ask users on what they like to give them personalised news OR I can track the articles they click
# The second one may have issues with cookies I think but would be cooler, the first one could be a quicker one for users to see results