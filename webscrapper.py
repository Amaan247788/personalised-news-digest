# This file will scrape news data from Google News RSS Feeds since it is free and legally allowed

import pandas as pd
import feedparser
import schedule
import time
from datatime import datetime

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

schedule.every().day().at("09:00").do(scrape_news)





# Steps after this - I can now ask users on what they like to give them personalised news OR I can track the articles they click
# The second one may have issues with cookies I think but would be cooler, the first one could be a quicker one for users to see results