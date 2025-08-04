# Personalised News Digest

A machine learning-based tool for providing personalized news digests to users from web-scraped data. The project uses a Random Forest model with rule-based keyword filtering to classify news articles and provide personalized summaries based on user preferences.

## Features

- **Web Scraping**: Automated collection of news articles from Google News RSS feeds
- **Advanced Classification**: Machine learning model achieving 99.97% accuracy across 41 news categories
- **Rule-Based Filtering**: Multi-category keyword-based classification for improved accuracy
- **Multi-Label Support**: Articles can be assigned multiple categories (e.g., TECH,SCIENCE)
- **Personalized Selection**: User-driven category selection for customized news digests
- **Real-Time Processing**: Live article classification and categorization

## Project Structure

```
Personalise_News_Digest_Project/
├── webscrapper.py              # Web scraping and classification pipeline
├── bert_model.py               # Machine learning model training and evaluation
├── personalised_digest.py      # User interaction and personalized digest generation
├── text_preprocessing.py       # Text preprocessing utilities
├── requirements_bert           # Python dependencies
├── model_development_summary.txt # Project progress and achievements
└── README.md                   # This file
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Amaan247788/personalised-news-digest.git
cd personalised-news-digest
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements_bert
```

## Usage

### Web Scraping and Classification

To scrape news articles and classify them:

```bash
python webscrapper.py
```

This will:
- Scrape articles from Google News RSS feeds
- Preprocess and classify articles using rule-based filtering and ML model
- Save results to timestamped CSV files with predicted categories

### Personalized News Digest

To generate a personalized news digest:

```bash
python personalised_digest.py
```

This will:
- Load the latest classified news data
- Present available categories to the user
- Allow user to select categories of interest
- Generate personalized news summaries (coming soon)

## Model Performance

The current model achieves:
- **99.97% average accuracy** across all 41 categories
- **100% accuracy** for 40 out of 41 categories
- **Rule-based filtering** for TECH, SPORTS, POLITICS, BUSINESS, ENTERTAINMENT, SCIENCE
- **Multi-label classification** support for articles matching multiple categories

## Technical Stack

- **Python 3.x**
- **scikit-learn**: Machine learning implementation
- **NLTK**: Natural language processing
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **imbalanced-learn**: Handling class imbalance

## Recent Enhancements

- Integrated web scraping pipeline using Google News RSS feeds
- Developed rule-based keyword filtering system for improved accuracy
- Enabled multi-label category assignment
- Enhanced model robustness with hybrid rule-based + ML approach
- Automated CSV output with predicted categories
- Maintained robust version control and collaborative workflow

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 