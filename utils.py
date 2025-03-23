import requests
import string
import re 
import spacy
import yake
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
from newspaper import Article
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.summarizers.lsa import  LsaSummarizer
from sumy.utils import get_stop_words
from sklearn.feature_extraction.text import  TfidfVectorizer
from gtts import gTTS
from googletrans import Translator
nltk.download('punkt')
nltk.download('vader_lexicon')

# Load spaCy NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

class NewsAnalysis:
    def __init__(self, company_name):
        self.company_name = company_name

    def search_bbc_direct(self):
        """Query BBC's search page and extract news article URLs with exact phrase matching."""
    
        # BBC Search URLs (UK & Global)
        search_urls = [
            f'https://www.bbc.co.uk/search?q="{self.company_name}"',
            f'https://www.bbc.com/search?q="{self.company_name}"'
        ]

        headers = {"User-Agent": "Mozilla/5.0"}
        articles = set()  # Use a set to store unique URLs

        for url in search_urls:
            try:
                response = requests.get(url, headers=headers)
                
                # Check if request was successful
                if response.status_code != 200:
                    print(f"Failed to retrieve data from: {url}")
                    continue  # Move to the next URL if this one fails

                soup = BeautifulSoup(response.text, "html.parser")

                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    if not href.startswith("http"):
                        href = "https://www.bbc.com" + href
                    
                    # Extract only BBC News URLs containing "articles" or "business"
                    if href.split('/')[-1].isalnum() and  ("articles" in href or "business" in href):
                        articles.add(href)


                if len(articles) >= 10:  # Stop if we already have 10 articles
                    break
            except requests.exceptions.RequestException as e:
                print(f"❌ Network error while fetching BBC articles: {e}")
                continue


        return list(articles) if articles else [] # Return up to 10 unique links`
    
    def scrape_article(self,url):
        """Extracts full text from a news article URL"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            if not article.text.strip():
                raise ValueError("Scraped content is empty")
            return {
                "title": article.title if article.title else "No Title",
                "content": article.text,
                "summary": self.summarizer(article.text),
                'url':url,
                'sentiment': self.sentiment_analysis(article.text),
                'topics': self.extract_topics(article.text)
            }
        except Exception as e:
            print(f"Error scraping article {url}: {e}")
            return None
    
    def summarizer(self,paragraph, sentence_count=3):
        """Summarize article using LSA, with error handling."""
        try:

            parser = PlaintextParser.from_string(paragraph,Tokenizer('english'))

            lsa_summarizer = LsaSummarizer(stemmer=Stemmer('english'))
            lsa_summarizer.stop_words = get_stop_words('english')

            summary = lsa_summarizer(parser.document,sentence_count)

            return ' '.join([str(sentence) for sentence in summary])
        except Exception as e:
            print(f"Error summarizing article: {e}")
            return paragraph[:300]
    
    def sentiment_analysis(self,text):
        """Analyzes the sentiment of a given text."""
        try:

            sid = SentimentIntensityAnalyzer()
            sentiment_scores = sid.polarity_scores(text)
            if sentiment_scores["compound"] >= 0.05:
                sentiment = "Positive"
            elif sentiment_scores["compound"] <= -0.05:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            return sentiment
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return "Neutral"
    
    @staticmethod 
    def preprocess(text):
        """Preprocess text: remove HTML tags, emojis, stopwords, and lemmatize."""
        try:
            # Remove HTML tags
            text = BeautifulSoup(text, "html.parser").get_text()

            # Remove emojis using regex
            text = re.sub(r'[^\w\s]', '', text)  # Keeps only words, spaces
            
            # Tokenize
            tokens = word_tokenize(text)

            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word.lower() not in stop_words]

            # Lemmatize words
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]

            return ' '.join(tokens)
        except Exception as e:
            print(f"Error in text preprocessing: {e}")
            return text  # Return unprocessed text as fallback

    
    def extract_topics(self,text, top_n=5):
        """Extracts key topics from news content using Named Entity Recognition (NER), Keyword Extraction (YAKE), and TF-IDF."""
        try:
            preprocess_text = self.preprocess(text)
            # Extract Named Entities using spaCy
            doc = nlp(preprocess_text)
            named_entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "GPE", "EVENT"]]

            # Extract Keywords using YAKE (Now with Multi-word Phrases)
            kw_extractor = yake.KeywordExtractor(n=2, top=5)  # Extract top 5 multi-word keywords
            yake_keywords = [kw[0] for kw in kw_extractor.extract_keywords(preprocess_text)]


            tfidf_keywords = self.extract_tfidf_keywords(preprocess_text)

            # Clean & Merge Topics
            all_topics = set(named_entities + yake_keywords + tfidf_keywords)
            all_topics = [topic for topic in all_topics if topic.lower() not in string.punctuation]  # Remove punctuation

            return all_topics[:top_n]  # Return top N topics
        except Exception as e:
            print(f"Error extracting topics: {e}")
            return []
        
    @staticmethod
    def extract_tfidf_keywords(text, num_keywords=5):
        # Extract Keywords using TF-IDF (For Deeper Context)
        try:

            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))  # Allow unigrams and bigrams
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            tfidf_keywords = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
            return [kw[0] for kw in tfidf_keywords[:num_keywords]]
        except Exception as e:
            print(f"Error extracting TF-IDF keywords: {e}")
            return []
    
    def compare_coverage(self,sentiment_analysis):
        """Finds key differences in sentiment distribution and topic overlap."""
        
        positive_topics = set()
        negative_topics = set()

        for article in sentiment_analysis:
            if article["sentiment"] == "Positive":
                positive_topics.update(article["topics"])
            elif article["sentiment"] == "Negative":
                negative_topics.update(article["topics"])

        # Find common & unique topics
        common_topics = positive_topics.intersection(negative_topics)
        unique_positive = positive_topics - negative_topics
        unique_negative = negative_topics - positive_topics

        return {
            "common_topics": list(common_topics),
            "unique_positive_topics": list(unique_positive),
            
            "unique_negative_topics": list(unique_negative)
        }
    
   
    def text_to_speech(self,summary, filename="summary.mp3"):
        """Convert the summary to Hindi speech and save as an audio file."""
        try:
            text = Translator().translate(summary, dest="hi").text
            tts = gTTS(text=text, lang='hi')
            tts.save(filename)
            return filename
        except Exception as e:
            print(f"Error in text-to-speech conversion: {e}")
            return None

    
    def generate_report_summary(self,sentiment_counts, topic_comparison):
        """Generates a natural language summary of sentiment trends and coverage differences."""

        try:

            # Step 1: Determine Overall Sentiment Trend
            total_articles = sum(sentiment_counts.values())
            most_common_sentiment = max(sentiment_counts, key=sentiment_counts.get)
            
            if sentiment_counts["Positive"] > sentiment_counts["Negative"]:
                trend_summary = f"The majority of news articles are positive, highlighting good news for the company. "
            elif sentiment_counts["Negative"] > sentiment_counts["Positive"]:
                trend_summary = f"Most news articles have a negative tone, indicating challenges faced by the company. "
            else:
                trend_summary = f"The news coverage is evenly distributed between positive and negative sentiments. "

            # Step 2: Compare Topic Coverage
            positive_topics = topic_comparison.get("unique_positive_topics", [])
            negative_topics = topic_comparison.get("unique_negative_topics", [])
            common_topics = topic_comparison.get("common_topics", [])

            topic_summary = "The key themes in the news coverage include: "
            if common_topics:
                topic_summary += f"Common topics across all articles include {', '.join(common_topics)}. "
            if positive_topics:
                topic_summary += f"Positive articles focus on {', '.join(positive_topics)}. "
            if negative_topics:
                topic_summary += f"Meanwhile, negative articles highlight concerns related to {', '.join(negative_topics)}."

            # Step 3: Generate Final Analysis
            final_summary = f"{trend_summary}{topic_summary} In summary, the company's latest news is mostly {most_common_sentiment}."

            return final_summary
        except Exception as e:
            print(f"Error generating report summary: {e}")
            return ""
    
    def generate_comparative_report(self,articles):
        """Generates a structured comparative sentiment analysis report."""

        try:


            sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
            for article in articles:
                sentiment_counts[article["sentiment"]] += 1
            
            topic_comparison = self.compare_coverage(articles)

            summary = self.generate_report_summary(sentiment_counts, topic_comparison)  # Generate a natural language summary

            report = {
                "Sentiment Distribution": sentiment_counts,
                "Coverage Differences": topic_comparison,
                "Final Sentiment Analysis": summary,
                "Audio": self.text_to_speech(summary)  # Convert summary to Hindi speech

            }

            return report
        except Exception as e:
            print(f"Error generating comparative report: {e}")
            return {}
    
    def main(self):
        """Main function to fetch news articles, analyze sentiment, and generate a comparative report."""
        
        # Step 1: Search for news articles
        article_urls = self.search_bbc_direct()
        articles = [self.scrape_article(url) for url in article_urls if url]
        
        articles = [article for article in articles if article]  # Remove None values

        final_report = {
            'articles':articles,
            'report':self.generate_comparative_report(articles)
        }

        return final_report



    

    

