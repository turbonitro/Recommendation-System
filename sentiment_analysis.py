#Analiza sentymentu
import pandas as pd
from googletrans import Translator
from transformers import pipeline

# Wczytanie pliku CSV
file_path = '/all_users_table.csv'
df = pd.read_csv(file_path, sep=';', encoding='utf-8')

# Filtracja komentarzy, które nie są puste
comments = df['comment'].dropna()

# Inicjalizacja tłumacza
translator = Translator()

# Tłumaczenie komentarzy na język angielski
def translate_to_english(text):
    translation = translator.translate(text, dest='en')
    return translation.text

comments_english = comments.apply(translate_to_english)

# Analiza sentymentu za pomocą modelu DistilBERT dla języka angielskiego
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Przeprowadzenie analizy sentymentu
sentiments = comments_english.apply(lambda x: sentiment_analyzer(x)[0])

# Dodanie wyników analizy sentymentu do oryginalnego DataFrame
df.loc[comments_english.index, 'translated_comment'] = comments_english
df.loc[comments_english.index, 'sentiment_label'] = sentiments.apply(lambda x: x['label'])
df.loc[comments_english.index, 'sentiment_score'] = sentiments.apply(lambda x: x['score'])

# Wyświetlenie wyników
print(df[['comment', 'translated_comment', 'sentiment_label', 'sentiment_score']].head())

# Zapisanie wyników do nowego pliku CSV z obsługą polskich znaków
df.to_csv('/sentiment_analysis_results_english.csv', sep=';', index=False, encoding='utf-8-sig')
