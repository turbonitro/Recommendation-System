import pandas as pd
import keyboard
from sklearn.neighbors import NearestNeighbors
import sys

# Wczytanie danych z plików
ratings_file_path = r'\user-artist-rating-averages-sorted.xlsx'
sentiment_file_path = r'\sentiment_analysis_results_english.csv'

# Wczytanie danych
user_artist_df = pd.read_excel(ratings_file_path)
sentiment_df = pd.read_csv(sentiment_file_path, sep=';', encoding='utf-8-sig')

# Przekształcenie danych do macierzy użytkownik-artysta
user_artist_matrix = user_artist_df.pivot_table(index='user', columns='artist', values='rating').fillna(0)

# Tworzymy model najbliższego sąsiada
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_artist_matrix)


# Funkcja do znalezienia najbardziej podobnych użytkowników do danego użytkownika, z wykluczeniem samego siebie
def find_similar_users(user_id, n_neighbors=3):
    user_vector = user_artist_matrix.loc[user_id].values.reshape(1, -1)
    distances, indices = model_knn.kneighbors(user_vector,
                                              n_neighbors=n_neighbors + 1)  # Zwiększamy liczbę sąsiadów o 1, żeby wykluczyć samego siebie

    # Odwracamy wartości podobieństwa (1 - wartość kosinusowa)
    similarity_scores = 1 - distances.flatten()
    similar_users = user_artist_matrix.index[indices.flatten()].tolist()

    # Usunięcie samego siebie z listy podobnych użytkowników
    filtered_similar_users = [(u, s) for u, s in zip(similar_users, similarity_scores) if u != user_id and s >= 0.06]

    return filtered_similar_users[:n_neighbors]


# Funkcja do znalezienia najlepiej ocenianych pozycji dla danego typu z hierarchią priorytetów
def find_best_recommendation_for_type(user_entries, content_type, user_existing_titles, previous_recommendations):
    # Filtracja pozycji według typu
    type_entries = user_entries[user_entries['type'] == content_type]

    # Łączna lista wykluczeń (pozycje już istniejące u użytkownika i wcześniej rekomendowane)
    exclusion_list = user_existing_titles + previous_recommendations

    # Szukanie w kolejności priorytetów:
    # 1. Ocena 5, Sentyment POSITIVE
    positive_high_rated_entries = type_entries[
        (type_entries['rating'] == 5) & (type_entries['sentiment_label'] == 'POSITIVE')]
    positive_high_rated_entries = positive_high_rated_entries[
        ~positive_high_rated_entries['title'].isin(exclusion_list)]
    if not positive_high_rated_entries.empty:
        return positive_high_rated_entries.iloc[0][['title', 'artist']]

    # 2. Ocena 5, dowolny sentyment
    high_rated_entries = type_entries[type_entries['rating'] == 5]
    high_rated_entries = high_rated_entries[~high_rated_entries['title'].isin(exclusion_list)]
    if not high_rated_entries.empty:
        return high_rated_entries.iloc[0][['title', 'artist']]

    # 3. Ocena 4, Sentyment POSITIVE
    positive_mid_rated_entries = type_entries[
        (type_entries['rating'] == 4) & (type_entries['sentiment_label'] == 'POSITIVE')]
    positive_mid_rated_entries = positive_mid_rated_entries[~positive_mid_rated_entries['title'].isin(exclusion_list)]
    if not positive_mid_rated_entries.empty:
        return positive_mid_rated_entries.iloc[0][['title', 'artist']]

    # 4. Ocena 4, dowolny sentyment
    mid_rated_entries = type_entries[type_entries['rating'] == 4]
    mid_rated_entries = mid_rated_entries[~mid_rated_entries['title'].isin(exclusion_list)]
    if not mid_rated_entries.empty:
        return mid_rated_entries.iloc[0][['title', 'artist']]

    # Jeśli żadna z powyższych opcji nie była dostępna
    return None


# Funkcja do znalezienia najpopularniejszych pozycji w bazie dla danego typu
def find_most_popular_for_type(content_type, user_existing_titles, previous_recommendations):
    type_entries = sentiment_df[sentiment_df['type'] == content_type]

    # Zliczanie popularności tytułów
    popular_titles = type_entries['title'].value_counts().index.tolist()

    # Łączna lista wykluczeń (pozycje już istniejące u użytkownika i wcześniej rekomendowane)
    exclusion_list = user_existing_titles + previous_recommendations

    # Filtracja tytułów, które użytkownik już posiada lub były już rekomendowane
    popular_titles = [title for title in popular_titles if title not in exclusion_list]

    # Zwrócenie najbardziej popularnego tytułu, który nie jest jeszcze w kolekcji użytkownika
    if popular_titles:
        most_popular_entry = type_entries[type_entries['title'] == popular_titles[0]].iloc[0]
        return most_popular_entry[['title', 'artist']]
    return None


# Główna funkcja do generowania rekomendacji
def generate_recommendations(user_id, previous_recommendations):
    similar_users = find_similar_users(user_id, n_neighbors=3)

    if not similar_users:
        print(
            "Nie znaleziono wystarczająco podobnych użytkowników. Generowanie rekomendacji na podstawie najpopularniejszych tytułów.")
        similar_users = []  # Ustawiamy listę podobnych użytkowników na pustą

    # Pobieramy istniejące tytuły użytkownika
    user_existing_titles = sentiment_df[sentiment_df['user'] == user_id]['title'].tolist()

    recommendations = {'book': None, 'movie': None, 'music': None}

    # Sprawdzenie rekomendacji dla każdego typu
    for content_type in recommendations.keys():
        recommendation_found = False
        for similar_user in similar_users:
            similar_user_entries = sentiment_df[sentiment_df['user'] == similar_user[0]]
            recommendation = find_best_recommendation_for_type(similar_user_entries, content_type, user_existing_titles,
                                                               previous_recommendations)

            if recommendation is not None:
                recommendations[content_type] = recommendation
                recommendation_found = True
                break  # Przerywamy pętlę, gdy znajdziemy odpowiednią rekomendację

        if not recommendation_found:
            # Jeśli nie znaleziono rekomendacji na podstawie podobnych użytkowników, wybierz najpopularniejszy tytuł
            recommendations[content_type] = find_most_popular_for_type(content_type, user_existing_titles,
                                                                       previous_recommendations)

    # Aktualizacja listy poprzednich rekomendacji
    for content_type in recommendations.keys():
        if recommendations[content_type] is not None:
            previous_recommendations.append(recommendations[content_type]['title'])

    # Wyświetlenie rekomendacji
    print("\nRekomendacje:")
    for content_type in recommendations.keys():
        if recommendations[content_type] is not None:
            title = recommendations[content_type]['title']
            artist = recommendations[content_type]['artist']
            if content_type == 'book':
                print(f"Rekomendowana książka: {title} autorstwa {artist}")
            elif content_type == 'movie':
                print(f"Rekomendowany film: {title} reżyserii {artist}")
            elif content_type == 'music':
                print(f"Rekomendowany album muzyczny: {title} autorstwa {artist}")
        else:
            print(f"Rekomendowana {content_type}: none")


# Funkcja główna obsługująca ponowne uruchamianie programu
def main():
    # Pytanie o username
    user_id = input("Cześć, jaki jest twój Username? ").strip()

    if user_id not in sentiment_df['user'].unique():
        print(f"Użytkownik {user_id} nie istnieje w bazie danych.")
        return

    previous_recommendations = []

    while True:
        # Sprawdzanie, czy naciśnięto klawisz Esc, aby wyłączyć program
        if keyboard.is_pressed('esc'):
            print("\nNaciśnięto Esc. Program zostanie zamknięty.")
            sys.exit()

        generate_recommendations(user_id, previous_recommendations)

        # Sprawdzamy, czy użytkownik chce wygenerować kolejne rekomendacje
        repeat = input("\nCzy chcesz wygenerować kolejne rekomendacje? (tak/nie): ").strip().lower()
        if repeat != 'tak':
            break


if __name__ == "__main__":
    main()
