import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Load your data
data = pd.read_csv("data.csv")
genre_data = pd.read_csv('data_by_genres.csv')
year_data = pd.read_csv('data_by_year.csv')

# Define the number_cols outside of functions for global access
number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms',
               'energy', 'explicit', 'instrumentalness', 'key', 'liveness',
               'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

# Create the song cluster pipeline (this should be done after loading the data)
song_cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=20, verbose=False))
])

# Fit the pipeline on your data (assumes X is your feature DataFrame)
X = data.select_dtypes(np.number)
song_cluster_pipeline.fit(X)
data['cluster_label'] = song_cluster_pipeline.predict(X)

# Spotipy client initialization
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='086ce9273ec149168816ac82214269e1',
                                                           client_secret='6ffa5cc37b944f63a28b820b5f7044ef'))


def flatten_dict_list(dict_list):
    flattened_dict = defaultdict(list)
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
    return flattened_dict


def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q='track: {} year: {}'.format(name, year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)


def get_song_data(song, spotify_data):
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) & (spotify_data['year'] == song['year'])].iloc[
            0]
        return song_data
    except IndexError:
        return find_song(song['name'], song['year'])


def get_mean_vector(song_list, spotify_data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print(f'Warning: {song["name"]} does not exist in Spotify or in the database')
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)
    song_matrix = np.array(song_vectors)
    return np.mean(song_matrix, axis=0)


def recommend_songs(song_list, spotify_data, n_songs=10):
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    song_center = get_mean_vector(song_list, spotify_data)

    # Get only available numerical columns for scaling
    available_number_cols = [col for col in number_cols if col in spotify_data.columns]

    # Check if there's data to scale
    if len(available_number_cols) == 0:
        raise ValueError("No valid columns available for scaling.")

    scaler = song_cluster_pipeline.named_steps['scaler']
    scaled_data = scaler.transform(spotify_data[available_number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))

    # Compute distances for recommendations
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]

    return rec_songs[metadata_cols].to_dict(orient='records')


# Streamlit interface
st.title("Music Recommendation System")

# Input for user song
song_name = st.text_input("Enter a song name:")
song_year = st.number_input("Enter the release year:", min_value=1900, max_value=2024, value=2024)

if st.button("Get Recommendations"):
    if song_name and song_year:
        try:
            recommended_songs = recommend_songs([{'name': song_name, 'year': song_year}], data)
            if recommended_songs:
                st.write("Recommended Songs:")
                for song in recommended_songs:
                    st.write(f"{song['name']} ({song['year']}) by {song['artists']}")
            else:
                st.write("No recommendations found.")
        except ValueError as e:
            st.write(f"Error: {str(e)}")
    else:
        st.write("Please enter a valid song name and year.")
