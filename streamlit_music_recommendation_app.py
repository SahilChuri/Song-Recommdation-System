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

# --- Load your data ---
data = pd.read_csv("data.csv")
genre_data = pd.read_csv('data_by_genres.csv')
year_data = pd.read_csv('data_by_year.csv')

# --- Define the number_cols globally ---
number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms',
               'energy', 'explicit', 'instrumentalness', 'key', 'liveness',
               'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

# --- Create the song cluster pipeline ---
song_cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=20, verbose=False))
])

X = data.select_dtypes(np.number)
song_cluster_pipeline.fit(X)
data['cluster_label'] = song_cluster_pipeline.predict(X)

# --- Spotify credentials ---
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='086ce9273ec149168816ac82214269e1',
                                                           client_secret='6ffa5cc37b944f63a28b820b5f7044ef'))


# --- Helper functions ---
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
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)
    song_matrix = np.array(song_vectors)
    return np.mean(song_matrix, axis=0)


def recommend_songs(song_list, spotify_data, n_songs=10):
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    song_center = get_mean_vector(song_list, spotify_data)

    available_number_cols = [col for col in number_cols if col in spotify_data.columns]

    if len(available_number_cols) == 0:
        raise ValueError("No valid columns available for scaling.")

    scaler = song_cluster_pipeline.named_steps['scaler']
    scaled_data = scaler.transform(spotify_data[available_number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))

    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    rec_songs = spotify_data.iloc[index]

    # Ensure no duplicates: filter out songs already in the input list
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]

    # Limit to n_songs while avoiding duplicates
    unique_rec_songs = rec_songs[metadata_cols].drop_duplicates().head(n_songs)

    return unique_rec_songs.to_dict(orient='records')


# --- Streamlit interface styling ---
st.markdown("""
    <style>
    /* Dark theme background */
    body {
        background-color: #1e1e1e;
        color: #ffffff;
    }

    /* Title styling */
    h1 {
        font-family: 'Roboto', sans-serif;
        color: #00ffff;
        text-align: center;
    }

    /* Input fields */
    .stTextInput input {
        border-radius: 10px;
        border: none;
        padding: 10px;
        background-color: #333333;
        color: #ffffff;
        transition: all 0.3s ease;
    }

    .stTextInput input:hover{
        background-color: #444444;
    }

    /* Recommendation cards */
    .song-card {
        background-color: #2e2e2e;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0, 255, 255, 0.2);
        border: 2px solid #00ffff;
        color: #ffffff;
    }

    /* Custom Font */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    </style>
    """, unsafe_allow_html=True)

# --- App Title ---
st.title("🎧 Music Recommendation System")

st.image("dj.jpg", use_column_width=True)  # Replace with your image path

# --- Song Input ---
song_name = st.text_input("Enter a song name:")
song_year = st.number_input("Enter the release year:", min_value=1900, max_value=2024, value=2024)

if st.button("Get Recommendations"):
    if song_name and song_year:
        try:
            recommended_songs = recommend_songs([{'name': song_name, 'year': song_year}], data)
            if recommended_songs:
                st.write("### Recommended Songs:")
                left_column, right_column = st.columns(2)

                # Distributing the songs into the left and right columns
                for i, song in enumerate(recommended_songs):
                    if i < 5:  # First five songs in the left column
                        with left_column:
                            st.markdown(f"""
                            <div class='song-card'>
                                <h4>{song['name']} ({song['year']})</h4>
                                <p><strong>Artist(s):</strong> {song['artists']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:  # Remaining songs in the right column
                        with right_column:
                            st.markdown(f"""
                            <div class='song-card'>
                                <h4>{song['name']} ({song['year']})</h4>
                                <p><strong>Artist(s):</strong> {song['artists']}</p>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.write("No recommendations found.")
        except ValueError as e:
            st.write(f"Error: {str(e)}")
    else:
        st.write("Please enter a valid song name and year.")
