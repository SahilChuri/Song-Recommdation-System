🎶 Song Recommendation System 🎶
A personalized music recommendation system that suggests songs based on audio features, built with machine learning and displayed through an interactive Streamlit web app.

📚 Project Overview
The Song Recommendation System uses machine learning to provide song recommendations based on musical features such as danceability, energy, acousticness, and more. The system clusters songs with similar features and finds recommendations using Euclidean distance-based similarity. It’s designed to offer an intuitive user experience via a Streamlit web application.

🛠️ Features
Songs Recommendations: Groups songs into clusters based on audio characteristics for accurate suggestions.
Interactive Web Interface: User-friendly interface powered by Streamlit.
Spotify Integration: Uses Spotipy API to fetch additional song details.

📊 Datasets
Primary Dataset (data.csv): Contains individual song data with features like valence, danceability, and more.
Genre Dataset (data_by_genres.csv): Aggregated genre information.
Year Dataset (data_by_year.csv): Trends across years for analysis.

🧠 How It Works
Data Preprocessing: Loads and standardizes song data.
Clustering: Uses KMeans clustering on song features to create genre and song groups.
Dimensionality Reduction: Reduces data with TSNE and PCA for visualization.
Recommendation: Calculates Euclidean distances to recommend similar songs.
Streamlit Interface: Provides a smooth UI to interact with the recommendation engine.

🌐 Technologies
Python 🐍
Streamlit 🌐
Scikit-Learn 📊
Spotipy API 🎵

📈 Future Enhancements
Dynamic Playlist Creation: Generate playlists for specific moods or genres.
User Feedback Loop: Incorporate feedback to improve recommendation accuracy.
Multilingual Support: Expand the system to handle diverse languages.

![image](https://github.com/user-attachments/assets/b7f3e730-8382-4111-a298-c873d9ec4bdb)
![image](https://github.com/user-attachments/assets/0a1693c1-7c79-4941-aa60-6c92027a510f)



