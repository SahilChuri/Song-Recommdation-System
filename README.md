# 🎶 Song Recommendation System 🎶

A personalized music recommendation system that suggests songs based on audio features, built with machine learning and displayed through an interactive Streamlit web app.

## 📚 Project Overview

The **Song Recommendation System** uses machine learning to provide song recommendations based on musical features such as *danceability*, *energy*, *acousticness*, and more. The system clusters songs with similar features and finds recommendations using Euclidean distance-based similarity. It’s designed to offer an intuitive user experience via a Streamlit web application.

---

## 🛠️ Features

- **Clustered Recommendations**: Groups songs into clusters based on audio characteristics for accurate suggestions.
- **Interactive Web Interface**: User-friendly interface powered by Streamlit.
- **Spotipy Integration**: Uses Spotipy API to fetch additional song details.

---

## 📊 Datasets

- **Primary Dataset** (`data.csv`): Contains individual song data with features like `valence`, `danceability`, and more.
- **Genre Dataset** (`data_by_genres.csv`): Aggregated genre information.
- **Year Dataset** (`data_by_year.csv`): Trends across years for analysis.

---

## 🧠 How It Works

1. **Data Preprocessing**: Loads and standardizes song data.
2. **Clustering**: Uses KMeans clustering on song features to create genre and song groups.
3. **Dimensionality Reduction**: Reduces data with TSNE and PCA for visualization.
4. **Recommendation**: Calculates Euclidean distances to recommend similar songs.
5. **Streamlit Interface**: Provides a smooth UI to interact with the recommendation engine.

---

## 🌐 Technologies

- **Python** 🐍
- **Streamlit** 🌐
- **Scikit-Learn** 📊
- **Spotipy API** 🎵

---
![image](https://github.com/user-attachments/assets/35e739fb-8b9d-4fd7-92b3-5cc4d9a50b33)
![image](https://github.com/user-attachments/assets/e96568a8-9fde-4bc2-a45a-c9e776570ab6)
