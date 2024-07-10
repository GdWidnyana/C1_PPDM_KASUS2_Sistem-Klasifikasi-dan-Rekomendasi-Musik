import streamlit as st
import torch
import torchaudio
import dill as pickle
import os
import librosa
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

  
# Define RNNClassifier class
class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNClassifier, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.rnn.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Load scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Function to extract features from an audio file
def extract_features(file_path, offset):
    try:
        y, sr = librosa.load(file_path, sr=None, offset=offset, duration=5.0)  # Load audio with offset
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None

    features = {}

    features['length'] = len(y) / sr
    features['chroma_stft_mean'] = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    features['chroma_stft_var'] = np.var(librosa.feature.chroma_stft(y=y, sr=sr))
    features['rms_mean'] = np.mean(librosa.feature.rms(y=y))
    features['rms_var'] = np.var(librosa.feature.rms(y=y))
    features['spectral_centroid_mean'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['spectral_centroid_var'] = np.var(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['spectral_bandwidth_mean'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features['spectral_bandwidth_var'] = np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features['rolloff_mean'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features['rolloff_var'] = np.var(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features['zero_crossing_rate_mean'] = np.mean(librosa.feature.zero_crossing_rate(y))
    features['zero_crossing_rate_var'] = np.var(librosa.feature.zero_crossing_rate(y))
    features['harmony_mean'] = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr))
    features['harmony_var'] = np.var(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features['perceptr_mean'] = np.mean(mfccs)
    features['perceptr_var'] = np.var(mfccs)
    features['tempo'] = librosa.beat.tempogram(y=y, sr=sr)[0, 0]

    for i in range(1, 21):
        features[f'mfcc{i}_mean'] = np.mean(mfccs[i-1])
        features[f'mfcc{i}_var'] = np.var(mfccs[i-1])

    return features

# Load the trained model
with open('the_best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Genre mapping
genre_mapping = {
    0: 'Blues', 1: 'Classical', 2: 'Country', 3: 'Disco', 4: 'Hiphop',
    5: 'Jazz', 6: 'Metal', 7: 'Pop', 8: 'Reggae', 9: 'Rock'
}

# Load the saved model and labels for music recommendation
sim_df_names = joblib.load('model_file_audio.joblib')
labels = joblib.load('genre label.joblib')
ori_scaled = joblib.load('scaler.joblib')

# Load the CSV file with genre information
data_cs = pd.read_csv('data_CS.csv')

# Function to find similar songs within the same genre
def find_similar_songs(audio_features, genre, top_n=10):
    # Filter data to include only those with the same genre
    same_genre_labels = labels[labels['label'] == genre].index
    same_genre_indices = labels.index.get_indexer(same_genre_labels)
    filtered_scaled = ori_scaled[same_genre_indices]
    filtered_labels = labels.loc[same_genre_labels]

    similarity = cosine_similarity([audio_features], filtered_scaled)
    sim_series = pd.Series(similarity.flatten(), index=filtered_labels.index)
    sim_series = sim_series.sort_values(ascending=False)
    return sim_series.head(top_n)

# Function to find similar songs across all genres
def find_similar_songs_all_genres(audio_features, top_n=10):
    similarity = cosine_similarity([audio_features], ori_scaled)
    sim_series = pd.Series(similarity.flatten(), index=labels.index)
    sim_series = sim_series.sort_values(ascending=False).drop(labels.index[0])
    return sim_series.head(top_n)

# Function to extract audio features
def extract_audio_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    harmony = librosa.effects.harmonic(y)
    perceptr = librosa.effects.percussive(y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)  # Extracting the tempo value
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    
    # Collecting feature names
    feature_names = [
        "chroma_stft_mean", "chroma_stft_var",
        "rms_mean", "rms_var",
        "spectral_centroid_mean", "spectral_centroid_var",
        "spectral_bandwidth_mean", "spectral_bandwidth_var",
        "rolloff_mean", "rolloff_var",
        "zero_crossing_rate_mean", "zero_crossing_rate_var",
        "harmony_mean", "harmony_var",
        "percussive_mean", "percussive_var",
        "tempo"
    ]
    
    mfccs_mean_names = [f"mfcc_{i}_mean" for i in range(1, 21)]
    mfccs_var_names = [f"mfcc_{i}_var" for i in range(1, 21)]
    feature_names.extend(mfccs_mean_names)
    feature_names.extend(mfccs_var_names)
    
    # Creating feature vector
    feature_vector = [
        np.mean(chroma_stft), np.var(chroma_stft),
        np.mean(rms), np.var(rms),
        np.mean(spec_cent), np.var(spec_cent),
        np.mean(spec_bw), np.var(spec_bw),
        np.mean(rolloff), np.var(rolloff),
        np.mean(zcr), np.var(zcr),
        np.mean(harmony), np.var(harmony),
        np.mean(perceptr), np.var(perceptr),
        tempo
    ]
    feature_vector.extend(np.mean(mfccs, axis=1))
    feature_vector.extend(np.var(mfccs, axis=1))
    
    # Creating dataframe
    feature_df = pd.DataFrame([feature_vector], columns=feature_names)
    
    return feature_df

# Load your trained model
model_cs = joblib.load('model_cs1')

# Fungsi untuk mengolah fitur penting
def important_features(dataset):
    data = dataset.copy()
    data["imp"] = (data["Song Name"] + ' ' + data["Artist"] + " " + data["Genre"]).str.lower()  # Convert to lowercase
    return data

# Load data
@st.cache_data  # Updated cache function
def load_data():
    data = pd.read_csv('spotify_dataset.csv')  # Pastikan path sesuai lokasi file dataset Anda
    data = important_features(data)
    data["ids"] = [i for i in range(data.shape[0])]
    return data

data = load_data()

# Function to recommend songs and show similarity scores
def recommend_songs(title):
    title_lower = title.lower()  # Convert input to lowercase to handle case insensitivity
    try:
        movie_id = data[data['Song Name'].str.lower() == title_lower]["ids"].values[0]
        scores = list(enumerate(model_cs[movie_id]))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        sorted_scores = sorted_scores[1:]  # Exclude self match
        recommendations = [(data[data["ids"] == movie[0]]["Song Name"].values[0], movie[1]) for movie in sorted_scores][:10]
        return recommendations
    except IndexError:
        return [("No match found", 0)]

# Streamlit interface
st.markdown("<h1 style='text-align: center;'>SISTEM KLASIFIKASI GENRE DAN REKOMENDASI MUSIK BERBASIS KONTEN</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>KELOMPOK C1 - PPDM UDAYANA 2024</h3>", unsafe_allow_html=True)
st.write("----------------------------------------------------------------")

# Add radio button for menu selection
menu = st.radio("Pilih Menu:", ["Klasifikasi Genre Musik", "Rekomendasi Lagu berdasarkan File Audio", "Rekomendasi Lagu berdasarkan Judul Lagu"])

st.write("----------------------------------------------------------------")

if menu == "Klasifikasi Genre Musik":
    st.markdown("<h2 style='text-align: center;'>Klasifikasi Genre Musik</h2>", unsafe_allow_html=True)
    st.write('Upload file audio untuk memprediksi genre.')
    st.write('Silakan pilih file lagu dari link drive ini : https://drive.google.com/drive/u/2/folders/1t7Z5_pac5-cvVK2NR6tiY5ddSOmiqJXh.')

    # Allow multiple file uploads
    uploaded_files = st.file_uploader("Pilih file audio...", type=["wav","mp3"], accept_multiple_files=True)

    if st.button('Prediksi'):
        results = []

        for uploaded_file in uploaded_files:
            with open("temp.wav", "wb") as f:
                f.write(uploaded_file.getbuffer())

            features = extract_features("temp.wav", 10)
            if features is not None:
                features_arr = scaler.transform([list(features.values())])
                features_tensor = torch.tensor(features_arr, dtype=torch.float32)

                if torch.cuda.is_available():
                    features_tensor = features_tensor.to('cuda')
                    model = model.to('cuda')

                model.eval()
                with torch.no_grad():
                    outputs = model(features_tensor.unsqueeze(1))
                    _, predicted = torch.max(outputs, 1)
                    predicted_genre = genre_mapping[predicted.item()]

                results.append({'Nama File': uploaded_file.name, 'Genre yang Diprediksi': predicted_genre})

        if results:
            df_results = pd.DataFrame(results)
            st.write(df_results)

elif menu == "Rekomendasi Lagu berdasarkan File Audio":
    st.markdown("<h2 style='text-align: center;'>Rekomendasi Musik Berdasarkan Audio</h2>", unsafe_allow_html=True)
    st.write("----------------------------------------------------------------")
    st.write('Silakan pilih file lagu dari link drive ini : https://drive.google.com/drive/u/2/folders/1t7Z5_pac5-cvVK2NR6tiY5ddSOmiqJXh.')
    uploaded_file = st.file_uploader("Upload file audio", type=["wav", "mp3"])

    if uploaded_file is not None:
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        feature_df = extract_audio_features("temp_audio.wav")
        
        st.write("### Fitur Audio yang Dihasilkan:")
        st.write(feature_df)
        
        file_name = uploaded_file.name
        input_genre = data_cs[data_cs['filename'] == file_name]['label'].values[0]
        
        st.write(f"### Genre Terdeteksi: {input_genre}")
        
        st.write("### 10 Lagu Serupa dalam Genre yang Sama:")
        similar_songs = find_similar_songs(feature_df.values.flatten(), input_genre)
        
        for idx, (filename, similarity) in enumerate(similar_songs.items(), start=1):
            st.write(f"{idx}. {filename} - Similaritas: {similarity:.4f}")
        
        st.audio(uploaded_file)
        
        if st.button('Dapatkan rekomendasi dari genre lain'):
            st.write("### 10 Lagu Serupa dari Genre Lain:")
            similar_songs_all_genres = find_similar_songs_all_genres(feature_df.values.flatten())
            
            for idx, (filename, similarity) in enumerate(similar_songs_all_genres.items(), start=1):
                st.write(f"{idx}. {filename} - Similaritas: {similarity:.4f}")

elif menu == "Rekomendasi Lagu berdasarkan Judul Lagu":
    st.markdown("<h2 style='text-align: center;'>Rekomendasi Lagu dari Judul Lagu</h2>", unsafe_allow_html=True)
    st.write("----------------------------------------------------------------")
    st.write('Silakan pilih judul lagu dari link excel ini : https://docs.google.com/spreadsheets/d/10SxMm-1A02HR7v14PUt2KrnWCdN4AbQH/edit?usp=sharing&ouid=115051467428069348225&rtpof=true&sd=true')
# Add selectbox for input method selection
    input_method = st.selectbox("Pilih metode input:", ["Input Manual", "Upload File Excel"])

    if input_method == "Input Manual":
        st.subheader("Input Manual")
        song_name = st.text_input("Masukkan nama lagu", "Old Town Road")
        if st.button('Rekomendasikan Lagu'):
            recommendations = recommend_songs(song_name)
            for i, (song, score) in enumerate(recommendations):
                st.write(f"{i+1}. {song} (Skor Similaritas: {score:.3f})")

    elif input_method == "Upload File Excel":
        st.subheader("Upload File Excel")
        uploaded_file = st.file_uploader("Pilih file Excel (.xlsx)", type='xlsx')
        if uploaded_file:
            df_uploaded = pd.read_excel(uploaded_file)
            if "Judul Lagu" in df_uploaded.columns:
                recommendations_list = []
                for title in df_uploaded["Judul Lagu"]:
                    recommendations = recommend_songs(title)
                    recommendations_list.append({
                        "Judul Lagu": title,
                        "Rekomendasi": ", ".join([song for song, score in recommendations])
                    })
                df_recommendations = pd.DataFrame(recommendations_list)
                st.write(df_recommendations)
        
                st.download_button(
                    label="Download hasil rekomendasi sebagai CSV",
                    data=df_recommendations.to_csv(index=False).encode('utf-8'),
                    file_name='hasil_rekomendasi.csv',
                    mime='text/csv',
                )
            else:
                st.error("Kolom 'Judul Lagu' tidak ditemukan di file Excel. Harap periksa file Anda.")
    
