import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QComboBox, QPushButton, QTextEdit, QVBoxLayout

class MusicRecommendationApp(QWidget):
    def __init__(self, df):
        super().__init__()

        self.df = df
        self.song_features = self.preprocess_data()
        self.final_similarity_matrix = self.run_pso()
        # self.song_features, self.final_similarity_matrix = self.run_pso()

        self.init_ui()

    def init_ui(self):
        # Widgets
        self.label = QLabel('Select the seed track:')
        self.track_combobox = QComboBox(self)
        self.track_combobox.addItems(self.df['track_name'].tolist())

        self.button = QPushButton('Recommend Songs', self)
        self.output_field = QTextEdit(self)
        self.output_field.setReadOnly(True)

        # Layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.track_combobox)
        layout.addWidget(self.button)
        layout.addWidget(self.output_field)

        # Event handling
        self.button.clicked.connect(self.recommend_songs)

        # Window settings
        self.setGeometry(100, 100, 400, 300)
        self.setWindowTitle('Music Recommendation App')
        self.show()

    def preprocess_data(self):
        label_encoder_artist = LabelEncoder()
        self.df['artist_encoded'] = label_encoder_artist.fit_transform(self.df['artist_name'])

        label_encoder_genre = LabelEncoder()
        self.df['genre_encoded'] = label_encoder_genre.fit_transform(self.df['genre'])

        song_features = self.df[['artist_encoded', 'genre_encoded', 'danceability', 'energy', 'loudness']].values

        return song_features

    def run_pso(self):
        num_particles = 10
        num_dimensions = self.song_features.shape[1]
        max_iterations = 50
        c1 = 2.0
        c2 = 2.0
        w = 0.7

        particles_position = np.random.rand(num_particles, num_dimensions)
        particles_velocity = np.random.rand(num_particles, num_dimensions)

        personal_best_positions = particles_position.copy()
        personal_best_fitness = np.zeros(num_particles)

        global_best_position = np.zeros(num_dimensions)
        global_best_fitness = float('inf')

        for iteration in range(max_iterations):
            for i in range(num_particles):
                fitness = self.fitness_function(particles_position[i])

                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_positions[i] = particles_position[i].copy()

                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = particles_position[i].copy()

            for i in range(num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                inertia_term = w * particles_velocity[i]
                cognitive_term = c1 * r1 * (personal_best_positions[i] - particles_position[i])
                social_term = c2 * r2 * (global_best_position - particles_position[i])
                particles_velocity[i] = inertia_term + cognitive_term + social_term
                particles_position[i] = particles_position[i] + particles_velocity[i]

        optimal_weights = global_best_position
        final_similarity_matrix = np.dot(optimal_weights * self.song_features, (optimal_weights * self.song_features).T)

        # return self.song_features, final_similarity_matrix
        return final_similarity_matrix

    def fitness_function(self, weights):
        weighted_features = weights * self.song_features
        similarity_matrix = np.dot(weighted_features, weighted_features.T)

        target_similarity_matrix = np.random.rand(similarity_matrix.shape[0], similarity_matrix.shape[0])

        fitness = np.mean((similarity_matrix - target_similarity_matrix) ** 2)

        return fitness

    def recommend_songs(self):
    
        seed_song_name = self.track_combobox.currentText().strip()

       
        seed_song_index = df[df['track_name'] == seed_song_name].index[0]
        

            # Example: Assuming num_recommendations is set to 5
        num_recommendations = 5

            # Call the function to get recommended song indices
        recommended_indices = self.recommend_song(seed_song_index, num_recommendations)
        recommended_tracks = self.df.loc[recommended_indices]['track_name'].tolist()

            # Display or use the recommended indices to retrieve song information from your dataset
        self.output_field.clear()
        self.output_field.append(f"Recommended Songs: {recommended_tracks}")

        # except IndexError:
        #     self.output_field.clear()
        #     self.output_field.append("Song not found. Please enter a valid track name.")


    def recommend_song(self, seed_song_index, num_recommendations):
        seed_song_index=int(seed_song_index)
        seed_song_similarity_scores = self.final_similarity_matrix[seed_song_index, :]
        sorted_indices = np.argsort(seed_song_similarity_scores)[::-1]
        recommended_indices = [i for i in sorted_indices if i != seed_song_index]
        top_recommendations = recommended_indices[:num_recommendations]

        return top_recommendations

if __name__ == '__main__':
    dataset_path = 'tcc_ceds_music.csv'
    df =pd.read_csv(dataset_path)

    app = QApplication(sys.argv)
    ex = MusicRecommendationApp(df)
    sys.exit(app.exec_())


