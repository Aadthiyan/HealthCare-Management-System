import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib

class RecommendationModel:
    def __init__(self, data_path, model_filename, specialist_dataset_filename, general_physician_dataset_filename):
        # Load the appointment data from the CSV file
        self.data = pd.read_csv(data_path)

        # Preprocess data and build TF-IDF vectors
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.compute_tfidf_matrix()

        # Compute cosine similarity between appointments
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

        # Load the trained model
        self.model = joblib.load(model_filename)

        # Load the dataset from a CSV file for specialists
        self.specialist_dataset = pd.read_csv(specialist_dataset_filename)
        
        # Create the condition-to-index mapping for specialists
        self.specialist_condition_to_index = {condition: index for index, condition in enumerate(self.specialist_dataset['Condition'])}
        
        # Create the list of specialist doctors
        self.specialist_doctors = self.specialist_dataset['Doctor'].tolist()

        # Load the dataset from a CSV file for general physicians
        self.general_physician_dataset = pd.read_csv(general_physician_dataset_filename)
        
        # Create the condition-to-index mapping for general physicians
        self.general_physician_condition_to_index = {condition: index for index, condition in enumerate(self.general_physician_dataset['Condition'])}
        
        # Create the list of general physicians
        self.general_physician_doctors = self.general_physician_dataset['Doctor'].tolist()

    def compute_tfidf_matrix(self):
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        return tfidf_vectorizer.fit_transform(self.data['medical_condition'])

    def get_recommendations(self, appointment_index, num_recommendations=5):
        # Get the pairwise similarity scores of all appointments with the specified appointment
        sim_scores = list(enumerate(self.cosine_sim[appointment_index]))

        # Sort the appointments based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the top-n most similar appointments
        sim_scores = sim_scores[1:num_recommendations + 1]

        # Get the appointment indices
        appointment_indices = [x[0] for x in sim_scores]

        # Return the top-n most similar appointments
        return appointment_indices

    def recommend_doctor(self, patient_condition, num_recommendations=1):
        """
        Returns a tuple: (specialist_type, doctor_name)
        """
        import difflib, random
        # 1. Try exact match in specialists
        if patient_condition in self.specialist_condition_to_index:
            idx = self.specialist_condition_to_index[patient_condition]
            specialist_type = self.specialist_dataset.iloc[idx]['Specialist'] if 'Specialist' in self.specialist_dataset.columns else 'Specialist'
            doctor_name = self.specialist_doctors[idx]
            return specialist_type, doctor_name
        # 2. Try exact match in general physicians
        if patient_condition in self.general_physician_condition_to_index:
            idx = self.general_physician_condition_to_index[patient_condition]
            specialist_type = self.general_physician_dataset.iloc[idx]['Specialist'] if 'Specialist' in self.general_physician_dataset.columns else 'General Physician'
            doctor_name = self.general_physician_doctors[idx]
            return specialist_type, doctor_name
        # 3. Fuzzy match in specialists
        close_specialist = difflib.get_close_matches(patient_condition, self.specialist_condition_to_index.keys(), n=1, cutoff=0.7)
        if close_specialist:
            idx = self.specialist_condition_to_index[close_specialist[0]]
            specialist_type = self.specialist_dataset.iloc[idx]['Specialist'] if 'Specialist' in self.specialist_dataset.columns else 'Specialist'
            doctor_name = self.specialist_doctors[idx]
            return specialist_type, doctor_name
        # 4. Fuzzy match in general physicians
        close_general = difflib.get_close_matches(patient_condition, self.general_physician_condition_to_index.keys(), n=1, cutoff=0.7)
        if close_general:
            idx = self.general_physician_condition_to_index[close_general[0]]
            specialist_type = self.general_physician_dataset.iloc[idx]['Specialist'] if 'Specialist' in self.general_physician_dataset.columns else 'General Physician'
            doctor_name = self.general_physician_doctors[idx]
            return specialist_type, doctor_name
        # 5. Use TF-IDF similarity to find closest condition in appointment data
        if hasattr(self, 'tfidf_vectorizer') and hasattr(self, 'tfidf_matrix'):
            query_vec = self.tfidf_vectorizer.transform([patient_condition])
            cosine_similarities = linear_kernel(query_vec, self.tfidf_matrix).flatten()
            top_idx = cosine_similarities.argmax()
            # Try to map the most similar condition to a doctor
            similar_condition = self.data.iloc[top_idx]['medical_condition']
            # Try specialist first
            if similar_condition in self.specialist_condition_to_index:
                idx = self.specialist_condition_to_index[similar_condition]
                specialist_type = self.specialist_dataset.iloc[idx]['Specialist'] if 'Specialist' in self.specialist_dataset.columns else 'Specialist'
                doctor_name = self.specialist_doctors[idx]
                return specialist_type, doctor_name
            # Then general physician
            if similar_condition in self.general_physician_condition_to_index:
                idx = self.general_physician_condition_to_index[similar_condition]
                specialist_type = self.general_physician_dataset.iloc[idx]['Specialist'] if 'Specialist' in self.general_physician_dataset.columns else 'General Physician'
                doctor_name = self.general_physician_doctors[idx]
                return specialist_type, doctor_name
        # 6. Fallback: random general physician
        doctor_name = random.choice(self.general_physician_doctors)
        specialist_type = 'General Physician'
        return specialist_type, doctor_name
        import difflib, random
        # 1. Try exact match in specialists
        if patient_condition in self.specialist_condition_to_index:
            idx = self.specialist_condition_to_index[patient_condition]
            return self.specialist_doctors[idx]
        # 2. Try exact match in general physicians
        if patient_condition in self.general_physician_condition_to_index:
            idx = self.general_physician_condition_to_index[patient_condition]
            return self.general_physician_doctors[idx]
        # 3. Fuzzy match in specialists
        close_specialist = difflib.get_close_matches(patient_condition, self.specialist_condition_to_index.keys(), n=1, cutoff=0.7)
        if close_specialist:
            idx = self.specialist_condition_to_index[close_specialist[0]]
            return self.specialist_doctors[idx]
        # 4. Fuzzy match in general physicians
        close_general = difflib.get_close_matches(patient_condition, self.general_physician_condition_to_index.keys(), n=1, cutoff=0.7)
        if close_general:
            idx = self.general_physician_condition_to_index[close_general[0]]
            return self.general_physician_doctors[idx]
        # 5. Use TF-IDF similarity to find closest condition in appointment data
        if hasattr(self, 'tfidf_vectorizer') and hasattr(self, 'tfidf_matrix'):
            query_vec = self.tfidf_vectorizer.transform([patient_condition])
            cosine_similarities = linear_kernel(query_vec, self.tfidf_matrix).flatten()
            top_idx = cosine_similarities.argmax()
            # Try to map the most similar condition to a doctor
            similar_condition = self.data.iloc[top_idx]['medical_condition']
            # Try specialist first
            if similar_condition in self.specialist_condition_to_index:
                idx = self.specialist_condition_to_index[similar_condition]
                return self.specialist_doctors[idx]
            # Then general physician
            if similar_condition in self.general_physician_condition_to_index:
                idx = self.general_physician_condition_to_index[similar_condition]
                return self.general_physician_doctors[idx]
        # 6. Fallback: random general physician
        return random.choice(self.general_physician_doctors)

if __name__ == "__main__":
    data_path = "recommend/data/input/appointments.csv"  # Replace with the path to your dataset
    model_filename = 'recommend/data/output/model.pkl'  # Replace with the actual filename
    specialist_dataset_filename = 'recommend/data/input/specialist.csv'  # Replace with the actual file path
    general_physician_dataset_filename = 'recommend/data/input/general.csv'  # Replace with the actual file path

    model = RecommendationModel(data_path, model_filename, specialist_dataset_filename, general_physician_dataset_filename)

    # Example: Get recommendations for a specific appointment index
    appointment_index = 5  # Replace with the index of the appointment you want recommendations for
    recommendations = model.get_recommendations(appointment_index)
    print(f"Recommendations for appointment at index {appointment_index}:")
    for recommendation_index in recommendations:
        print(f"Recommended appointment at index {recommendation_index}")

    # Example: Get doctor recommendation based on user input for patient's condition
    patient_condition = input("Enter patient's condition: ")
    recommended_doctor = model.recommend_doctor(patient_condition)
    print(f"Recommended Doctor for '{patient_condition}': {recommended_doctor}")
