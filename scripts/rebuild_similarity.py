import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def main() -> None:
    # Load medicines dict produced earlier
    medicines_dict = pickle.load(open('medicine_dict.pkl', 'rb'))
    medicines = pd.DataFrame(medicines_dict)

    if 'Drug_Name' not in medicines.columns:
        raise RuntimeError("medicine_dict.pkl must contain a 'Drug_Name' column")

    names = medicines['Drug_Name'].astype(str).fillna('')

    # Vectorize names (you may switch to TF-IDF if you prefer)
    vectorizer = CountVectorizer(token_pattern=r"[A-Za-z0-9_\-]+")
    X = vectorizer.fit_transform(names)

    # Compute pairwise cosine similarity
    sim = cosine_similarity(X)

    # Save as pickle for app consumption
    with open('similarity.pkl', 'wb') as f:
        pickle.dump(sim, f)

    print(f"similarity.pkl written. Shape: {sim.shape}")


if __name__ == '__main__':
    main()


