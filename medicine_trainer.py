import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

df = pd.read_csv("recommend/data/input/final.csv")

X = df.drop('drug', axis=1)
y = df['drug']

X=X.drop('Unnamed: 0',axis=1)

encoder = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['disease'])], remainder='passthrough')
x_encoded= encoder.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(x_encoded,y,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=10, random_state=42)
# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
# y_pred = model.predict(X_test)

# Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred)


# Save the trained model
model_filename = 'recommend/data/output/medi_model.pkl'  # Choose a filename
joblib.dump(model, model_filename)

# Print a message after training
print("Model has been trained & successfully and saved as", model_filename)
