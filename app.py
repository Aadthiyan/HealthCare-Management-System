
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
import re
import secrets
from config import SQLALCHEMY_DATABASE_URI
from secret import SECRET_KEY
import datetime
import pandas as pd
import csv
import joblib
from werkzeug.datastructures import ImmutableMultiDict
from recommend.doctor import RecommendationModel
import pandas
import pickle
from email.message import EmailMessage
import ssl
import smtplib
import random
from flask import logging
import pandas as pd
import numpy as np
from joblib import load
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import msgConstant as msgCons
import re
import os

# Initialize Flask app first
app = Flask(__name__)
app.secret_key = SECRET_KEY

# Configure PostgreSQL Database using environment variables
PG_HOST = os.environ.get('PG_HOST', 'localhost')
PG_PORT = os.environ.get('PG_PORT', '5432')
PG_DB = os.environ.get('PG_DB', 'healthcare')
PG_USER = os.environ.get('PG_USER', 'postgres')
PG_PASSWORD = os.environ.get('PG_PASSWORD', 'root')

# Configure database URI
app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy with app
db = SQLAlchemy(app)

# Load the trained model (optional in serverless deploys)
model_filename = 'recommend/data/output/medi_model.pkl'
try:
    model = joblib.load(model_filename)
except Exception as e:
    print(f"Warning: unable to load medi_model.pkl: {e}")
    model = None

#______________________
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

def make_token():
    """
    Creates a cryptographically-secure, URL-safe string
    """
    return secrets.token_urlsafe(16) 

all_result = {
    'name':'',
    'age':0,
    'gender':'',
    'symptoms':[]
}

def predict_symptom(user_input, symptom_list):
    # Convert user input to lowercase and split into tokens
    user_input_tokens = user_input.lower().replace("_"," ").split()

    # Calculate cosine similarity between user input and each symptom
    similarity_scores = []
    for symptom in symptom_list:
        # Convert symptom to lowercase and split into tokens
        symptom_tokens = symptom.lower().replace("_"," ").split()

        # Create count vectors for user input and symptom
        count_vector = np.zeros((2, len(set(user_input_tokens + symptom_tokens))))
        for i, token in enumerate(set(user_input_tokens + symptom_tokens)):
            count_vector[0][i] = user_input_tokens.count(token)
            count_vector[1][i] = symptom_tokens.count(token)

        # Calculate cosine similarity between count vectors
        similarity = cosine_similarity(count_vector)[0][1]
        similarity_scores.append(similarity)

    # Return symptom with highest similarity score
    max_score_index = np.argmax(similarity_scores)
    return symptom_list[max_score_index]

# Load the dataset into a pandas dataframe
df = pd.read_excel('dataset.xlsx')

# Get all unique symptoms
symptoms = set()
for s in df['Symptoms']:
    for symptom in s.split(','):
        symptoms.add(symptom.strip())

def predict_disease_from_symptom(symptom_list):
    user_symptoms = symptom_list
    # Vectorize symptoms using CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Symptoms'])
    user_X = vectorizer.transform([', '.join(user_symptoms)])

    # Compute cosine similarity between user symptoms and dataset symptoms
    similarity_scores = cosine_similarity(X, user_X)

    # Find the most similar disease(s)
    max_score = similarity_scores.max()
    max_indices = similarity_scores.argmax(axis=0)
    diseases = set()
    for i in max_indices:
        if similarity_scores[i] == max_score:
            diseases.add(df.iloc[i]['Disease'])

    # Output results
    if len(diseases) == 0:
        return "<b>No matching diseases found</b>",""
    elif len(diseases) == 1:
        print("The most likely disease is:", list(diseases)[0])
        disease_details = getDiseaseInfo(list(diseases)[0])
        return f"<b>{list(diseases)[0]}</b><br>{disease_details}",list(diseases)[0]
    else:
        return "The most likely diseases are<br><b>"+ ', '.join(list(diseases))+"</b>",""

    symptoms_dict = {'itching': 0, 'skin_rash': 0, 'nodal_skin_eruptions': 0, 'continuous_sneezing': 0,
                'shivering': 0, 'chills': 0, 'joint_pain': 0, 'stomach_pain': 0, 'acidity': 0, 'ulcers_on_tongue': 0,
                'muscle_wasting': 0, 'vomiting': 0, 'burning_micturition': 0, 'spotting_ urination': 0, 'fatigue': 0,
                'weight_gain': 0, 'anxiety': 0, 'cold_hands_and_feets': 0, 'mood_swings': 0, 'weight_loss': 0,
                'restlessness': 0, 'lethargy': 0, 'patches_in_throat': 0, 'irregular_sugar_level': 0, 'cough': 0,
                'high_fever': 0, 'sunken_eyes': 0, 'breathlessness': 0, 'sweating': 0, 'dehydration': 0,
                'indigestion': 0, 'headache': 0, 'yellowish_skin': 0, 'dark_urine': 0, 'nausea': 0, 'loss_of_appetite': 0,
                'pain_behind_the_eyes': 0, 'back_pain': 0, 'constipation': 0, 'abdominal_pain': 0, 'diarrhoea': 0, 'mild_fever': 0,
                'yellow_urine': 0, 'yellowing_of_eyes': 0, 'acute_liver_failure': 0, 'fluid_overload': 0, 'swelling_of_stomach': 0,
                'swelled_lymph_nodes': 0, 'malaise': 0, 'blurred_and_distorted_vision': 0, 'phlegm': 0, 'throat_irritation': 0,
                'redness_of_eyes': 0, 'sinus_pressure': 0, 'runny_nose': 0, 'congestion': 0, 'chest_pain': 0, 'weakness_in_limbs': 0,
                'fast_heart_rate': 0, 'pain_during_bowel_movements': 0, 'pain_in_anal_region': 0, 'bloody_stool': 0,
                'irritation_in_anus': 0, 'neck_pain': 0, 'dizziness': 0, 'cramps': 0, 'bruising': 0, 'obesity': 0, 'swollen_legs': 0,
                'swollen_blood_vessels': 0, 'puffy_face_and_eyes': 0, 'enlarged_thyroid': 0, 'brittle_nails': 0, 'swollen_extremeties': 0,
                'excessive_hunger': 0, 'extra_marital_contacts': 0, 'drying_and_tingling_lips': 0, 'slurred_speech': 0,
                'knee_pain': 0, 'hip_joint_pain': 0, 'muscle_weakness': 0, 'stiff_neck': 0, 'swelling_joints': 0, 'movement_stiffness': 0,
                'spinning_movements': 0, 'loss_of_balance': 0, 'unsteadiness': 0, 'weakness_of_one_body_side': 0, 'loss_of_smell': 0,
                'bladder_discomfort': 0, 'foul_smell_of urine': 0, 'continuous_feel_of_urine': 0, 'passage_of_gases': 0, 'internal_itching': 0,
                'toxic_look_(typhos)': 0, 'depression': 0, 'irritability': 0, 'muscle_pain': 0, 'altered_sensorium': 0,
                'red_spots_over_body': 0, 'belly_pain': 0, 'abnormal_menstruation': 0, 'dischromic _patches': 0, 'watering_from_eyes': 0,
                'increased_appetite': 0, 'polyuria': 0, 'family_history': 0, 'mucoid_sputum': 0, 'rusty_sputum': 0, 'lack_of_concentration': 0,
                'visual_disturbances': 0, 'receiving_blood_transfusion': 0, 'receiving_unsterile_injections': 0, 'coma': 0,
                'stomach_bleeding': 0, 'distention_of_abdomen': 0, 'history_of_alcohol_consumption': 0, 'fluid_overload.1': 0,
                'blood_in_sputum': 0, 'prominent_veins_on_calf': 0, 'palpitations': 0, 'painful_walking': 0, 'pus_filled_pimples': 0,
                'blackheads': 0, 'scurring': 0, 'skin_peeling': 0, 'silver_like_dusting': 0, 'small_dents_in_nails': 0, 'inflammatory_nails': 0,
                'blister': 0, 'red_sore_around_nose': 0, 'yellow_crust_ooze': 0}
    
    # Set value to 1 for corresponding symptoms
    for s in symptom_list:
        index = predict_symptom(s, list(symptoms_dict.keys()))
        print('User Input: ',s," Index: ",index)
        symptoms_dict[index] = 1
    
    # Put all data in a test dataset
    df_test = pd.DataFrame(columns=list(symptoms_dict.keys()))
    df_test.loc[0] = np.array(list(symptoms_dict.values()))
    print(df_test.head()) 
    # Load pre-trained model
    clf = load(str("model/random_forest.joblib"))
    result = clf.predict(df_test)

    disease_details = getDiseaseInfo(result[0])
    
    # Cleanup
    del df_test
    
    return f"<b>{result[0]}</b><br>{disease_details}",result[0]

# Get all unique diseases
diseases = set(df['Disease'])

def get_symtoms(user_disease):
    # Vectorize diseases using CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Disease'])
    user_X = vectorizer.transform([user_disease])

    # Compute cosine similarity between user disease and dataset diseases
    similarity_scores = cosine_similarity(X, user_X)

    # Find the most similar disease(s)
    max_score = similarity_scores.max()
    print(max_score)
    if max_score < 0.7:
        print("No matching diseases found")
        return False,"No matching diseases found"
    else:
        max_indices = similarity_scores.argmax(axis=0)
        symptoms_set = set()
        for i in max_indices:
            if similarity_scores[i] == max_score:
                symptoms_set.update(set(df.iloc[i]['Symptoms'].split(',')))
        # Output results

        print("The symptoms of", user_disease, "are:")
        for sym in symptoms_set:
            print(str(sym).capitalize())

        return True,symptoms_set

from duckduckgo_search import DDGS

def getDiseaseInfo(keywords):
    try:
        ddgs = DDGS()
        results = list(ddgs.text(keywords, region='wt-wt', safesearch='moderate', timelimit='y', max_results=1))
        if results:
            return results[0]['body']
        else:
            return "No information found."
    except Exception as e:
        return f"Error retrieving information: {str(e)}"

userSession = {}

@app.route('/ask',methods=['GET','POST'])
def chat_msg():
    user_message = request.args["message"].lower()
    sessionId = request.args["sessionId"]

    rand_num = random.randint(0,4)
    response = []
    if request.args["message"]=="undefined":
        response.append(msgCons.WELCOME_GREET[rand_num])
        response.append("What is your good name?")
        return jsonify({'status': 'OK', 'answer': response})
    else:
        currentState = userSession.get(sessionId)

        if currentState ==-1:
            response.append("Hi "+user_message+", To predict your disease based on symptopms, we need some information about you. Please provide accordingly.")
            userSession[sessionId] = userSession.get(sessionId) +1
            all_result['name'] = user_message            

        if currentState==0:
            username = all_result['name']
            response.append(username+", what is you age?")
            userSession[sessionId] = userSession.get(sessionId) +1

        if currentState==1:
            pattern = r'\d+'
            result = re.findall(pattern, user_message)
            if len(result)==0:
                response.append("Invalid input please provide valid age.")
            else:                
                if float(result[0])<=0 or float(result[0])>=130:
                    response.append("Invalid input please provide valid age.")
                else:
                    all_result['age'] = float(result[0])
                    username = all_result['name']
                    response.append(username+", Choose Option ?")            
                    response.append("1. Predict Disease")
                    response.append("2. Check Disease Symtoms")
                    userSession[sessionId] = userSession.get(sessionId) +1

        if currentState==2:
            if '2' in user_message.lower() or 'check' in user_message.lower():
                username = all_result['name']
                response.append(username+", What's Disease Name?")
                userSession[sessionId] = 20
            else:
                username = all_result['name']
                response.append(username+", What symptoms are you experiencing?")         
                response.append('<a href="/diseases" target="_blank">Symptoms List</a>')   
                userSession[sessionId] = userSession.get(sessionId) +1

        if currentState==3:
            all_result['symptoms'].extend(user_message.split(","))
            username = all_result['name']
            response.append(username+", What kind of symptoms are you currently experiencing?")            
            response.append("1. Check Disease")   
            response.append('<a href="/diseases" target="_blank">Symptoms List</a>')   
            userSession[sessionId] = userSession.get(sessionId) +1

        if currentState==4:
            if '1' in user_message or 'disease' in user_message:
                disease,type = predict_disease_from_symptom(all_result['symptoms'])  
                response.append("<b>The following disease may be causing your discomfort</b>")
                response.append(disease)
                response.append(f'<a href="https://www.google.com/search?q={type} disease hospital near me" target="_blank">Search Near By Hospitals</a>')   
                userSession[sessionId] = 10
            else:
                all_result['symptoms'].extend(user_message.split(","))
                username = all_result['name']
                response.append(username+", Could you describe the symptoms you're suffering from?")            
                response.append("1. Check Disease")   
                response.append('<a href="/diseases" target="_blank">Symptoms List</a>')   
                userSession[sessionId] = userSession.get(sessionId) +1
    
        if currentState==5:
            if '1' in user_message or 'disease' in user_message:
                disease,type = predict_disease_from_symptom(all_result['symptoms'])  
                response.append("<b>The following disease may be causing your discomfort</b>")
                response.append(disease)
                response.append(f'<a href="https://www.google.com/search?q={type} disease hospital near me" target="_blank">Search Near By Hospitals</a>')   
                userSession[sessionId] = 10
            else:
                all_result['symptoms'].extend(user_message.split(","))
                username = all_result['name']
                response.append(username+", What are the symptoms that you're currently dealing with?")            
                response.append("1. Check Disease")   
                response.append('<a href="/diseases" target="_blank">Symptoms List</a>')   
                userSession[sessionId] = userSession.get(sessionId) +1

        if currentState==6:    
            if '1' in user_message or 'disease' in user_message:
                disease,type = predict_disease_from_symptom(all_result['symptoms'])  
                response.append("The following disease may be causing your discomfort")
                response.append(disease)
                response.append(f'<a href="https://www.google.com/search?q={type} disease hospital near me" target="_blank">Search Near By Hospitals</a>')   
                userSession[sessionId] = 10
            else:
                all_result['symptoms'].extend(user_message.split(","))
                username = all_result['name']
                response.append(username+", What symptoms have you been experiencing lately?")            
                response.append("1. Check Disease")   
                response.append('<a href="/diseases" target="_blank">Symptoms List</a>')   
                userSession[sessionId] = userSession.get(sessionId) +1

        if currentState==7:
            if '1' in user_message or 'disease' in user_message:
                disease,type = predict_disease_from_symptom(all_result['symptoms'])  
                response.append("<b>The following disease may be causing your discomfort</b>")
                response.append(disease)
                response.append(f'<a href="https://www.google.com/search?q={type} disease hospital near me" target="_blank">Search Near By Hospitals</a>')   
                userSession[sessionId] = 10
            else:
                all_result['symptoms'].extend(user_message.split(","))
                username = all_result['name']
                response.append(username+", What are the symptoms that you're currently dealing with?")            
                response.append("1. Check Disease")   
                response.append('<a href="/diseases" target="_blank">Symptoms List</a>')   
                userSession[sessionId] = userSession.get(sessionId) +1

        if currentState==8:    
            if '1' in user_message or 'disease' in user_message:
                disease,type = predict_disease_from_symptom(all_result['symptoms'])  
                response.append("The following disease may be causing your discomfort")
                response.append(disease)
                response.append(f'<a href="https://www.google.com/search?q={type} disease hospital near me" target="_blank">Search Near By Hospitals</a>')   
                userSession[sessionId] = 10
            else:
                all_result['symptoms'].extend(user_message.split(","))
                username = all_result['name']
                response.append(username+", What symptoms have you been experiencing lately?")            
                response.append("1. Check Disease")   
                response.append('<a href="/diseases" target="_blank">Symptoms List</a>')   
                userSession[sessionId] = userSession.get(sessionId) +1

        if currentState==10:
            response.append('<a href="/user" target="_blank">Predict Again</a>')   

        if currentState==20:
            result,data = get_symtoms(user_message)
            if result:
                response.append(f"The symptoms of {user_message} are")
                for sym in data:
                    response.append(sym.capitalize())
            else:
                response.append(data)

            userSession[sessionId] = 2
            response.append("")
            response.append("Choose Option ?")            
            response.append("1. Predict Disease")
            response.append("2. Check Disease Symtoms")

        return jsonify({'status': 'OK', 'answer': response})

# Load the recommendation model
data_path = "recommend/data/input/appointments.csv"
model_filename = 'recommend/data/output/model.pkl'
specialist_dataset_filename = 'recommend/data/input/specialist.csv'
general_physician_dataset_filename = 'recommend/data/input/general.csv'

# Initialize the recommendation model with error handling
try:
    recommendation_model = RecommendationModel(data_path, model_filename, specialist_dataset_filename, general_physician_dataset_filename)
except Exception as e:
    print(f"Error loading recommendation model: {e}")
    recommendation_model = None

# Define SQLAlchemy models
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), nullable=False, unique=True)
    password = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    phone = db.Column(db.String(15), nullable=False)

class Appointment(db.Model):
    __tablename__ = 'appointments'
    id = db.Column(db.Integer, primary_key=True)
    token = db.Column(db.String(10), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    dob = db.Column(db.Date, nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    email = db.Column(db.String(255), nullable=False)
    specialist = db.Column(db.String(255), nullable=False)
    patient_condition = db.Column(db.String(255), nullable=False)
    medical_history = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, server_default=db.func.current_timestamp())

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            session['loggedin'] = True
            session['id'] = user.id
            session['username'] = user.username
            print("Session variables set successfully:", session)
            return redirect(url_for('dashboard'))
        else:
            flash('Incorrect username or password!', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form and 'name' in request.form and 'phone' in request.form:
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        name = request.form['name']
        phone = request.form['phone']
        user = User.query.filter_by(username=username).first()
        if user:
            flash('Account already exists!', 'danger')
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            flash('Invalid email address!', 'danger')
        elif not re.match(r'[A-Za-z0-9]+', username):
            flash('Username must contain only characters and numbers!', 'danger')
        elif not username or not password or not email or not name or not phone:
            flash('Please fill out the form!', 'danger')
        else:
            new_user = User(username=username, password=password, email=email, name=name, phone=phone)
            db.session.add(new_user)
            db.session.commit()
            flash('You have successfully registered!', 'success')
    return render_template('register.html')

@app.route('/booking')
def booking():
    return render_template('booking.html')

@app.route('/dashboard')
def dashboard():
    # Using SQLAlchemy instead of MySQL cursor
    appointments = Appointment.query.all()
    recent_appointments = Appointment.query.limit(3).all()
    total_appointments = Appointment.query.count()
    
    username = session.get('username')
    if username:
        individual_history = Appointment.query.filter_by(name=username).count()
    else:
        individual_history = 0
    
    return render_template('patient.html', 
                         data=appointments, 
                         data1=recent_appointments, 
                         row_count=total_appointments,
                         Individual_history=individual_history)

def generate_token():
    # Generate a random token (e.g., a 16-character alphanumeric string)
    return secrets.token_hex(8)

@app.route('/recommend_First', methods=['POST'])
def recommend_First():
    if request.method == 'POST':
        # Get form data
        patient_condition = request.form['patient_condition']
        medical_history = request.form['medical-history']

        recommended_doctor = None
        recommendation_source = "ML Model"

        # First, try the ML model if available
        specialist_type = None
        doctor_name = None
        if recommendation_model is not None:
            try:
                specialist_type, doctor_name = recommendation_model.recommend_doctor(patient_condition)
                print(f"ML Model recommended: Specialist={specialist_type}, Doctor={doctor_name}")
            except Exception as e:
                print(f"ML Model failed: {e}")
                specialist_type, doctor_name = None, None

        # If ML model failed or returned no result, use fallback
        if not specialist_type or not doctor_name or doctor_name == "Unknown":
            specialist_type = get_fallback_specialist(patient_condition)
            doctor_name = None
            recommendation_source = "Rule-based Fallback"
            flash(f'Using {recommendation_source} system for specialist recommendation.', 'info')
        else:
            recommendation_source = "ML Model"
            flash(f'Specialist recommended using {recommendation_source}.', 'success')

        print(f"Final recommendation: Specialist={specialist_type}, Doctor={doctor_name} (Source: {recommendation_source})")

        session['specialist_type'] = specialist_type
        session['recommended_doctor'] = doctor_name
        session['patient_condition'] = patient_condition
        session['medical_history'] = medical_history

        return render_template('recommendation_confirmation.html', 
                             specialist_type=specialist_type,
                             recommended_doctor=doctor_name, 
                             form_data=request.form,
                             recommendation_source=recommendation_source)

def get_fallback_specialist(patient_condition):
    """
    Enhanced fallback function to recommend specialist based on medical keywords
    This complements your trained ML model for cases it doesn't cover
    """
    condition_lower = patient_condition.lower()
    
    # Enhanced keyword-to-specialist mapping for medical conditions
    specialist_mapping = {
        # Cardiovascular
        'heart': 'Cardiologist',
        'cardiac': 'Cardiologist', 
        'chest pain': 'Cardiologist',
        'hypertension': 'Cardiologist',
        'blood pressure': 'Cardiologist',
        'palpitation': 'Cardiologist',
        'arrhythmia': 'Cardiologist',
        
        # Endocrine
        'diabetes': 'Endocrinologist',
        'diabetic': 'Endocrinologist',
        'sugar': 'Endocrinologist',
        'thyroid': 'Endocrinologist',
        'hormone': 'Endocrinologist',
        'insulin': 'Endocrinologist',
        
        # Dermatology
        'skin': 'Dermatologist',
        'rash': 'Dermatologist',
        'acne': 'Dermatologist',
        'eczema': 'Dermatologist',
        'psoriasis': 'Dermatologist',
        'dermatitis': 'Dermatologist',
        
        # Orthopedics
        'bone': 'Orthopedist',
        'joint': 'Orthopedist',
        'fracture': 'Orthopedist',
        'back pain': 'Orthopedist',
        'arthritis': 'Orthopedist',
        'knee pain': 'Orthopedist',
        'shoulder': 'Orthopedist',
        
        # Ophthalmology
        'eye': 'Ophthalmologist',
        'vision': 'Ophthalmologist',
        'sight': 'Ophthalmologist',
        'cataract': 'Ophthalmologist',
        'glaucoma': 'Ophthalmologist',
        
        # ENT
        'ear': 'ENT Specialist',
        'nose': 'ENT Specialist',
        'throat': 'ENT Specialist',
        'hearing': 'ENT Specialist',
        'sinus': 'ENT Specialist',
        'tonsil': 'ENT Specialist',
        
        # Gynecology
        'pregnancy': 'Gynecologist',
        'pregnant': 'Gynecologist',
        'menstrual': 'Gynecologist',
        'uterus': 'Gynecologist',
        'ovarian': 'Gynecologist',
        'pcos': 'Gynecologist',
        
        # Nephrology/Urology
        'kidney': 'Nephrologist',
        'renal': 'Nephrologist',
        'urine': 'Urologist',
        'bladder': 'Urologist',
        'prostate': 'Urologist',
        
        # Gastroenterology
        'stomach': 'Gastroenterologist',
        'digestive': 'Gastroenterologist',
        'liver': 'Gastroenterologist',
        'intestine': 'Gastroenterologist',
        'gastric': 'Gastroenterologist',
        'abdominal': 'Gastroenterologist',
        
        # Mental Health
        'mental': 'Psychiatrist',
        'depression': 'Psychiatrist',
        'anxiety': 'Psychiatrist',
        'stress': 'Psychiatrist',
        'bipolar': 'Psychiatrist',
        
        # Pulmonology
        'lung': 'Pulmonologist',
        'breathing': 'Pulmonologist',
        'asthma': 'Pulmonologist',
        'copd': 'Pulmonologist',
        'respiratory': 'Pulmonologist',
        
        # Oncology
        'cancer': 'Oncologist',
        'tumor': 'Oncologist',
        'malignant': 'Oncologist',
        'chemotherapy': 'Oncologist',
        
        # Pediatrics
        'child': 'Pediatrician',
        'infant': 'Pediatrician',
        'baby': 'Pediatrician',
        'pediatric': 'Pediatrician',
        
        # Neurology
        'brain': 'Neurologist',
        'headache': 'Neurologist',
        'migraine': 'Neurologist',
        'seizure': 'Neurologist',
        'epilepsy': 'Neurologist',
        'stroke': 'Neurologist',
        
        # General symptoms that might need specific specialists
        'fever': 'General Physician',
        'cold': 'General Physician',
        'flu': 'General Physician',
        'fatigue': 'General Physician'
    }
    
    # Check for keywords in the condition (prioritize more specific matches)
    matched_specialists = []
    for keyword, specialist in specialist_mapping.items():
        if keyword in condition_lower:
            matched_specialists.append((len(keyword), specialist))  # Longer keywords get priority
    
    if matched_specialists:
        # Return the specialist with the longest matching keyword (most specific)
        matched_specialists.sort(reverse=True)
        return matched_specialists[0][1]
    
    # Default to General Physician if no specific condition is found
    return 'General Physician'

@app.route('/confirm_booking_appointment')
def confirm_booking_appointment():
    return render_template('booking.html')

@app.route('/book_appointment', methods=['POST'])
def book_appointment():
    if request.method == 'POST':
        # Get form data
        form_data = request.form.to_dict(flat=False)
        specialist=session.get('recommended_doctor')
        patient_condition=session.get('patient_condition')
        medical_history=session.get('medical_history')

        name = request.form['name']
        age = request.form['age']
        dob_str = request.form['dob']  # Get the date of birth as a string
        phone = request.form['phone']
        email = request.form['email']

        if not name or not age or not dob_str or not phone or not email:
            flash('All fields are required', 'danger')
            return redirect(url_for('booking'))

        # Parse and convert the date string to the "YYYY-MM-DD" format
        formats = ["%d/%m/%Y", "%d-%m-%Y"]

        dob = None
        for fmt in formats:
            try:
                dob = datetime.datetime.strptime(dob_str, fmt).strftime("%Y-%m-%d")
                break
            except ValueError:
                continue

        if dob is None:
            flash('Invalid date format', 'danger')
            return redirect(url_for('booking'))

        # Generate the token with the format "HC0000" using SQLAlchemy
        last_appointment = Appointment.query.order_by(Appointment.id.desc()).first()
        if last_appointment and last_appointment.token:
            last_token_number = int(last_appointment.token[2:])
            new_token_number = last_token_number + 1
            token = f'HC{new_token_number:04d}'
        else:
            token = 'HC0001'

        print("Specialist value:", specialist)
        if not specialist:
            flash('Specialist must be selected.', 'danger')
            return redirect(url_for('booking'))

        # Create and add new appointment
        new_appointment = Appointment(
            token=token,
            name=name,
            age=age,
            dob=dob,
            phone=phone,
            email=email,
            specialist=specialist,
            patient_condition=patient_condition,
            medical_history=medical_history
        )
        db.session.add(new_appointment)
        db.session.commit()

        flash(f'Appointment booked successfully! Your appointment token is: {token}', 'success')

        # Email notification
        try:
            email_sender = 'appointmentshealthcare@gmail.com'
            email_password = 'icfebluyjdvisofd'
            email_receiver = email

            subject = "HealthCare Appointments Booking"
            body = f"""
            Appointment booked successfully! Your appointment token is: {token}
            Your Recommended doctor is {specialist}
            """

            em = EmailMessage()
            em['From'] = email_sender
            em['To'] = email_receiver
            em['subject'] = subject
            em.set_content(body)

            context = ssl.create_default_context()

            with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
                smtp.login(email_sender, email_password)
                smtp.sendmail(email_sender, email_receiver, em.as_string())
        except Exception as e:
            print(f"Email sending failed: {e}")

        # Pass the recommended specialist to the booking form
        return render_template('recommend.html', recommended_doctor=specialist, form_data=request.form)

    return render_template('booking.html')

@app.route('/recommendfirst')
def recommendFirst():
    return render_template('recommendfirst.html')

@app.route('/display_tokens')
def display_tokens():
    # Using SQLAlchemy instead of MySQL cursor
    appointments = Appointment.query.with_entities(Appointment.token).all()
    token_list = [appointment.token for appointment in appointments]
    return render_template('token.html', token_list=token_list)

@app.route('/recommend_appointment')
def recommend_appointment_route():
    if recommendation_model is None:
        return jsonify({'error': 'Recommendation service unavailable'}), 503
    
    try:
        appointment_index = 5  # Replace with the index of the appointment you want recommendations for
        num_recommendations = 5  # Adjust the number of recommendations as needed
        recommendations = recommendation_model.get_recommendations(appointment_index, num_recommendations)
        # You can pass the recommendations to a template or return them as JSON
        return render_template('recommend.html', recommendations=recommendations)
    except Exception as e:
        print(f"Error in recommendation: {e}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

# Define a function to get appointment details based on the index
def get_appointment_details(appointment_index):
    appointment = Appointment.query.get(appointment_index)
    if appointment:
        return {
            'id': appointment.id,
            'token': appointment.token,
            'name': appointment.name,
            'age': appointment.age,
            'phone': appointment.phone,
            'patient_condition': appointment.patient_condition,
            'specialist': appointment.specialist,
            'medical_history': appointment.medical_history,
            'timestamp': appointment.timestamp
        }
    return None

# Define a context processor to make get_appointment_details available globally
@app.context_processor
def utility_processor():
    def get_appointment_details_wrapper(appointment_index):
        return get_appointment_details(appointment_index)
    return dict(get_appointment_details=get_appointment_details_wrapper)

@app.route('/recommendations/<int:appointment_index>')
def show_recommendations(appointment_index):
    if recommendation_model is None:
        return jsonify({'error': 'Recommendation service unavailable'}), 503
    
    try:
        num_recommendations = 5  # You can adjust this to your preferred number of recommendations

        # Call the recommendation model to get appointment recommendations
        recommendations = recommendation_model.get_recommendations(appointment_index, num_recommendations)

        # Create a list to hold appointment details for the recommendations
        recommendation_details = [get_appointment_details(index) for index in recommendations]

        # Pass the recommendations and their details to the template for rendering
        return render_template('recommendations.html', recommendations=recommendation_details)
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return jsonify({'error': 'Failed to get recommendations'}), 500

# Route for recommendations page
@app.route('/recommendations_page')
def recommendations_page():
    # You need to define 'recommendations' before this function or pass it as needed
    recommendations = []  # Replace with actual recommendations logic
    recommendations_with_index = [(i+1, recommendation) for i, recommendation in enumerate(recommendations)]
    return render_template('recommendations.html', css_style=url_for('static', filename='style.css'), recommendations=recommendations_with_index)

# Medical record route
@app.route('/medical_record')
def medical_record():
    username = session.get('username')
    if not username:
        flash('Please log in to view your medical records.', 'danger')
        return redirect(url_for('login'))
    records = Appointment.query.filter_by(name=username).all()
    return render_template('medical_record.html', records=records)

@app.route('/medicine_request', methods=['GET', 'POST'])
def requestmedicinetemplate():
    recommended = []
    selected_medicine = None
    if request.method == 'POST':
        medicine_name = request.form.get('medicine')
        selected_medicine = medicine_name
        if medicine_name:
            try:
                recommended = recommend(medicine_name)
            except Exception as e:
                flash(f'Error recommending medicine: {e}', 'danger')
    return render_template('medication_request.html', css_style=url_for('static', filename='style.css'), medicines=medicines['Drug_Name'].values, recommended=recommended, selected_medicine=selected_medicine)

# Load medicine data
try:
    medicines_dict = pickle.load(open('medicine_dict.pkl','rb'))
    medicines = pd.DataFrame(medicines_dict)
    similarity = pickle.load(open('similarity.pkl','rb'))
except FileNotFoundError:
    print("Medicine pickle files not found. Medicine recommendation feature will not work.")
    medicines = pd.DataFrame()
    similarity = []

def recommend(medicine):
    try:
        index = medicines[medicines['Drug_Name'] == medicine].index[0]
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        recommended_medicine = []
        for i in distances[1:6]:
            recommended_medicine.append(medicines.iloc[i[0]].Drug_Name)
        return recommended_medicine
    except (IndexError, KeyError):
        return []

# Routes for medicine recommendation
@app.route('/request_medicine', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_medicine_name = request.form['medicine_name']
        recommendations = recommend(selected_medicine_name)
        recommendations_with_index = [(i+1, recommendation) for i, recommendation in enumerate(recommendations)]
        return render_template('recommendations.html',css_style=url_for('static', filename='style.css'), recommendations=recommendations_with_index)
    
    if not medicines.empty:
        medicine_names = medicines['Drug_Name'].values
    else:
        medicine_names = []
    
    return render_template('index.html', css_style=url_for('static', filename='style.css'), medicines=medicine_names)

@app.route("/chatbot")
def index_auth():
    my_id = make_token()
    userSession[my_id] = -1
    return render_template("index_auth.html",sessionId=my_id)

@app.route("/upload")
def bmi():
    return render_template("bmi.html")

@app.route("/diseases")
def diseases():
    return render_template("diseases.html")

@app.route("/user")
def user():
    return render_template("user.html")

# Error handlers (uncomment if you create the template files)
# @app.errorhandler(404)
# def not_found_error(error):
#     return render_template('404.html'), 404

# @app.errorhandler(500)
# def internal_error(error):
#     db.session.rollback()
#     return render_template('500.html'), 500

# Simple JSON error handlers (current implementation)
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Page not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
            print("Database tables created successfully!")
        except Exception as e:
            print(f"Error creating database tables: {e}")
    app.run(debug=True)