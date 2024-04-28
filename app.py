from flask import Flask, render_template, request, redirect, url_for, session, flash, request, jsonify
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import secrets
from config import DATABASE_CONFIG
from secret import SECRET_KEY
import mysql.connector
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
from flask import jsonify
import secrets
from flask import Flask, render_template, flash, redirect, url_for, session, logging, request, session
from flask_sqlalchemy import SQLAlchemy
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





# Load the trained model
model_filename = 'recommend/data/output/medi_model.pkl'
model = joblib.load(model_filename)

app = Flask(__name__)
app.secret_key = SECRET_KEY

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


# Import Dependencies
# import gradio as gr


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

        

    symptoms = {'itching': 0, 'skin_rash': 0, 'nodal_skin_eruptions': 0, 'continuous_sneezing': 0,
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
        index = predict_symptom(s, list(symptoms.keys()))
        print('User Input: ',s," Index: ",index)
        symptoms[index] = 1
    
    # Put all data in a test dataset
    df_test = pd.DataFrame(columns=list(symptoms.keys()))
    df_test.loc[0] = np.array(list(symptoms.values()))
    print(df_test.head()) 
    # Load pre-trained model
    clf = load(str("model/random_forest.joblib"))
    result = clf.predict(df_test)

    disease_details = getDiseaseInfo(result[0])
    
    # Cleanup
    del df_test
    
    return f"<b>{result[0]}</b><br>{disease_details}",result[0]



import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
        symptoms = set()
        for i in max_indices:
            if similarity_scores[i] == max_score:
                symptoms.update(set(df.iloc[i]['Symptoms'].split(',')))
        # Output results

        print("The symptoms of", user_disease, "are:")
        for sym in symptoms:
            print(str(sym).capitalize())

        return True,symptoms


# from duckduckgo_search import ddg

# def getDiseaseInfo(keywords):
#     results = ddg(keywords, region='wt-wt', safesearch='Off', time='y')
#     return results[0]['body']
    
from duckduckgo_search import DDGS

def getDiseaseInfo(keywords):
    results = DDGS(keywords, region='wt-wt', safesearch='Off', time='y')
    return results[0]['body']

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

            else:response.append(data)

            userSession[sessionId] = 2
            response.append("")
            response.append("Choose Option ?")            
            response.append("1. Predict Disease")
            response.append("2. Check Disease Symtoms")




                

        return jsonify({'status': 'OK', 'answer': response})



#________________________



# Use the database configuration from the config file
db_connection = mysql.connector.connect(**DATABASE_CONFIG)

# Check and create the required database and tables if they don't exist
with db_connection.cursor() as cursor:
    # Create the database
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DATABASE_CONFIG['database']}")
    cursor.execute(f"USE {DATABASE_CONFIG['database']}")
    

    # Create the users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255) NOT NULL,
            password VARCHAR(255) NOT NULL,
            email VARCHAR(255) NOT NULL,
            name VARCHAR(255) NOT NULL,
            phone VARCHAR(15) NOT NULL
        )
    """)

    # Create the appointments table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            id INT AUTO_INCREMENT PRIMARY KEY,
            token VARCHAR(10) NOT NULL,
            name VARCHAR(255) NOT NULL,
            age INT NOT NULL,
            dob DATE NOT NULL,
            phone VARCHAR(15) NOT NULL,
            email VARCHAR(255) NOT NULL,
            specialist VARCHAR(255) NOT NULL,
            patient_condition VARCHAR(255) NOT NULL,
            medical_history TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

# Commit the changes
db_connection.commit()


# Load the recommendation model
data_path = "recommend/data/input/appointments.csv"
model_filename = 'recommend/data/output/model.pkl'
specialist_dataset_filename = 'recommend/data/input/specialist.csv'
general_physician_dataset_filename = 'recommend/data/input/general.csv'
recommendation_model = RecommendationModel(data_path, model_filename, specialist_dataset_filename, general_physician_dataset_filename)
        
app.config['MYSQL_HOST'] = DATABASE_CONFIG['host']
app.config['MYSQL_USER'] = DATABASE_CONFIG['user']
app.config['MYSQL_PASSWORD'] = DATABASE_CONFIG['password']
app.config['MYSQL_DB'] = DATABASE_CONFIG['database']

mysql = MySQL(app)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        account = cursor.fetchone()
        if account and account['password'] == password:
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            print("Session variables set successfully:", session)
            return redirect(url_for('dashboard'))
        else:
            flash('danger', 'Incorrect username or password!')

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
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        account = cursor.fetchone()
        if account:
            flash('danger', 'Account already exists!')
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            flash('danger', 'Invalid email address!')
        elif not re.match(r'[A-Za-z0-9]+', username):
            flash('danger', 'Username must contain only characters and numbers!')
        elif not username or not password or not email or not name or not phone:
            flash('danger', 'Please fill out the form!')
        else:
            cursor.execute('INSERT INTO users (username, password, email, name, phone) VALUES (%s, %s, %s, %s, %s)',
                           (username, password, email, name, phone))
            mysql.connection.commit()
            flash('You have successfully registered!', 'You have successfully registered!')

    return render_template('register.html')

@app.route('/booking')
def booking():
    return render_template('booking.html')

@app.route('/dashboard')
def dashboard():
    cur = mysql.connection.cursor()
    cur.execute("SELECT name, age, phone, patient_condition FROM appointments")
    data = cur.fetchall()
    cur.execute("SELECT name, age, phone, patient_condition FROM appointments  LIMIT 3")
    data1=cur.fetchall()
    cur.execute("SELECT COUNT(*) FROM appointments")
    row_count = cur.fetchone()[0]
    username = session['username']
    cur.execute("SELECT COUNT(*) FROM appointments WHERE name = %s", (username,))
    Individual_history = cur.fetchone()[0]
    cur.close()
    return render_template('patient.html', data=data,data1=data1, row_count=row_count,Individual_history=Individual_history)
def generate_token():
    # Generate a random token (e.g., a 16-character alphanumeric string)
    return secrets.token_hex(8)

@app.route('/recommend_First', methods=['POST'])
def recommend_First():
      if request.method == 'POST':
        # Get form data
        patient_condition = request.form['patient_condition']
        medical_history = request.form['medical-history']

        # Get the recommended specialist from the AI recommendation model
        recommended_doctor = recommendation_model.recommend_doctor(patient_condition)

        session['recommended_doctor']=recommended_doctor
        session['patient_condition']=patient_condition
        session['medical_history']=medical_history

        return render_template('recommendation_confirmation.html', recommended_doctor=recommended_doctor, form_data=request.form)
      
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
            flash('danger', 'All fields are required')
            return redirect(url_for('booking'))

        # Parse and convert the date string to the "YYYY-MM-DD" format
        formats = ["%d/%m/%Y", "%d-%m-%Y"]

        dob = None
        for format in formats:
            try:
                dob = datetime.datetime.strptime(dob_str, format).strftime("%Y-%m-%d")
                break
            except ValueError:
                continue

        if dob is None:
            flash('danger', 'Invalid date format')
            return redirect(url_for('booking'))
        
        

        # Generate the token with the format "HC0000"
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT MAX(token) AS max_token FROM appointments')
        max_token = cursor.fetchone()
        if max_token and max_token['max_token']:
            last_token_number = int(max_token['max_token'][2:])  # Extract the numeric part
            new_token_number = last_token_number + 1
            token = f'HC{new_token_number:04d}'  # Format the new token number
        else:
            # If there are no existing tokens, start from "HC0001"
            token = 'HC0001'

        # Insert data into the database, including the generated token and specialist information
        cursor.execute('INSERT INTO appointments (token, name, age, dob, phone, email, specialist, patient_condition, medical_history) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)',
                       (token, name, age, dob, phone, email, specialist, patient_condition, medical_history))
        mysql.connection.commit()
        cursor.close()

        # Fetch the details of the newly booked appointment
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM appointments WHERE token = %s', (token,))
        new_appointment = cursor.fetchone()

        flash('success', f'Appointment booked successfully! Your appointment token is: {token}')


        #-------------------------------------------------------------------------------------------
        
        email_sender = 'appointmentshealthcare@gmail.com'

        email_password = 'icfebluyjdvisofd'

        email_receiver = email

        subject = "HealthCare Appointments Booking"

        body = """
        'Appointment booked successfully! Your appointment token is: {}  Your Recommended doctor is {}'
        """.format(token, specialist)


        em = EmailMessage()
        em['From'] = email_sender
        em['To'] = email_receiver
        em['subject'] = subject
        em.set_content(body)

        context = ssl.create_default_context()

        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:

            smtp.login(email_sender, email_password)
            smtp.sendmail(email_sender, email_receiver, em.as_string())



        # Pass the recommended specialist to the booking form
        return render_template('recommend.html', recommended_doctor=specialist, form_data=request.form)
        

    return render_template('booking.html')

@app.route('/recommendfirst')
def recommendFirst():
    return render_template('recommendfirst.html')


@app.route('/display_tokens')
def display_tokens():
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT token FROM appointments')
    tokens = cursor.fetchall()
    cursor.close()

    token_list = [token['token'] for token in tokens]

    return render_template('token.html', token_list=token_list)

@app.route('/recommend_appointment')
def recommend_appointment_route():
    appointment_index = 5  # Replace with the index of the appointment you want recommendations for
    num_recommendations = 5  # Adjust the number of recommendations as needed
    recommendations = recommendation_model.get_recommendations(appointment_index, num_recommendations)

    # You can pass the recommendations to a template or return them as JSON
    return render_template('recommend.html', recommendations=recommendations)

# Define a function to get appointment details based on the index
def get_appointment_details(appointment_index):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * FROM appointments WHERE id = %s', (appointment_index,))
    appointment_details = cursor.fetchone()
    cursor.close()
    return appointment_details

# Define a context processor to make get_appointment_details available globally
@app.context_processor
def utility_processor():
    def get_appointment_details_wrapper(appointment_index):
        return get_appointment_details(appointment_index)

    return dict(get_appointment_details=get_appointment_details_wrapper)

# Modify the 'recommendations.html' route
@app.route('/recommendations/<int:appointment_index>')
def show_recommendations(appointment_index):
    num_recommendations = 5  # You can adjust this to your preferred number of recommendations

    # Call the recommendation model to get appointment recommendations
    recommendations = recommendation_model.get_recommendations(appointment_index, num_recommendations)

    # Create a list to hold appointment details for the recommendations
    recommendation_details = [get_appointment_details(index) for index in recommendations]

    # Pass the recommendations and their details to the template for rendering
    return render_template('recommends.html', recommendations=recommendation_details)

@app.route('/medical_record')
def display_medical_record():
    username = session['username']
    print(username)
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM appointments WHERE name = %s", (username,))
    data = cur.fetchall()
    cur.close()
        # Process the data or pass it to a template
    return render_template('medical_record.html',data=data)

@app.route('/suggest_specialist', methods=['POST'])
def suggest_specialist():
    # Get the patient condition from the AJAX request
    patient_condition = request.json['patient_condition']

    # Use your AI algorithm to determine the suitable specialist
    suggested_specialist = RecommendationModel(patient_condition)

    # Return the suggested specialist
    return jsonify(suggested_specialist)

# Recommendation function
def recommend(medicine):
    medicine_index = medicines[medicines['Drug_Name'] == medicine].index[0]
    distances = similarity[medicine_index]
    medicines_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_medicines = []
    for i in medicines_list:
        recommended_medicines.append(medicines.iloc[i[0]].Drug_Name)
    return recommended_medicines

@app.route('/recommend_medicine', methods=['POST'])
def requestmedicine():
    if request.method == 'POST':
        selected_medicine_name = request.form['medicine_name']
        recommendations = recommend(selected_medicine_name)
        recommendations_with_index = [(i+1, recommendation) for i, recommendation in enumerate(recommendations)]
        return render_template('recommendations.html',css_style=url_for('static', filename='style.css'), recommendations=recommendations_with_index)

        # # Print predictions
        # print("Predicted medicine:", predictions)
    return render_template('medication_request.html')

@app.route('/medicine_request')
def requestmedicinetemplate():
    return render_template('medication_request.html', css_style=url_for('static', filename='style.css'), medicines=medicines['Drug_Name'].values)


# Load data
medicines_dict = pickle.load(open('medicine_dict.pkl','rb'))
medicines = pd.DataFrame(medicines_dict)
similarity = pickle.load(open('similarity.pkl','rb'))



# Routes
@app.route('/request_medicine', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_medicine_name = request.form['medicine_name']
        recommendations = recommend(selected_medicine_name)
        recommendations_with_index = [(i+1, recommendation) for i, recommendation in enumerate(recommendations)]
        return render_template('recommendations.html',css_style=url_for('static', filename='style.css'), recommendations=recommendations_with_index)
    return render_template('index.html', css_style=url_for('static', filename='style.css'), medicines=medicines['Drug_Name'].values)


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



if __name__ == '__main__':
    app.run(debug=True)







#------------------------------------------------------------

