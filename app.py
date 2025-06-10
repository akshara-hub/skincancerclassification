from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import sqlite3
import pandas as pd
import numpy as np
import pickle
import sqlite3
import random

import smtplib 
from email.message import EmailMessage
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import textwrap

def generate_pdf(report_path, image_path, prediction):
    c = canvas.Canvas(report_path, pagesize=letter)
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Skin Cancer Diagnosis Report")

    # Image Section
    c.setFont("Helvetica", 14)
    c.drawString(100, 720, "Uploaded Image:")
    c.drawImage(image_path, 100, 500, width=200, height=200)      # Place image in PDF

    # Wrap Prediction Text
    c.setFont("Helvetica", 12)
    wrapped_text = textwrap.wrap(f"Prediction: {prediction}", width=80) #Adjust width as needed
    y_position=480        #Start position

    for line in wrapped_text:
        c.drawString(100, y_position, line)
        y_position -= 20         # Move to the next line

    # Extra spacing before "Suggested Next Steps"
    y_position -= 10
    c.drawString(100,y_position,"")

    # Suggested Next Steps
    y_position -= 20
    c.setFont("Helvetica",14)
    suggested_steps = {
        "Actinic Keratoses": [
            "-> Avoid sun exposure and use sunscreen regularly.",
            "-> Consult a dermatologist for early treatment."
        ],
        "Basal cell carcinoma": [
            "-> Seek medical advice immediately.",
            "-> Treatment options include surgery, radiation, or topical medications."
        ],
        "Benign Keratosis-like Lesions": [
            "-> Regularly check for any changes in size or color.",
            "-> Consult a dermatologist if you notice rapid growth."
        ],
        "Dermatofibroma": [
            "-> Usually harmless, but monitor for changes.",
            "-> Consult a doctor if it becomes painful or itchy."
        ],
        "Melanoma": [
            "-> Seek immediate medical attention as it can be aggressive.",
            "-> Treatment may include surgery, chemotherapy, or immunotherapy."
        ],
        "Melanocytic nevus": [
            "-> Monitor for any irregularities in color or size.",
            "-> Consult a dermatologist if changes occur."
        ],
        "Vascular lesions": [
            "-> Typically harmless, but seek medical advice for unusual growths.",
            "-> Laser treatment may be an option for cosmetic removal."
        ]
    }

    # Find the corresponding next steps based on prediction
    for key in suggested_steps:
        if key.lower() in prediction.lower():
            steps = suggested_steps[key]
            break
        else:
            steps = ["-> Monitor regularly.", "-> Consult a dermatologist if needed."]

    c.drawString(100, y_position, "Suggested Next Steps:")
    y_position -= 30
    c.drawString(100, y_position, steps[0])
    y_position -= 35
    c.drawString(100, y_position, steps[1])

    c.save() # Save the PDF file correctly

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

# allow files of a specific type
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# function to check the file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model_path2 = 'model.h5' # load .h5 Model

from keras import backend as K # type: ignore

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



#classes2 = {0:"APHIDS",1:"ARMYWORM",2:"BEETLE",3:"BOLLWORM",4:"GRASSHOPPER",5:"MITES",6:"MOSQUITO",7:"SAWFLY",8:"STEM BORER"}
CTS = load_model(model_path2, custom_objects={'f1_score' : f1_m, 'precision_score' : precision_m, 'recall_score' : recall_m}, compile=False)

from tensorflow.keras.utils import load_img, img_to_array # type: ignore

def model_predict2(image_path,model):
    print("Predicted")
    image = load_img(image_path,target_size=(128,128))
    image = img_to_array(image)
    image = image/255
    image = np.expand_dims(image,axis=0)
    
    result = np.argmax(model.predict(image))
    print(result)
    #prediction = classes2[result]  
    
    if result == 0:
        return "Actinic Keratoses (Solar Keratoses) and Intraepithelial Carcinoma (Bowen's disease) which are referred to in this paper as AKIEC","result.html"        
    elif result == 1:
        return "Basal cell carcinoma (BCC) is a type of skin cancer that forms in the basal cells of your skin.","result.html"
    elif result == 2:
        return "Benign Keratosis-like Lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl)","result.html"
    elif result == 3:
        return "A dermatofibroma is a common overgrowth of the fibrous tissue situated in the dermis (the deeper of the two main layers of the skin)","result.html"

    elif result == 4:
        return "Melanoma is a kind of skin cancer that starts in the melanocytes. ","result.html"
    elif result == 5:
        return "Melanocytic nevus is non-cancerous disorder of pigment-producing skin cells commonly called birthmarks or moles.","result.html"
    elif result == 6:
        return "vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc).","result.html"
    

@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("Entered")
    
    file = request.files['file']  # Get uploaded file
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    print("@@ Predicting class......")
    pred, output_page = model_predict2(file_path, CTS)

    # ✅ Generate PDF Report
    report_filename = f"{filename.split('.')[0]}_report.pdf"
    report_path = os.path.join("static/reports", report_filename)
    os.makedirs("static/reports", exist_ok=True)  # Ensure reports folder exists
    generate_pdf(report_path, file_path, pred)

    print("DEBUG: PDF should be saved at:", report_path)
    
    return render_template(output_page, pred_output=pred, img_src=file_path, pdf_link=f"/static/reports/{report_filename}")

@app.route('/')
@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')

@app.route('/index')
def index():
	return render_template('index.html')

@app.route("/signup")
def signup():
    global otp, username, name, email, number, password
    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    otp = random.randint(1000,5000)
    print(otp)
    msg = EmailMessage()
    msg.set_content("Your OTP is : "+str(otp))
    msg['Subject'] = 'OTP'
    msg['From'] = "evotingotp4@gmail.com"
    msg['To'] = email
      
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("evotingotp4@gmail.com", "xowpojqyiygprhgr")
    s.send_message(msg)
    s.quit()
    return render_template("val.html")

@app.route('/predict_lo', methods=['POST'])
def predict_lo():
    global otp, username, name, email, number, password
    if request.method == 'POST':
        message = request.form['message']
        print(message)
        if int(message) == otp:
            print("TRUE")
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
            con.commit()
            con.close()
            return render_template("signin.html")
    return render_template("signup.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signin.html")

@app.route("/notebook")
def notebook():
    return render_template("Notebook.html")

from flask import send_from_directory # type: ignore

@app.route('/static/reports/<filename>')
def download_report(filename):
    return send_from_directory("static/reports", filename)

@app.route('/result')
def result_page():
    return render_template("result.html")
   
if __name__ == '__main__':
    app.run(debug=False)