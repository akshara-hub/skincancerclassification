import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from tensorflow.keras import backend as K

# Define custom metrics (needed for loading the model)
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# Load the trained model with custom metrics
model = tf.keras.models.load_model("model.h5", custom_objects={
    "precision_m": precision_m,
    "recall_m": recall_m,
    "f1_m": f1_m
})

# ✅ Ensure class_labels matches your model's number of output classes
class_labels = ["Actinic keratoses", "Basal cell carcinoma", "Benign keratosis", 
                "Dermatofibroma", "Melanoma", "Nevus", "Vascular lesion"]  # ✅ FIXED: Updated class list

# Function to preprocess and predict
def predict_skin_cancer(img_path):
    img = image.load_img(img_path, target_size=(128, 128))  # ✅ Ensure input size matches model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    predictions = model.predict(img_array)
    confidence = np.max(predictions)  # Get confidence score
    class_index = np.argmax(predictions)  # Predicted class index

    # ✅ Prevent IndexError
    if class_index >= len(class_labels):
        predicted_label = "Unknown"  # Handle unexpected index
    else:
        predicted_label = class_labels[class_index]

    return predicted_label, confidence

# Function to generate PDF report
def generate_pdf(report_path, img_name, prediction, confidence):
    c = canvas.Canvas(report_path, pagesize=letter)

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Skin Cancer AI Diagnosis Report")

    # Image Name
    c.setFont("Helvetica", 12)
    c.drawString(100, 720, f"Image: {img_name}")

    # Model Prediction
    c.drawString(100, 700, f"Prediction: {prediction}")
    c.drawString(100, 680, f"Confidence Score: {confidence:.2f}")

    # Suggested Next Steps
    c.drawString(100, 650, "Suggested Next Steps:")
    if confidence < 0.7:
        c.drawString(120, 630, "-> Confidence is low. Please consult a dermatologist.")
    else:
        c.drawString(120, 630, "-> AI is confident, but a doctor’s confirmation is recommended.")

    # Save PDF
    c.save()

# Process all images in the "test_images" folder
input_folder = "test_images"
output_folder = "reports"

# Create reports folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all images in the folder
for img_name in os.listdir(input_folder):
    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):  # Check for valid image files
        img_path = os.path.join(input_folder, img_name)

        # Get prediction
        predicted_label, confidence = predict_skin_cancer(img_path)

        # Generate report
        report_path = os.path.join(output_folder, f"{img_name.split('.')[0]}_report.pdf")
        generate_pdf(report_path, img_name, predicted_label, confidence)

        print(f"✅ Report generated: {report_path}")

print("\nAll reports are generated successfully! Check the 'reports' folder.")
