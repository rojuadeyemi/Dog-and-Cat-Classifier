from flask import Flask, request, render_template, redirect
import numpy as np
import os
from tensorflow.keras.models import load_model
from utility.utility_functions import preprocess_image

# define flask app
app = Flask(__name__)

# Load the trained model
model = load_model('model/CNN.keras')

# Configure upload folder
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')

# Create upload folder, if it does not exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Home endpoint
@app.route('/')
def main():
    return render_template('index.html')

# predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save the uploaded image
        image_path = os.path.join('uploads', file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        
        # Preprocess Image for prediction
        img_array = preprocess_image(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        
        # Predict
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        confidence_score = float(np.max(predictions[0]))  # Using max probability for confidence score
        
        # Map prediction to label
        text = {
            "status": f"The picture uploaded contains an image of a Cat" 
                        if predicted_class[0] == 0 else f"The picture uploaded contains an image of a Dog",
            "remark": f"{{with probability of {confidence_score:.0%}}}"
        }

        # Use relative path to 'uploads' for URL mapping
        return render_template('after.html', text=text, image_path=f'uploads/{file.filename}')
    
if __name__ == '__main__':
    app.run(debug=True)