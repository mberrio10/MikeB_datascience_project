import joblib
from flask_cors import CORS, cross_origin
from flask import Flask, render_template, request, jsonify

# Load trained model and vectorizer
model = joblib.load('/models/baseline_spam_detector.pkl')  # Make sure this file exists
vectorizer = joblib.load('/models/tfidf_vectorizer.pkl')  # Load saved vectorizer used for training

app = Flask(__name__)
CORS(app, supports_credentials=True)# Enable CORS globally

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
@cross_origin() # Enable CORS for this specific route

def predict():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Invalid input format"}), 400

    url_input = [data["url"]]  # Convert input to list format for vectorizer
    url_vectorized = vectorizer.transform(url_input)  # Apply TF-IDF transformation
    
    prediction = model.predict(url_vectorized)
    result = "Spam" if prediction[0] == 1 else "Safe"

    return jsonify({"url": url_input[0], "prediction": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)