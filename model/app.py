from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.applications.densenet import preprocess_input
import io
from PIL import Image
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# --- Load model and encodings at startup ---
print("Loading model...")
model = load_model('model_2.keras', compile=False)
print("Model loaded successfully!")

print("Loading tokenizer mappings...")
with open("wordtoix.pkl", "rb") as f:
    words_to_index = pickle.load(f)

with open("ixtoword.pkl", "rb") as f:
    index_to_words = pickle.load(f)
print("Tokenizer loaded successfully!")

# --- Load image encoder (DenseNet121) ---
from keras.applications.densenet import DenseNet121
from keras.models import Model as KerasModel

print("Loading image encoder...")
input_shape = (224, 224, 3)
base_model = DenseNet121(weights="imagenet", include_top=False, pooling="avg", input_shape=input_shape)
print("Image encoder loaded successfully!")

# --- Parameters ---
max_length = 124  # same as training

# --- Caption generation function ---
def generate_caption(image_features, max_steps=40, temperature=0.5, top_k=3):
    """
    Generate a caption for a given image feature vector.
    """
    in_text = 'startseq'
    generated_words = []
    
    for _ in range(max_steps):
        # Convert current text to sequence
        sequence = [words_to_index.get(w, words_to_index.get('<unk>', 1)) for w in in_text.split()]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        
        # Predict next word probabilities
        yhat = model([image_features, sequence], training=False)
        probabilities = yhat.numpy().ravel()
        
        # Temperature scaling
        probabilities = np.exp(np.log(probabilities + 1e-10) / temperature)
        probabilities /= np.sum(probabilities)
        
        # Top-k sampling
        top_indices = np.argsort(probabilities)[-top_k:]
        top_probs = probabilities[top_indices] / np.sum(probabilities[top_indices])
        yhat_index = np.random.choice(top_indices, p=top_probs)
        
        word = index_to_words.get(yhat_index, '<unk>')
        
        # Stop conditions
        if word == 'endseq':
            break
        if word in ['<unk>', 'xxxx', 'startseq']:
            continue
        
        # Avoid repetition
        if len(generated_words) >= 2 and word == generated_words[-1] == generated_words[-2]:
            break
        
        generated_words.append(word)
        in_text += ' ' + word
    
    return ' '.join(generated_words)

# --- Extract image features ---
def extract_features(image_file):
    """
    Extract features from uploaded image using DenseNet121
    """
    # Read image
    img = Image.open(io.BytesIO(image_file.read()))
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to model input size
    img = img.resize((224, 224))
    
    # Convert to array
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess
    img_array = preprocess_input(img_array)
    
    # Extract features
    features = base_model.predict(img_array, verbose=0)
    
    return features

@app.route('/generate-report', methods=['POST'])
def generate_report():
    """
    API endpoint to generate X-ray report from uploaded image
    """
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        print(f"Processing image: {image_file.filename}")
        
        # Extract image features
        print("Extracting image features...")
        image_features = extract_features(image_file)
        print(f"Features shape: {image_features.shape}")
        
        # Generate caption
        print("Generating caption...")
        caption = generate_caption(image_features)
        print(f"Generated caption: {caption}")
        
        # Calculate confidence (mock for now - you can implement actual confidence calculation)
        confidence = 0.85 + np.random.uniform(-0.1, 0.1)
        confidence = max(0.5, min(0.99, confidence))
        
        # Format response
        response = {
            'findings': caption.capitalize() + '.',
            'impression': extract_impression(caption),
            'confidence': float(confidence),
            'timestamp': np.datetime64('now').astype(str)
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 500

def extract_impression(caption):
    """
    Extract key impression from full caption
    """
    # Simple heuristic: look for key phrases
    if 'no acute' in caption.lower() or 'no evidence' in caption.lower():
        return "No acute abnormality identified"
    elif 'normal' in caption.lower():
        return "Normal study"
    elif 'clear' in caption.lower():
        return "Clear lungs"
    else:
        # Return first sentence or first 50 chars
        first_sentence = caption.split('.')[0]
        return first_sentence if len(first_sentence) < 60 else caption[:60] + "..."

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'message': 'X-Ray Report Generator API is running'
    }), 200

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 X-Ray Report Generator API Starting...")
    print("="*50)
    print("\n📡 Server will run at: http://localhost:5000")
    print("🏥 API endpoint: http://localhost:5000/generate-report")
    print("💚 Health check: http://localhost:5000/health")
    print("\n" + "="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)