import os
import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
import warnings

import sys
sys.path.append(os.path.dirname(__file__))

from yedroudj_net_64 import yedroudj_net_64
from srm_filter_kernel import all_normalized_hpf_list

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

app = FastAPI(title="Steganalysis Detection API")

INPUT_SIZE = 256
MODEL = None

def load_model(weights_path: str):
    model = yedroudj_net_64(
        input_shape=(INPUT_SIZE, INPUT_SIZE, 1),
        all_normalized_hpf_list=all_normalized_hpf_list
    )
    model.compile(metrics=["accuracy"])
    model.load_weights(weights_path)
    print(f"Model loaded: {weights_path}")
    return model

def preprocess_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'L':
        img = img.convert('L')
    if img.size != (INPUT_SIZE, INPUT_SIZE):
        img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array[..., np.newaxis]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.on_event("startup")
async def startup_event():
    global MODEL
    weights_path = os.environ.get("MODEL_WEIGHTS_PATH")
    if not weights_path:
        print("Specify weights path via --weights")
        return
    if not os.path.exists(weights_path):
        print(f"Weights not found: {weights_path}")
        return
    MODEL = load_model(weights_path)

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse('''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Steganalysis Detector</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 25px 50px rgba(0,0,0,0.3);
            max-width: 900px;
            width: 100%;
            padding: 40px;
        }

        h1 {
            color: #1a1a2e;
            margin-bottom: 8px;
            font-size: 32px;
        }

        .badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: normal;
            margin-left: 10px;
        }

        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
            border-left: 3px solid #667eea;
            padding-left: 15px;
        }

        .upload-area {
            border: 3px dashed #ddd;
            border-radius: 16px;
            padding: 50px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #fafafa;
        }

        .upload-area:hover {
            border-color: #667eea;
            background: #f5f5ff;
            transform: translateY(-2px);
        }

        .upload-area.dragover {
            border-color: #667eea;
            background: #eeedff;
        }

        .upload-icon {
            font-size: 64px;
            margin-bottom: 15px;
        }

        .upload-text {
            color: #666;
            margin-bottom: 20px;
            font-size: 16px;
        }

        .file-input {
            display: none;
        }

        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 32px;
            border-radius: 30px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s ease;
            margin: 5px;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        .btn-secondary {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }

        .btn-secondary:hover {
            box-shadow: 0 5px 15px rgba(245, 87, 108, 0.4);
        }

        .preview-container {
            margin-top: 30px;
            text-align: center;
            display: none;
            flex-direction: column;
            align-items: center;
            gap: 15px;
        }

        .preview-image {
            max-width: 100%;
            max-height: 350px;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }

        .result {
            margin-top: 25px;
            padding: 25px;
            border-radius: 16px;
            display: none;
        }

        .result.infected {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            color: white;
        }

        .result.clean {
            background: linear-gradient(135deg, #a8e6cf 0%, #3b8d5e 100%);
            color: white;
        }

        .result-title {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 15px;
        }

        .probability-section {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            padding: 15px;
            margin: 15px 0;
        }

        .probability-bar-container {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
            height: 40px;
        }

        .probability-bar {
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 15px;
            font-weight: bold;
            transition: width 0.6s ease;
        }

        .probability-bar.stego {
            background: linear-gradient(90deg, #ff4757, #ff6b81);
        }

        .probability-bar.cover {
            background: linear-gradient(90deg, #2ed573, #7bed9f);
        }

        .probability-labels {
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
            font-size: 14px;
        }

        .confidence {
            font-size: 28px;
            font-weight: bold;
            text-align: center;
            margin: 15px 0;
        }

        .loader {
            display: none;
            text-align: center;
            margin: 30px 0;
        }

        .loader.active {
            display: block;
        }

        .spinner {
            width: 60px;
            height: 60px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>
            Steganalysis Detector
            <span class="badge">v1.0</span>
        </h1>
        <div class="subtitle">
            Detection of hidden data in images
        </div>

        <div class="upload-area" id="uploadArea">
            <div class="upload-icon">📸</div>
            <div class="upload-text">Drag and drop or click to select</div>
            <button class="btn" onclick="document.getElementById('fileInput').click()">Select file</button>
            <input type="file" id="fileInput" class="file-input" accept="image/jpeg,image/png,image/bmp,image/gif">
        </div>

        <div class="loader" id="loader">
            <div class="spinner"></div>
            <p style="margin-top: 15px; color: #666;">Analyzing...</p>
        </div>

        <div class="preview-container" id="previewContainer">
            <img id="previewImage" class="preview-image" alt="Preview">
            <button class="btn btn-secondary" onclick="resetAnalysis()">New image</button>
        </div>

        <div id="result" class="result"></div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const previewContainer = document.getElementById('previewContainer');
        const previewImage = document.getElementById('previewImage');
        const loader = document.getElementById('loader');
        const resultDiv = document.getElementById('result');

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                processFile(file);
            } else {
                showError('Please select an image file');
            }
        });

        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                processFile(file);
            }
        });

        function processFile(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                previewContainer.style.display = 'flex';
                uploadArea.style.display = 'none';
            };
            reader.readAsDataURL(file);
            analyzeImage(file);
        }

        async function analyzeImage(file) {
            loader.classList.add('active');
            resultDiv.style.display = 'none';

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Server error');
                }

                const data = await response.json();
                displayResult(data);
            } catch (error) {
                console.error('Error:', error);
                showError(error.message || 'Error analyzing image');
            } finally {
                loader.classList.remove('active');
            }
        }

        function displayResult(data) {
            const isInfected = data.is_infected;
            const confidence = (data.confidence * 100).toFixed(1);
            const probs = data.probabilities;

            resultDiv.className = `result ${isInfected ? 'infected' : 'clean'}`;
            resultDiv.innerHTML = `
                <div class="result-title">
                    ${isInfected ? 'HIDDEN DATA DETECTED' : 'CLEAN IMAGE'}
                </div>
                <div class="confidence">
                    Confidence: ${confidence}%
                </div>
                <div class="probability-section">
                    <div class="probability-bar-container">
                        <div class="probability-bar ${isInfected ? 'stego' : 'cover'}" 
                             style="width: ${isInfected ? (probs.has_payload * 100) : (probs.no_payload * 100)}%">
                            ${isInfected ? `${(probs.has_payload * 100).toFixed(1)}%` : `${(probs.no_payload * 100).toFixed(1)}%`}
                        </div>
                    </div>
                    <div class="probability-labels">
                        <span>No payload: ${(probs.no_payload * 100).toFixed(1)}%</span>
                        <span>Has payload: ${(probs.has_payload * 100).toFixed(1)}%</span>
                    </div>
                </div>
            `;
            resultDiv.style.display = 'block';
        }

        function showError(message) {
            resultDiv.className = 'result';
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = `
                <div style="background: #fee; color: #c00; padding: 20px; border-radius: 12px; text-align: center;">
                    Error: ${message}
                </div>
            `;
        }

        function resetAnalysis() {
            previewContainer.style.display = 'none';
            uploadArea.style.display = 'block';
            resultDiv.style.display = 'none';
            fileInput.value = '';
            previewImage.src = '';
        }
    </script>
</body>
</html>
    ''')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(400, detail="File must be an image")
    
    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(400, detail="File size must not exceed 50MB")
    
    if MODEL is None:
        raise HTTPException(503, detail="Model not loaded")
    
    try:
        processed_image = preprocess_image(contents)
        predictions = MODEL.predict(processed_image, verbose=0)
        
        prob_cover = float(predictions[0][0])
        prob_stego = float(predictions[0][1])
        
        is_stego = prob_stego > 0.5
        confidence = prob_stego if is_stego else prob_cover
        
        return {
            "is_infected": is_stego,
            "probabilities": {
                "no_payload": round(prob_cover, 6),
                "has_payload": round(prob_stego, 6)
            },
            "confidence": round(confidence, 6)
        }
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy" if MODEL is not None else "degraded"}

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", "-w", required=True, help="Path to .h5 weights file")
    parser.add_argument("--port", "-p", type=int, default=8000)
    args = parser.parse_args()
    
    if not os.path.exists(args.weights):
        print(f"File not found: {args.weights}")
        sys.exit(1)
    
    os.environ["MODEL_WEIGHTS_PATH"] = args.weights
    
    print("=" * 60)
    print("Steganalysis API Server")
    print("=" * 60)
    print(f"Weights: {args.weights}")
    print(f"http://localhost:{args.port}")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")