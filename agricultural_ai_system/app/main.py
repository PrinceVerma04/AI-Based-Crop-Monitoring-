from flask import Flask, render_template, request, jsonify
import os
import io
import base64
import numpy as np
from PIL import Image
import cv2
import torch
import tensorflow as tf
import psutil
import gc
import logging
from typing import Dict, Any

# Configure TensorFlow and PyTorch memory settings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logging.error(f"GPU memory config error: {str(e)}")

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

# Import your analysis modules with lazy loading
from agricultural_ai_system.crop_monitor import CropMonitor
from agricultural_ai_system.disease_detector import PlantDiseaseDetector
from agricultural_ai_system.nutrient_analyser import NutrientAnalyzer
from agricultural_ai_system.pest_detector import PestDetector
from agricultural_ai_system.soil_monitor import SoilImageAnalyzer
from agricultural_ai_system.weed_detector import WeedDetector

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Memory management utilities
def memory_report() -> Dict[str, float]:
    """Generate system memory usage report"""
    mem = psutil.virtual_memory()
    return {
        'used_mb': mem.used / 1024 / 1024,
        'available_mb': mem.available / 1024 / 1024,
        'percent_used': mem.percent
    }

def cleanup() -> None:
    """Clean up memory resources"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info(f"Memory after cleanup: {memory_report()}")

# Lazy-loaded model instances
def get_crop_monitor():
    return CropMonitor.get_instance()

def get_disease_detector():
    return PlantDiseaseDetector.get_instance()

def get_nutrient_analyzer():
    return NutrientAnalyzer.get_instance()

def get_pest_detector():
    return PestDetector.get_instance()

def get_soil_analyzer():
    return SoilImageAnalyzer.get_instance()

def get_weed_detector():
    return WeedDetector.get_instance()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/system_status')
def system_status():
    """Endpoint to check system health"""
    try:
        status = {
            'memory': memory_report(),
            'models_loaded': {
                'crop_monitor': get_crop_monitor() is not None,
                'disease_detector': get_disease_detector() is not None,
                'nutrient_analyzer': get_nutrient_analyzer() is not None,
                'pest_detector': get_pest_detector() is not None,
                'soil_analyzer': get_soil_analyzer() is not None,
                'weed_detector': get_weed_detector() is not None
            }
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_image():
    # Initialize variables
    temp_path = None
    analysis_results = {}
    
    try:
        # Memory check before processing
        mem_before = memory_report()
        logging.info(f"Memory at start: {mem_before}")

        # Validate input
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        analysis_type = request.form.get('analysis_type')
        if not analysis_type:
            return jsonify({'error': 'No analysis type specified'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No selected image file'}), 400

        # Save and validate image
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(temp_path)
        
        img = Image.open(temp_path).convert('RGB')
        img_array = np.array(img)
        
        # Process based on analysis type
        if analysis_type == 'crop':
            analyzer = get_crop_monitor()
            if not analyzer:
                return jsonify({'error': 'Crop monitor not available'}), 503
            analysis_results = analyzer.analyze_crop(temp_path)
        
        elif analysis_type == 'disease':
            analyzer = get_disease_detector()
            if not analyzer:
                return jsonify({'error': 'Disease detector not available'}), 503
            analysis_results = analyzer.generate_visual_report(temp_path)
        
        elif analysis_type == 'nutrient':
            analyzer = get_nutrient_analyzer()
            if not analyzer:
                return jsonify({'error': 'Nutrient analyzer not available'}), 503
            analysis_results = analyzer.analyze_leaf(temp_path)
        
        elif analysis_type == 'pest':
            analyzer = get_pest_detector()
            if not analyzer:
                return jsonify({'error': 'Pest detector not available'}), 503
            analysis_results = analyzer.predict_pest(temp_path)
        
        elif analysis_type == 'soil':
            analyzer = get_soil_analyzer()
            if not analyzer:
                return jsonify({'error': 'Soil analyzer not available'}), 503
            analysis_results = analyzer.analyze_soil_image(temp_path)
        
        elif analysis_type == 'weed':
            analyzer = get_weed_detector()
            if not analyzer:
                return jsonify({'error': 'Weed analyzer not available'}), 503
            analysis_results = analyzer.detect_weed(temp_path)
            if 'visualization' in analysis_results:
                _, buffer = cv2.imencode('.jpg', 
                    cv2.cvtColor(analysis_results['visualization'], cv2.COLOR_RGB2BGR))
                analysis_results['visualization'] = \
                    f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        
        else:
            return jsonify({'error': 'Invalid analysis type'}), 400

        # Add memory metrics to response
        mem_after = memory_report()
        analysis_results['system_metrics'] = {
            'memory_before_mb': mem_before['used_mb'],
            'memory_after_mb': mem_after['used_mb'],
            'memory_delta_mb': mem_after['used_mb'] - mem_before['used_mb']
        }

        return jsonify(analysis_results)

    except Exception as e:
        logging.error(f"Analysis error: {str(e)}")
        return jsonify({
            'error': str(e),
            'system_metrics': {
                'memory_at_error': memory_report()
            }
        }), 500

    finally:
        # Cleanup in all cases
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        cleanup()

if __name__ == '__main__':
    # Pre-warm models in development
    if os.getenv('FLASK_ENV') == 'development':
        logging.info("Pre-warming models...")
        try:
            get_crop_monitor()
            get_disease_detector()
            get_nutrient_analyzer()
            get_weed_detector() 
            cleanup()
        except Exception as e:
            logging.warning(f"Pre-warm failed: {str(e)}")

    app.run(host='0.0.0.0', port=5000, debug=False)