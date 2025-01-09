import logging
from flask import Flask, request, jsonify
from utils.image_processing import preprocess_uploaded_file
from utils.section_processing import preprocess_transformed_image
from utils.helper_functions import create_temp_output_dir, cleanup_temp_output_dir

# Initialize Flask app and logging
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/process_image', methods=['POST'])
def detect_omr():
    """
    API endpoint to process an OMR image and extract relevant information.
    """
    try:
        # Validate and preprocess the uploaded file
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file provided'}), 400

        img = preprocess_uploaded_file(file)
        
        # Create a temporary output directory
        output_dir = create_temp_output_dir()
        try:
            # Main image processing logic
            result = preprocess_transformed_image(img, output_dir)
            return jsonify(result)
        finally:
            cleanup_temp_output_dir(output_dir)
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting application.")
    app.run(host='0.0.0.0', port=5000)
