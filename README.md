# Evaleasy MS OMR

Evaleasy MS OMR is an Optical Mark Recognition (OMR) system designed to process and analyze scanned answer sheets. This project uses image processing techniques to detect and interpret marked bubbles on answer sheets, extracting relevant information such as student IDs, assessment IDs, and answers.

## Features

- Detect and preprocess uploaded images
- Extract and sort contours
- Detect and mark bubbles in different sections
- Group bubbles into rows and assign labels
- Map detected bubbles to corresponding letters/numbers
- Generate results including student information and answers

## Requirements

- Python 3.6+
- Flask
- OpenCV
- NumPy

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/1javid/evaleasy-ms-omr.git
    cd evaleasy-ms-omr
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Start the Flask application:
    ```bash
    python main.py
    ```

2. Use a tool like Postman or cURL to send a POST request to the `/process_image` endpoint with the image file:
    ```bash
    curl -X POST -F 'file=@path_to_your_image_file' http://127.0.0.1:5000/process_image
    ```

3. The API will return a JSON response containing the extracted information and a Base64-encoded image of the processed answer sheet.

## Project Structure

- `main.py`: The main Flask application file.
- `utils/`: Contains helper functions and image processing utilities.
- `bubble_detection.py`: Functions for detecting and processing bubbles.
- `contour_detection.py`: Functions for extracting and sorting contours.
- `image_processing.py`: Functions for preprocessing images.
- `section_processing.py`: Functions for processing different sections of the answer sheet.