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

- Python 3.8+
- Django 5.1.5
- Flask
- OpenCV
- NumPy

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/1javid/evaleasy-ms-omr.git
    cd evaleasy-ms-omr/ms_omr
    ```

2. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Apply the migrations:
    ```sh
    python manage.py migrate
    ```

4. Run the development server:
    ```sh
    python manage.py runserver 8004

## Usage

1. Start the Django development server:
    ```sh
    python manage.py runserver 8004
    ```

2. Use a tool like Postman or cURL to send a POST request to the `/process_image` endpoint with the image file:
    ```bash
    curl -X POST -F 'file=@path_to_your_image_file' http://127.0.0.1:8004/process_image
    ```

3. The API will return a JSON response containing the extracted information and a Base64-encoded image of the processed answer sheet.

## Project Structure

- `bubble_detection.py`: Functions for detecting and processing bubbles.
- `contour_detection.py`: Functions for extracting and sorting contours.
- `image_processing.py`: Functions for preprocessing images.
- `section_processing.py`: Functions for processing different sections of the answer sheet.