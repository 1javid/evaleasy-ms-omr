import cv2
import os
import numpy as np

def preprocess_uploaded_file(file):
    """
    Decode and validate the uploaded file as an image.
    :param file: Uploaded file object.
    :return: Decoded image.
    """
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Invalid image file")
    return img

def preprocess_with_blur_and_threshold(img):
    """
    Preprocess the input image by converting to grayscale, applying Gaussian blur, 
    and thresholding.
    :param img: Original input image.
    :return: Thresholded image.
    """
    # Draw thick rectangle around the image to avoid black contour around the image
    cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (255, 255, 255), thickness=75)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return thresh

def save_image_with_border(img, contours, output_dir):
    """
    Save the image with contours marked in green.
    :param img: Input image.
    :param contours: List of contours to be marked.
    :param output_dir: Directory to save the output image.
    """
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    cv2.imwrite(os.path.join(output_dir, 'image_with_border.png'), img)