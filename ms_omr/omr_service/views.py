import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


import shutil
import tempfile
import numpy as np
import cv2
import os
import base64

def detect_bubbles(section, detect_marked):
    """
    Detect bubbles in a section using Hough Circles and filter out unmarked ones based on intensity.
    :param section: Image section.
    :param detect_marked: Flag to detect only marked bubbles (if True).
    :return: List of detected bubbles (x, y, radius).
    """
    
    gray_section = cv2.cvtColor(section, cv2.COLOR_BGR2GRAY)    
    blurred = cv2.GaussianBlur(gray_section, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.1, minDist=15,
                               param1=50, param2=25, minRadius=8, maxRadius=15)

    section_bubbles = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if detect_marked:
                # Crop the circular region from the grayscale image
                mask = np.zeros_like(gray_section)
                cv2.circle(mask, (x, y), r, 255, -1)
                bubble_region = cv2.bitwise_and(gray_section, gray_section, mask=mask)
                
                # Calculate the mean intensity within the circle
                mean_intensity = cv2.mean(bubble_region, mask=mask)[0]
    
                # Assume filled bubbles have a lower mean intensity (darker)
                if mean_intensity < 100:  # Threshold for marking
                    section_bubbles.append((x, y, r))
            else:
                # For all bubbles, add them regardless of intensity
                section_bubbles.append((x, y, r))

    return section_bubbles

def detect_marked_section_bubbles(img, contours, output_dir):
    """
    Process all sections, detecting bubbles and saving outputs.
    :param img: Input image.
    :param contours: List of contours from the image.
    :param output_dir: Directory to save outputs.
    :return: Tuple of sections and detected bubbles.
    """
    sections, marked_bubbles_list = {}, {}

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 50:  # Filter out small regions
            section = img[y:y + h, x:x + w]
            sections[f"section_{i}"] = section

            # Detect and mark bubbles
            marked_bubbles = detect_bubbles(section, detect_marked=True)
            if not marked_bubbles:
                continue
            marked_bubbles_list[f"section_{i}"] = marked_bubbles

            # Highlight the marked bubbles
            output = section.copy()
            for (x, y, r) in marked_bubbles:
                cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.imwrite(f"{output_dir}/section_{i}_highlighted.png", output)

    return sections, marked_bubbles_list

def group_bubbles_into_rows(marked_bubbles, min_y_threshold=20):
    """
    Group bubbles into rows based on their y-coordinates.
    :param marked_bubbles: List of bubbles as (x, y, radius) tuples.
    :param min_y_threshold: Minimum y-difference to group bubbles into rows.
    :return: List of rows, where each row is a list of (x, y, radius) tuples.
    """
    marked_bubbles.sort(key=lambda v: (v[1], v[0]))  # Sort by y-coordinate, then by x-coordinate

    rows, current_row = [], []

    for (x, y, r) in marked_bubbles:
        if not current_row:
            # First circle, add it to the current row
            current_row.append((x, y, r))
        else:
            # Check if this bubble belongs to the same row
            last_y = current_row[-1][1]
            if abs(last_y - y) < min_y_threshold:
                current_row.append((x, y, r))  # Add to the same row
            else:
                # If the difference is too large, it's a new row
                rows.append(current_row)
                current_row = [(x, y, r)]  # Start new row

    if current_row:
        rows.append(current_row)  # Add the last row if exists

    return rows

def assign_labels_to_rows(rows, section_name):
    """
    Assign labels (letters/numbers) to rows based on the section-specific rules.
    :param rows: List of rows, where each row is a list of (x, y, radius) tuples.
    :param section_name: Name of the section to determine the labeling rules.
    :return: List of rows with assigned labels as (x, y, label) tuples.
    """
    
    bubble_map = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    number_map = '0123456789'

    mapped_rows = []
    if section_name == 'section_1' or section_name == 'section_2':  # Each row gets the same letter
        for i, row in enumerate(rows):
            letter = bubble_map[i % len(bubble_map)]  # Use letters cyclically
            mapped_row = [(x, y, letter) for x, y, r in row]
            mapped_rows.append(mapped_row)
    elif section_name == 'section_3' or section_name == 'section_4':  # Each row gets the same number
        for i, row in enumerate(rows):
            number = number_map[i % len(number_map)]  # Use numbers cyclically
            mapped_row = [(x, y, number) for x, y, r in row]
            mapped_rows.append(mapped_row)
    elif section_name == 'section_0' or section_name == 'section_7':  # Each bubble in a row gets a different letter
        for row in rows:
            row.sort(key=lambda v: v[0])  # Sort by x-coordinate
            mapped_row = [(x, y, bubble_map[i % len(bubble_map)]) for i, (x, y, r) in enumerate(row)]
            mapped_rows.append(mapped_row)
    
    return mapped_rows

def find_row_positions(sections, min_y_threshold=20):
    """
    Map detected bubbles to their rows and assign labels to them.
    :param sections: Dictionary of section names and their image sections.
    :param min_y_threshold: Minimum y-difference to group bubbles into rows.
    :return: Dictionary with section names as keys and rows with assigned labels as values.
    """
    row_positions = {}
   
    for s in sections:
        section = sections[s]

        marked_bubbles = detect_bubbles(section, detect_marked=False)
        
        # Group bubbles into rows
        rows = group_bubbles_into_rows(marked_bubbles, min_y_threshold)

        for i, row in enumerate(rows):
            row.sort(key=lambda v: v[0])  # Sort by x-coordinate

        # Assign labels based on section-specific rules
        mapped_rows = assign_labels_to_rows(rows, section_name=s)        
        row_positions[s] = mapped_rows

    return row_positions

def map_marked_section_bubbles(section, section_bubbles, row_bubble_positions, threshold):
    """
    Map detected bubbles to their corresponding letters/numbers.
    :param section: Name of the section being processed.
    :param section_bubbles: List of detected bubbles as (x, y, radius).
    :param row_bubble_positions: Dictionary of row bubble positions with labels.
    :param threshold: Threshold to match positions.
    :return: List of (row/col, label) tuples.
    """

    section_letters = []
    for (x, y, r) in section_bubbles:
        for idx, row in enumerate(row_bubble_positions[section]):
            for bubble_x, bubble_y, label in list(row):
                # Check if the position is similar (within the threshold)
                if abs(x - bubble_x) <= threshold and abs(y - bubble_y) <= threshold:
                    if section == 'section_0' or section == 'section_7':
                        section_letters.append((f"row_{idx + 1}", label))
                    else:
                        section_letters.append((f"col_{row.index((bubble_x, bubble_y, label)) + 1}", label))  # Store the column index
                    break  # Once we find a match, stop checking other positions for this bubble
    return section_letters

def extract_contours(thresh):
    """
    Extract contours from the thresholded image.
    :param thresh: Thresholded image.
    :return: List of contours.
    """
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def sort_contours(contours):
    """
    Sort contours by area with the greatest first.
    :param contours: List of contours.
    :return: Sorted contours.
    """
    return sorted(contours, key=cv2.contourArea, reverse=True)

def transform_image(img):
    """
    Transform the input image to get a top-down view of the largest detected contour.
    :param img: Input image.
    :return: Warped image with a top-down view of the largest contour.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 80)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest_contour) < 800000:
        cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 0), thickness=25)
        return transform_image(img)

    # Visualize the detected contour on the original image
    contour_visualization = img.copy()
    cv2.drawContours(contour_visualization, [largest_contour], -1, (0, 255, 0), 3)  # Green contour
    cv2.imwrite("output/contour_visualization.png", contour_visualization)

    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Ensure the approximated contour has 4 points (the corners of the paper)
    if len(approx) == 4:
        # Sort the points in top-left, top-right, bottom-right, bottom-left order
        points = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        # Calculate sums and differences of points to identify corners
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]  # Top-left
        rect[2] = points[np.argmax(s)]  # Bottom-right

        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]  # Top-right
        rect[3] = points[np.argmax(diff)]  # Bottom-left

        # Define the width and height of the new image
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        # Destination points for the top-down view of the paper
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)

        # Warp the perspective to get a top-down view of the paper
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        return warped

    else:
        print("Failed to detect a proper rectangular contour for the paper.")

def create_temp_output_dir():
    """
    Create a temporary directory for storing output files.
    """
    return tempfile.mkdtemp()

def cleanup_temp_output_dir(output_dir):
    """
    Clean up the temporary output directory.
    """
    shutil.rmtree(output_dir)

def sort_bubbles(bubbles, sort_by_row=True):
    """
    Sort bubbles by their coordinates, either by row or column.
    """
    if sort_by_row:
        return sorted(bubbles, key=lambda bubble: (bubble[1], bubble[0]))
    return sorted(bubbles, key=lambda bubble: (bubble[0], bubble[1]))

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

SECTION_NAME_INDEX = 2
LAST_NAME_INDEX = 1
ASSESSMENT_ID_INDEX = 3
STUDENT_ID_INDEX = 4
VARIANT_INDEX = 7
ANSWERS_SECTION_INDEX = 0

def preprocess_transformed_image(img, output_dir):
    """
    Main processing logic for the OMR image.
    """
    # Save the original image for reference
    cv2.imwrite(os.path.join(output_dir, "transformed_answer_sheet.png"), img)

    # Preprocess the image
    thresh = preprocess_with_blur_and_threshold(img)

    # Detect and sort contours
    cnts = extract_contours(thresh.copy())
    cnts = sort_contours(cnts)

    # Extract and process sections
    sections, marked_bubbles_list = detect_marked_section_bubbles(img, cnts, output_dir=output_dir)

    # Draw bounding rectangles on contours
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

    # Save the final image with borders
    save_image_with_border(img, cnts, output_dir=output_dir)

    # Sort marked bubbles for each section
    for section_name in marked_bubbles_list:
        marked_bubbles_list[section_name] = sort_bubbles(
            marked_bubbles_list[section_name],
            sort_by_row=(section_name in ['section_0', 'section_7'])
        )

    # Identify row positions for grouping bubbles
    row_bubble_positions = find_row_positions(sections)

    # Map bubbles to letters/numbers for each section
    section_letters = {}
    for section, bubbles in marked_bubbles_list.items():
        section_letters[section] = map_marked_section_bubbles(
            section, bubbles, row_bubble_positions, threshold=3
        )

    # Build the result dictionary
    result = build_result(section_letters, output_dir)

    return result

def build_result(section_letters, output_dir):
    """
    Construct the final result dictionary from section letters.
    """
    result = {
        'firstName': ''.join([letter for _, letter in section_letters.pop(f'section_{SECTION_NAME_INDEX}')]),
        'lastName': ''.join([letter for _, letter in section_letters.pop(f'section_{LAST_NAME_INDEX}')]),
        'assessmentID': ''.join([number for _, number in section_letters.pop(f'section_{ASSESSMENT_ID_INDEX}')]),
        'studentID': ''.join([number for _, number in section_letters.pop(f'section_{STUDENT_ID_INDEX}')]),
        'variant': ''.join([letter for _, letter in section_letters.pop(f'section_{VARIANT_INDEX}')]),
        'answers': {}
    }

    # Group answers by question number
    answers_temp = section_letters.pop(f'section_{ANSWERS_SECTION_INDEX}')
    answers_grouped = {}
    for row_or_col, letter in answers_temp:
        question_number = int(row_or_col.split('_')[1])  # Extract row/column number
        if question_number not in answers_grouped:
            answers_grouped[question_number] = []
        answers_grouped[question_number].append(letter)

    # Convert grouped answers to a sorted dictionary
    result['answers'] = {key: value for key, value in sorted(answers_grouped.items())}

    # Read the image and encode it in Base64
    transformed_img_path = os.path.join(output_dir, "transformed_answer_sheet.png")
    with open(transformed_img_path, 'rb') as img_file:
        img_data = base64.b64encode(img_file.read()).decode('utf-8')

    result['student_sheet'] = img_data  # Return Base64-encoded image data

    return result

logger = logging.getLogger(__name__)

@csrf_exempt
def process_image(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests allowed'}, status=405)
    try:
        file = request.FILES.get('file')
        if not file:
            return JsonResponse({'error': 'No file provided'}, status=400)
        img = preprocess_uploaded_file(file)
        output_dir = create_temp_output_dir()
        try:
            result = preprocess_transformed_image(img, output_dir)
            return JsonResponse(result)
        finally:
            cleanup_temp_output_dir(output_dir)
    except Exception as e:
        logger.error("Error processing image: %s", str(e), exc_info=True)
        return JsonResponse({'error': str(e)}, status=500)
