import cv2
import os
import base64
from helper_functions import sort_bubbles
from image_processing import preprocess_with_blur_and_threshold, save_image_with_border
from contour_detection import extract_contours, sort_contours
from bubble_detection import detect_marked_section_bubbles, find_row_positions, map_marked_section_bubbles

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
