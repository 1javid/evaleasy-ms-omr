import cv2
import numpy as np

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
