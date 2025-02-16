import cv2
import numpy as np

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
