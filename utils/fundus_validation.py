"""Fundus image validation for VisionCare."""

import cv2
import numpy as np


def validate_fundus_image(image: np.ndarray) -> tuple[bool, str]:
    """
    Validate a fundus image for quality and characteristics.
    
    Returns:
        (is_valid, message)
    """
    if image is None or len(image.shape) not in [2, 3]:
        return False, "Invalid fundus image format"
    
    # Check image dimensions - fundus images are typically larger than OCT
    height, width = image.shape[:2]
    if height < 300 or width < 300:
        return False, f"Fundus image too small: {width}x{height}. Expected minimum 300x300"
    
    if height > 5000 or width > 5000:
        return False, f"Fundus image too large: {width}x{height}. Expected maximum 5000x5000"
    
    # Check if image has color information (fundus images are typically RGB)
    if len(image.shape) == 2:
        # Convert grayscale to BGR for processing
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        # Convert RGBA to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif image.shape[2] != 3:
        return False, f"Fundus image must have 3 or 4 channels, got {image.shape[2]}"
    
    # Check for minimum brightness and contrast
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    
    if mean_brightness < 20:
        return False, "Fundus image is too dark"
    
    if mean_brightness > 230:
        return False, "Fundus image is too bright (overexposed)"
    
    if std_brightness < 5:
        return False, "Fundus image has insufficient contrast"
    
    # Check for circular/disc-like pattern (characteristic of fundus)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) < 5:
        return False, "Fundus image pattern not recognized (may not be a valid fundus photo)"
    
    # Calculate color distribution to detect fundus characteristics
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Fundus images typically have strong red/orange components
    red_mask = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    orange_mask = cv2.inRange(hsv, np.array([10, 100, 100]), np.array([25, 255, 255]))
    combined_mask = cv2.bitwise_or(red_mask, orange_mask)
    
    red_orange_ratio = np.sum(combined_mask > 0) / (height * width)
    
    if red_orange_ratio < 0.05:
        return False, "Fundus image color distribution unusual (low red/orange components)"
    
    return True, "Fundus image valid"


def detect_image_type(image: np.ndarray) -> str:
    """
    Detect whether image is OCT or Fundus based on characteristics.
    
    Returns:
        'OCT' or 'FUNDUS'
    """
    if image is None:
        return 'OCT'
    
    height, width = image.shape[:2]
    
    # OCT images are typically near-square, fundus images are more circular
    aspect_ratio = width / height if height > 0 else 1
    
    # OCT images are usually grayscale or have distinct layers
    if len(image.shape) == 2:
        # Single channel - likely OCT
        return 'OCT'
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # OCT images have high vertical entropy (distinct layers)
    # Calculate entropy along columns
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    # Fundus images have higher entropy and more uniform color
    if len(image.shape) == 3:
        # Check color variance
        b_var = np.var(image[:, :, 0])
        g_var = np.var(image[:, :, 1])
        r_var = np.var(image[:, :, 2])
        color_similarity = max(b_var, g_var, r_var) / (min(b_var, g_var, r_var) + 1)
        
        # Fundus images have more similar color distributions
        if color_similarity < 3:
            return 'FUNDUS'
    
    # If aspect ratio close to 1 and moderate entropy, likely OCT
    if abs(aspect_ratio - 1.0) < 0.3 and entropy < 5:
        return 'OCT'
    
    return 'FUNDUS'
