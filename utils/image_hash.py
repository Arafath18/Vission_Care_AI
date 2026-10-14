"""Image hashing utility for deduplication."""

import hashlib
import numpy as np
import cv2


def calculate_image_hash(image: np.ndarray, resize_to: tuple = (8, 8)) -> str:
    """
    Calculate perceptual hash of an image for deduplication.
    
    Uses Average Hash algorithm - compares image to average pixel value.
    More tolerant to compression and minor edits than file hash.
    
    Args:
        image: Input image as numpy array
        resize_to: Size to resize for hashing (default 8x8)
    
    Returns:
        Hex hash string
    """
    if image is None:
        return ""
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Resize to small size
    small = cv2.resize(gray, resize_to)
    
    # Calculate average pixel value
    avg = small.mean()
    
    # Create hash: 1 if pixel > avg, 0 otherwise
    hash_bits = (small > avg).flatten().astype(int)
    
    # Convert to hex string
    hash_hex = hex(int(''.join(map(str, hash_bits)), 2))[2:]
    
    return hash_hex.zfill(16)


def calculate_file_hash(image: np.ndarray) -> str:
    """
    Calculate exact file hash using SHA256.
    
    This is stricter than perceptual hash - any compression
    or modification will result in different hash.
    """
    if image is None:
        return ""
    
    # Serialize image to bytes
    image_bytes = image.tobytes()
    
    # Calculate SHA256
    return hashlib.sha256(image_bytes).hexdigest()


def images_are_similar(hash1: str, hash2: str, max_distance: int = 5) -> bool:
    """
    Compare two perceptual hashes using Hamming distance.
    
    Returns True if images are similar (Hamming distance <= max_distance).
    """
    if not hash1 or not hash2:
        return False
    
    try:
        # Convert hex hashes to integers
        h1 = int(hash1, 16)
        h2 = int(hash2, 16)
        
        # Calculate Hamming distance (count differing bits)
        xor = h1 ^ h2
        distance = bin(xor).count('1')
        
        return distance <= max_distance
    except (ValueError, AttributeError):
        return False
