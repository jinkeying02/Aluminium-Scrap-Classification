import cv2
import numpy as np

def extract_foreground_with_removal(alu_img_bgr, back_img_bgr, threshold_value=10):
    """
    Method 1: Background Removal (Black Background)
    Subtracts background from the aluminum image and applies an inverse binary threshold.
    Returns a BGR image with a pure black background.
    """
    if alu_img_bgr is None or back_img_bgr is None:
        return alu_img_bgr

    # Ensure background size matches input
    if alu_img_bgr.shape != back_img_bgr.shape:
        back_img_bgr = cv2.resize(back_img_bgr, (alu_img_bgr.shape[1], alu_img_bgr.shape[0]))

    # 1. Background Subtraction & Grayscale
    diff_img = cv2.subtract(alu_img_bgr, back_img_bgr)
    gray_img = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY) if len(diff_img.shape) == 3 else diff_img

    # 2. Thresholding and Morphology
    _, mask = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 3. Apply mask to extract foreground
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    foreground = cv2.bitwise_and(alu_img_bgr, mask_3ch)
    return foreground


def extract_foreground_with_blur_bg(alu_img_bgr, back_img_bgr, threshold_value=20, blur_kernel_size=(30, 30)):
    """
    Method 2: Background Blurring
    Extracts foreground and blends it with a heavily blurred version of the original image.
    Returns a BGR image with a blurred background.
    """
    if alu_img_bgr is None or back_img_bgr is None:
        return alu_img_bgr

    if alu_img_bgr.shape != back_img_bgr.shape:
        back_img_bgr = cv2.resize(back_img_bgr, (alu_img_bgr.shape[1], alu_img_bgr.shape[0]))

    diff_img = cv2.subtract(alu_img_bgr, back_img_bgr)
    gray_img = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY) if len(diff_img.shape) == 3 else diff_img

    _, initial_mask = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    eroded_mask = cv2.erode(initial_mask, kernel, iterations=1)
    processed_mask = cv2.dilate(eroded_mask, kernel, iterations=2)

    inverted_mask = cv2.bitwise_not(processed_mask)
    inverted_mask_3ch = cv2.cvtColor(inverted_mask, cv2.COLOR_GRAY2BGR)

    blurred_bg = cv2.GaussianBlur(alu_img_bgr, blur_kernel_size, 0)
    result_final = np.where(inverted_mask_3ch == 255, alu_img_bgr, blurred_bg)
    return result_final
 

# ==============================================================================
# LEGCY / ALTERNATIVE PROCESSORS (For Baseline Comparison)
# ==============================================================================
   
def extract_foreground_mog2(img_bgr, back_img_bgr, history=2, var_threshold=150):
    """
    Alternative Method 1: Foreground extraction using MOG2 background subtractor.
    Takes BGR images as input and returns a BGR image with background blurred.
    """
    if img_bgr is None or back_img_bgr is None:
        return img_bgr

    # Ensure shapes match
    if img_bgr.shape != back_img_bgr.shape:
        back_img_bgr = cv2.resize(back_img_bgr, (img_bgr.shape[1], img_bgr.shape[0]))

    # Initialize MOG2 locally to prevent serialization/pickle issues
    fgbg = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold, detectShadows=False)
    
    # Train on single background frame
    fgbg.apply(back_img_bgr)
    mog_mask = fgbg.apply(img_bgr, learningRate=-1)

    # Morphological operations
    kernel = np.ones((4, 4), np.uint8)
    mog_mask = cv2.erode(mog_mask, kernel, iterations=1)
    mog_mask = cv2.dilate(mog_mask, kernel, iterations=3)

    # Blend foreground with blurred background
    background_mask = cv2.bitwise_not(mog_mask)
    blurred_image = cv2.GaussianBlur(img_bgr, (21, 21), 0)
    
    foreground_only = cv2.bitwise_and(img_bgr, img_bgr, mask=mog_mask)
    background_only = cv2.bitwise_and(blurred_image, blurred_image, mask=background_mask)
    
    return cv2.add(foreground_only, background_only)


def extract_foreground_connected_components(alu_image_path, back_image_path, min_area_ratio=0.0005, max_area_ratio=0.8):
    """
    Alternative Method 2: MOG2 combined with Connected Component Analysis to filter out noise
    by keeping only the component closest to the image center.
    Returns the binary mask (Inverted: foreground black, background white as per legacy design).
    """
    alu_img = cv2.imread(alu_image_path)
    back_img = cv2.imread(back_image_path)

    if alu_img is None or back_img is None:
        return None

    if alu_img.shape != back_img.shape:
        back_img = cv2.resize(back_img, (alu_img.shape[1], alu_img.shape[0]))

    fgbg = cv2.createBackgroundSubtractorMOG2(history=2, varThreshold=150, detectShadows=False)
    fgbg.apply(back_img)
    initial_mog_mask = fgbg.apply(alu_img, learningRate=-1)

    # Morphological cleaning
    kernel = np.ones((3, 3), np.uint8)
    clean_mog_mask = cv2.morphologyEx(initial_mog_mask, cv2.MORPH_OPEN, kernel, iterations=3)
    clean_mog_mask = cv2.morphologyEx(clean_mog_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_mog_mask, connectivity=8)

    img_h, img_w = alu_img.shape[:2]
    center_x, center_y = img_w // 2, img_h // 2
    img_area = img_w * img_h
    
    min_area_threshold = img_area * min_area_ratio
    max_area_threshold = img_area * max_area_ratio

    best_label = -1
    min_distance_to_center = float('inf')

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area_threshold < area < max_area_threshold:
            cx, cy = centroids[i]
            distance = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
            if distance < min_distance_to_center:
                min_distance_to_center = distance
                best_label = i

    final_mask = np.zeros_like(labels, dtype=np.uint8)
    if best_label != -1:
        final_mask[labels == best_label] = 255

    # Return inverted mask as per original requirement
    return cv2.bitwise_not(final_mask)


# ==============================================================================
# ALTERNATIVE SEGMENTATION METHODS (Explored Baselines)
# ==============================================================================

def create_canny_mask(alu_img_path, back_img_path, lower_threshold=30, upper_threshold=200, aperture_size=3):
    """
    Alternative Method 3: Generates a binary mask using the Canny edge detection algorithm
    after background subtraction and smoothing.
    
    Returns:
        np.ndarray: Binary mask where edges are 255 and background is 0.
    """
    try:
        alu_img = cv2.imread(alu_img_path)
        back_img = cv2.imread(back_img_path)

        if alu_img is None or back_img is None:
            return None

        if alu_img.shape != back_img.shape:
            back_img = cv2.resize(back_img, (alu_img.shape[1], alu_img.shape[0]))

        # Background subtraction to isolate changes
        diff_image = cv2.subtract(alu_img, back_img)
        gray_image = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred_image, lower_threshold, upper_threshold, apertureSize=aperture_size)
        return edges
    except Exception as e:
        print(f"Error in Canny mask generation: {e}")
        return None


def perform_roi_extraction(img_bgr, min_contour_area=100):
    """
    Alternative Method 4: Performs adaptive thresholding and contour detection 
    To segment and extract Regions of Interest (ROIs) from a BGR image.
    
    Args:
        img_bgr (np.ndarray): Input BGR image matrix.
        min_contour_area (int): Minimum contour area to filter out small noise.
        
    Returns:
        tuple: (cropped_rois_bgr, roi_bboxes, binary_mask)
    """
    if img_bgr is None:
        return [], [], None

    try:
        # Pre-processing flow: Gray -> Blur -> Adaptive Threshold
        gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(gray_img, (9, 9), 0)
        
        # Adaptive thresholding optimized for darker objects on bright backgrounds
        binary_img = cv2.adaptiveThreshold(
            blurred_img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 5, 2
        )

        # Morphological operations to clean up small noise fragments
        kernel = np.ones((3, 3), np.uint8)
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=1)
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Find external boundaries
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cropped_rois_bgr = []
        roi_bboxes = []

        for contour in contours:
            if cv2.contourArea(contour) > min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                cropped_roi = img_bgr[y:y+h, x:x+w]
                cropped_rois_bgr.append(cropped_roi)
                roi_bboxes.append((x, y, w, h))

        return cropped_rois_bgr, roi_bboxes, binary_img
    except Exception as e:
        print(f"Error during ROI extraction: {e}")
        return [], [], None