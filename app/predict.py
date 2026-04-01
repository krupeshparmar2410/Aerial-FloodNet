"""
predict.py — FloodNet Prediction Helper
Loads saved .pkl models and runs predictions for all 3 tasks.
"""

import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
OUTPUTS_DIR= os.path.join(BASE_DIR, "outputs")

# ── Config (must match notebook) ──────────────────────────────
IMG_SIZE   = (128, 128)
PATCH_SIZE = (32, 32)
STEP       = 16

SEG_COLORS = {
    0:(55,71,79), 1:(255,61,90), 2:(255,152,0), 3:(224,64,251),
    4:(124,77,255), 5:(0,229,255), 6:(0,230,118), 7:(255,214,0),
    8:(64,196,255), 9:(105,240,174)
}
CLASS_NAMES = ["Background","Building-Flooded","Building-Non-Flood",
               "Road-Flooded","Road-Non-Flood","Water",
               "Tree","Vehicle","Pool","Grass"]


def extract_features(img_bgr):
    img  = cv2.resize(img_bgr, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(16,16),
                   cells_per_block=(2,2), feature_vector=True)
    color_feat = []
    for ch in range(3):
        hist = cv2.calcHist([img],[ch],None,[32],[0,256])
        color_feat.extend(hist.flatten())
    return np.concatenate([hog_feat, color_feat])


def predict_classification(img_bgr):
    clf   = joblib.load(os.path.join(MODELS_DIR, "task1_model.pkl"))
    feat  = extract_features(img_bgr)
    pred  = clf.predict([feat])[0]
    proba = clf.predict_proba([feat])[0]
    return {
        "label":      "Flooded" if pred==0 else "Non-Flooded",
        "confidence": round(float(proba[pred])*100, 2),
        "class_idx":  int(pred),
        "flooded":    bool(pred==0)
    }


def _sub_classify_patch(patch_bgr, binary_pred):
    """
    HSV-based heuristic to map a binary prediction (0/1) into one of the
    10 FloodNet semantic classes using the patch's colour profile.
    """
    p   = cv2.resize(patch_bgr, (32, 32)).astype(np.float32)
    hsv = cv2.cvtColor(p.astype(np.uint8), cv2.COLOR_BGR2HSV)

    avg_h = float(np.median(hsv[:, :, 0])) # OpenCV Hue is 0-179
    avg_s = float(np.mean(hsv[:, :, 1]))   # Saturation 0-255
    avg_v = float(np.mean(hsv[:, :, 2]))   # Brightness 0-255

    if binary_pred == 1:                       # ── FLOODED branch ──
        # Water: Blue tones or dark muddy water
        is_blue_water  = 85 < avg_h <= 140 and avg_s > 30
        is_muddy_water = avg_v < 120 and avg_s < 60
        if is_blue_water or is_muddy_water:
            return 5   # Water
        
        # Roads: Grey/concrete (bright to medium, low saturation)
        if 120 <= avg_v < 220 and avg_s < 55:
            return 3   # Road-Flooded
            
        # Buildings: All other flooded structures
        return 1       # Building-Flooded

    else:                                      # ── NON-FLOODED branch ──
        # Very dark shadows / Background
        if avg_v < 30:
            return 0   
            
        # Vegetation: Green (Hue 35-85)
        # Trees have shadow variance driving their average brightness down below 120. Grass is flat and bright.
        if 35 <= avg_h <= 85 and avg_s > 25:
            return 9 if avg_v > 120 else 6   # Grass vs Tree
            
        # Pool: Strong Blue (Hue 85-130)
        if 85 < avg_h <= 130 and avg_s > 60:
            return 8   # Pool
            
        # Vehicle: High saturation on a neutral background
        # Detect painted cars on driveways via sharp saturation spikes.
        # (Removed contrast logic to prevent sun glare on failed-RF water patches from registering as vehicles).
        if avg_s < 60:
            max_s = float(np.max(hsv[:, :, 1]))
            if max_s > 160:
                return 7   # Vehicle
            
        # Coloured Roofs: Red/Brown/Orange (Hue 0-35 or 150-179)
        if (avg_h < 35 or avg_h > 150) and avg_s > 30:
            return 2   # Building-NonFlood
            
        # Concrete Roads / Driveways / Sidewalks
        # Typically bright (V > 90) but highly desaturated (grey)
        if 90 <= avg_v < 220 and avg_s < 55:
            return 4   # Road-NonFlood
            
        # Catch-all for Buildings:
        # Dark asphalt shingles (V < 90) or extremely bright white roofs (V > 220)
        return 2       # Building-NonFlood


def predict_segmentation(img_bgr):
    import base64
    clf = joblib.load(os.path.join(MODELS_DIR, "task2_model.pkl"))
    img = cv2.resize(img_bgr, (256, 256))
    h, w = img.shape[:2]
    ph, pw = PATCH_SIZE

    patches, coords = [], []
    for r in range(0, h - ph, STEP):
        for c in range(0, w - pw, STEP):
            patches.append(img[r:r+ph, c:c+pw].flatten().astype(np.float32) / 255.0)
            coords.append((r, c))

    patch_arr    = np.array(patches)

    # ── predict_proba gives per-patch confidence ────────────────────────
    #    index 0 = non-flooded probability, index 1 = flooded probability
    probas       = clf.predict_proba(patch_arr)       # shape (N, 2)
    binary_preds = np.argmax(probas, axis=1)          # same result as predict()
    max_probas   = np.max(probas, axis=1)             # confidence per patch
    avg_conf     = float(np.mean(max_probas))         # mean image confidence (0-1)

    # ── initialise all 10 class counters at 0 ──────────────────────────
    class_counts = {name: 0 for name in CLASS_NAMES}

    overlay = img.copy()
    for (r, c), bp in zip(coords, binary_preds):
        cls  = _sub_classify_patch(img[r:r+ph, c:c+pw], bp)
        color = SEG_COLORS.get(cls, (128, 128, 128))
        overlay[r:r+ph, c:c+pw] = color
        class_counts[CLASS_NAMES[cls]] += 1

    blended = cv2.addWeighted(img, 0.4, overlay, 0.6, 0)
    
    # Restore mask aspect ratio to fix frontend misalignment / cropping overlaps
    orig_h, orig_w = img_bgr.shape[:2]
    scale = min(800 / orig_w, 800 / orig_h) if max(orig_w, orig_h) > 800 else 1.0
    target_w, target_h = int(orig_w * scale), int(orig_h * scale)
    blended_restored = cv2.resize(blended, (target_w, target_h))
    
    _, buf   = cv2.imencode(".png", blended_restored)

    total_patches = sum(class_counts.values())

    # ── per-class percentage of total patches ──────────────────────────
    class_pcts = {
        name: round(cnt / total_patches * 100, 1) if total_patches > 0 else 0.0
        for name, cnt in class_counts.items()
    }

    flooded_classes = {"Building-Flooded", "Road-Flooded", "Water"}
    flooded_pct = sum(class_pcts[n] for n in flooded_classes)
    safe_pct    = 100.0 - flooded_pct

    # ── Uncertainty logic — four independent signals ────────────────────
    #
    #  Signal 1 · avg_conf LOW
    #             mean patch confidence < 0.68 — model is generally unsure
    #
    #  Signal 2 · conf_std HIGH
    #             std of patch confidences > 0.13 — patches wildly disagree.
    #             Non-aerial images (charts, faces, screenshots) trigger this
    #             because the RF sees completely unfamiliar textures and
    #             oscillates between flooded/not-flooded per patch.
    #
    #  Signal 3 · ambiguous_mix
    #             flooded_pct is in the grey zone [20 %, 58 %] — cannot
    #             confidently call the scene flooded or safe.
    #
    #  Signal 4 · weak_not_flooded
    #             flooded_pct < 20 % but avg_conf < 0.75 — the model leans
    #             towards non-flooded yet is not certain enough. This catches
    #             non-aerial images that score low flood coverage simply
    #             because they have no water features, not because they are
    #             genuinely safe aerial scenes.
    #
    #  Any ONE signal → UNCERTAIN

    CONF_THRESHOLD    = 0.60   # relaxed
    CONF_STD_MAX      = 0.18   # relaxed to allow variance in safe images
    FLOOD_LOW         = 20.0   
    FLOOD_HIGH        = 58.0   
    CERTAIN_CONF      = 0.65   # relaxed

    conf_std          = float(np.std(max_probas))

    low_avg_conf      = avg_conf < CONF_THRESHOLD
    high_patch_spread = conf_std > CONF_STD_MAX
    ambiguous_mix     = FLOOD_LOW <= flooded_pct <= FLOOD_HIGH
    weak_not_flooded  = (flooded_pct < FLOOD_LOW and avg_conf < CERTAIN_CONF)

    clf_res = predict_classification(img_bgr)
    t1_flooded = clf_res["flooded"]
    t1_conf = clf_res["confidence"]

    # ── Out-Of-Distribution (OOD) / Graphic Check ─────────────────
    # Detects non-aerial graphics (like ECG lines, UI screenshots, or night photos) 
    # that fool the RF model with high confidence due to sheer uniformness.
    is_ood = False
    
    # 1. Pitch black dominance. Daylight aerial photography almost never exceeds 35% pure shadows.
    if class_pcts.get("Background", 0.0) > 35.0:
        is_ood = True
        
    # 2. Unnatural monolithic structures. 
    # If an image is >85% "Pool" or >85% "Building", it's almost certainly a solid blue 
    # or white/grey computer graphic (like a spreadsheet or solid background).
    if class_pcts.get("Building-Non-Flood", 0.0) > 85.0 or \
       class_pcts.get("Pool", 0.0) > 85.0 or \
       class_pcts.get("Vehicle", 0.0) > 50.0:
        is_ood = True

    if is_ood or high_patch_spread:
        verdict = "UNCERTAIN"
    elif t1_conf >= 80.0:
        verdict = "FLOODED" if t1_flooded else "NOT_FLOODED"
    else:
        if low_avg_conf or ambiguous_mix or weak_not_flooded:
            verdict = "UNCERTAIN"
        elif flooded_pct > FLOOD_HIGH:
            verdict = "FLOODED"
        else:
            verdict = "NOT_FLOODED"

    return {
        "mask_b64":      base64.b64encode(buf).decode("utf-8"),
        "class_dist":    class_counts,
        "class_pcts":    class_pcts,
        "total_patches": total_patches,
        "flooded_pct":   round(flooded_pct, 1),
        "safe_pct":      round(safe_pct, 1),
        "avg_conf":      round(avg_conf * 100, 1),
        "conf_std":      round(conf_std * 100, 1),
        "verdict":       verdict,
        "uncertain":     verdict == "UNCERTAIN",
    }


