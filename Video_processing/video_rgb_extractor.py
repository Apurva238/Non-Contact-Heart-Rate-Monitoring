import cv2
import csv
import os
import json


# ==========================================================
# CONFIGURATION
# ==========================================================

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = PROJECT_ROOT / "Video_processing" / "data" / "subject1" / "vid.mp4"
OUTPUT_DIR = PROJECT_ROOT / "Video_processing" / "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Minimum skin pixels required for a region to be considered valid
MIN_SKIN_PIXELS = 100

# Face tracking configuration
IOU_MATCH_THRESHOLD = 0.2
MAX_MISSING_FRAMES = 5

REGION_NAMES = [
    "forehead",
    "left_cheek",
    "right_cheek"
]

CHANNEL_NAMES = [
    "red",
    "green",
    "blue"
]


# ==========================================================
# FACE DETECTOR
# ==========================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError(
        "Error: Could not load the face detection model."
    )


# ==========================================================
# SKIN FILTER CONFIGURATION
# ==========================================================

# Basic HSV threshold for likely skin pixels.
# This is a heuristic and may require validation.

LOWER_SKIN = (0, 20, 40)
UPPER_SKIN = (25, 255, 255)


# ==========================================================
# FACE TRACKING HELPER
# ==========================================================

def calculate_iou(box_a, box_b):
    """
    Calculate Intersection over Union (IoU) between two
    bounding boxes.

    Each box is represented as:
        (x, y, width, height)

    IoU ranges from 0 to 1:
        0 = no overlap
        1 = complete overlap
    """

    ax, ay, aw, ah = map(int, box_a)
    bx, by, bw, bh = map(int, box_b)

    x_left = max(ax, bx)
    y_top = max(ay, by)

    x_right = min(ax + aw, bx + bw)
    y_bottom = min(ay + ah, by + bh)

    intersection_width = max(0, x_right - x_left)
    intersection_height = max(0, y_bottom - y_top)

    intersection_area = (
        intersection_width * intersection_height
    )

    area_a = aw * ah
    area_b = bw * bh

    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def select_face(faces, previous_face):
    """
    Select a face using the previous frame's bounding box.

    If a previous face exists, choose the detection with
    the highest IoU.

    If the overlap is insufficient, reinitialize by
    selecting the largest detected face.

    Returns:
        selected_face
        was_tracked
        best_iou
    """

    if previous_face is None:
        # Initialize tracking with the largest detected face.
        selected_face = max(
            faces,
            key=lambda face: face[2] * face[3]
        )

        return selected_face, False, 0.0

    # Calculate overlap with every detected face.
    iou_scores = [
        calculate_iou(previous_face, face)
        for face in faces
    ]

    best_index = max(
        range(len(faces)),
        key=lambda index: iou_scores[index]
    )

    best_iou = iou_scores[best_index]

    if best_iou >= IOU_MATCH_THRESHOLD:
        # Match the detection to the previous face.
        selected_face = faces[best_index]

        return selected_face, True, best_iou

    # Tracking match failed. Reinitialize using the
    # largest detected face.
    selected_face = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    return selected_face, False, best_iou


# ==========================================================
# LOAD VIDEO
# ==========================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Error: Could not open video: {VIDEO_PATH}"
    )

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

if fps <= 0:
    cap.release()
    raise RuntimeError(
        "Error: Invalid video FPS."
    )

print("Video loaded successfully!")
print(f"FPS: {fps}")
print(f"Total frames: {total_frames}")


# ==========================================================
# INITIALIZE EXTRACTION AND TRACKING
# ==========================================================

results = []

frame_number = 0

# Bounding box of the face selected in the previous
# successfully detected frame.
previous_face = None

# Number of consecutive frames without a detection.
frames_since_detection = 0

# Helps distinguish the initial face selection from
# subsequent reinitializations.
ever_detected_face = False


# ==========================================================
# MAIN VIDEO PROCESSING LOOP
# ==========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    timestamp = frame_number / fps

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect on a smaller frame for faster processing, then map
    # bounding boxes back to the original frame coordinates.
    detection_width = 640
    scale = detection_width / frame.shape[1]
    detection_height = max(1, int(frame.shape[0] * scale))
    small_gray = cv2.resize(gray, (detection_width, detection_height))

    # ------------------------------------------------------
    # STEP 1: DETECT FACES
    # ------------------------------------------------------

    # Run the detector periodically; reuse the last box between
    # detection frames to keep high-resolution videos responsive.
    if frame_number % 5 == 0 or previous_face is None:
        small_faces = face_cascade.detectMultiScale(
            small_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )
        faces = [
            (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
            for (x, y, w, h) in small_faces
        ]
    else:
        faces = [previous_face]

    # Initialize the current frame's output row.
    row = {
        "timestamp_s": timestamp,
        "valid_face": 0,
        "face_tracked": 0,
        "tracking_status": "not_detected"
    }

    # ------------------------------------------------------
    # STEP 2: SELECT AND TRACK THE FACE
    # ------------------------------------------------------

    if len(faces) > 0:

        selected_face, was_tracked, best_iou = select_face(
            faces,
            previous_face
        )

        x, y, w, h = map(
            int,
            selected_face
        )

        # Update the bounding box for the next frame.
        previous_face = (x, y, w, h)

        frames_since_detection = 0
        ever_detected_face = True

        row["valid_face"] = 1
        row["face_tracked"] = int(was_tracked)

        if was_tracked:
            row["tracking_status"] = "tracked"

        elif previous_face is not None and frame_number == 0:
            # Included for completeness; the first frame
            # normally has no previous face.
            row["tracking_status"] = "initialized"

        elif row["valid_face"] == 1 and not was_tracked:

            # First detection or a new tracking initialization.
            # Distinguish these using the face history.
            # The first actual detection is identified below.
            if frame_number == 0:
                row["tracking_status"] = "initialized"
            else:
                row["tracking_status"] = "reinitialized"

        # --------------------------------------------------
        # STEP 3: DEFINE FACIAL REGIONS
        # --------------------------------------------------

        regions = {
            "forehead": (
                x + int(0.25 * w),
                y + int(0.10 * h),
                int(0.50 * w),
                int(0.15 * h)
            ),

            "left_cheek": (
                x + int(0.10 * w),
                y + int(0.50 * h),
                int(0.30 * w),
                int(0.25 * h)
            ),

            "right_cheek": (
                x + int(0.60 * w),
                y + int(0.50 * h),
                int(0.30 * w),
                int(0.25 * h)
            )
        }

        frame_height, frame_width = frame.shape[:2]

        # --------------------------------------------------
        # STEP 4: EXTRACT RGB AND CHECK REGION VALIDITY
        # --------------------------------------------------

        for region_name, (rx, ry, rw, rh) in regions.items():

            # Keep ROI coordinates inside the frame.
            x1 = max(0, rx)
            y1 = max(0, ry)

            x2 = min(
                frame_width,
                rx + rw
            )

            y2 = min(
                frame_height,
                ry + rh
            )

            roi = frame[y1:y2, x1:x2]

            # Default values for an invalid ROI.
            skin_pixel_count = 0
            region_valid = False

            row[f"{region_name}_red"] = ""
            row[f"{region_name}_green"] = ""
            row[f"{region_name}_blue"] = ""

            if roi.size > 0:

                # Convert the ROI from BGR to HSV.
                hsv = cv2.cvtColor(
                    roi,
                    cv2.COLOR_BGR2HSV
                )

                # Identify likely skin pixels.
                skin_mask = cv2.inRange(
                    hsv,
                    LOWER_SKIN,
                    UPPER_SKIN
                )

                # Select pixels passing the skin filter.
                skin_pixels = roi[skin_mask > 0]

                skin_pixel_count = int(
                    (skin_mask > 0).sum()
                )

                # Determine whether the region is usable.
                region_valid = (
                    skin_pixel_count >= MIN_SKIN_PIXELS
                )

                if len(skin_pixels) > 0:

                    # OpenCV stores color channels as B, G, R.
                    blue, green, red = skin_pixels.mean(
                        axis=0
                    )

                    row[f"{region_name}_red"] = float(red)
                    row[f"{region_name}_green"] = float(green)
                    row[f"{region_name}_blue"] = float(blue)

            # Save the region quality information.
            row[
                f"{region_name}_skin_pixel_count"
            ] = skin_pixel_count

            row[
                f"{region_name}_valid"
            ] = region_valid

    else:

        # --------------------------------------------------
        # NO FACE DETECTED IN THIS FRAME
        # --------------------------------------------------

        frames_since_detection += 1

        # Retain the previous bounding box briefly, allowing
        # tracking to resume if the face reappears.
        if frames_since_detection > MAX_MISSING_FRAMES:

            previous_face = None
            frames_since_detection = 0

        # Mark every ROI as unavailable for this frame.
        for region_name in REGION_NAMES:

            row[f"{region_name}_red"] = ""
            row[f"{region_name}_green"] = ""
            row[f"{region_name}_blue"] = ""

            row[
                f"{region_name}_skin_pixel_count"
            ] = 0

            row[
                f"{region_name}_valid"
            ] = False

    # ------------------------------------------------------
    # STEP 5: STORE RESULTS
    # ------------------------------------------------------

    results.append(row)

    frame_number += 1

    if frame_number % 300 == 0:

        print(
            f"Processed {frame_number}/{total_frames} frames"
        )


# ==========================================================
# RELEASE VIDEO
# ==========================================================

cap.release()


# ==========================================================
# SAVE RGB SIGNALS TO CSV
# ==========================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "rgb_signals.csv"
)

fieldnames = [
    "timestamp_s",
    "valid_face",
    "face_tracked",
    "tracking_status"
]

# Add RGB, skin-pixel count, and validity columns
# for each region.
for region_name in REGION_NAMES:

    for channel in CHANNEL_NAMES:
        fieldnames.append(
            f"{region_name}_{channel}"
        )

    fieldnames.append(
        f"{region_name}_skin_pixel_count"
    )

    fieldnames.append(
        f"{region_name}_valid"
    )


with open(
    output_path,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)


# ==========================================================
# CALCULATE SUMMARY STATISTICS
# ==========================================================

frames_with_face = sum(
    row["valid_face"]
    for row in results
)

frames_without_face = sum(
    1
    for row in results
    if row["valid_face"] == 0
)

tracked_frames = sum(
    row["face_tracked"]
    for row in results
)

tracking_status_counts = {}

for row in results:

    status = row["tracking_status"]

    tracking_status_counts[status] = (
        tracking_status_counts.get(status, 0) + 1
    )

region_validity_summary = {}

for region_name in REGION_NAMES:

    valid_count = sum(
        1
        for row in results
        if row.get(f"{region_name}_valid", False)
    )

    region_validity_summary[region_name] = {
        "valid_frames": valid_count,
        "invalid_frames": frame_number - valid_count
    }


# ==========================================================
# SAVE METADATA
# ==========================================================

metadata = {
    "video_path": str(VIDEO_PATH),
    "fps": fps,
    "total_frames": total_frames,
    "processed_frames": frame_number,
    "duration_seconds": frame_number / fps,

    "frames_with_face": frames_with_face,
    "frames_without_face": frames_without_face,

    "tracking": {
        "method": "Haar face detection with IoU association; detector refreshed every 5 frames and last box reused between refreshes",
        "detection_interval_frames": 5,
        "detection_width_pixels": 640,
        "iou_match_threshold": IOU_MATCH_THRESHOLD,
        "max_missing_frames": MAX_MISSING_FRAMES,
        "frames_tracked": tracked_frames,
        "tracking_status_counts": tracking_status_counts
    },

    "regions": REGION_NAMES,
    "channels": CHANNEL_NAMES,

    "skin_filter": {
        "method": "HSV threshold",
        "lower": list(LOWER_SKIN),
        "upper": list(UPPER_SKIN),
        "minimum_skin_pixels": MIN_SKIN_PIXELS
    },

    "region_validity_summary": region_validity_summary
}

metadata_path = os.path.join(
    OUTPUT_DIR,
    "metadata.json"
)

with open(
    metadata_path,
    "w"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\nRGB extraction and tracking complete!")

print(f"Total frames processed: {frame_number}")
print(f"Frames with detected faces: {frames_with_face}")
print(f"Frames without detected faces: {frames_without_face}")
print(f"Frames matched to the previous face: {tracked_frames}")

print("\nTracking status counts:")

for status, count in tracking_status_counts.items():
    print(f"  {status}: {count}")

print("\nRegion validity:")

for region_name, summary in region_validity_summary.items():

    print(
        f"  {region_name}: "
        f"{summary['valid_frames']} valid, "
        f"{summary['invalid_frames']} invalid"
    )

print(f"\nRGB signals saved to: {output_path}")
print(f"Metadata saved to: {metadata_path}")