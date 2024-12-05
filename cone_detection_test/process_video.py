import cv2
import numpy as np
from ultralytics import YOLO

def detect_color(bbox, frame):
    """
    Detect the dominant color within a bounding box region.
    :param bbox: Bounding box coordinates [x1, y1, x2, y2]
    :param frame: Original video frame
    :return: Detected color (e.g., 'orange', 'yellow', etc.)
    """
    x1, y1, x2, y2 = map(int, bbox)
    cropped_region = frame[y1:y2, x1:x2]

    if cropped_region.size == 0:  # Handle empty region
        return "unknown"

    # Convert the region to HSV for better color segmentation
    hsv_region = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2HSV)

    # Define color ranges
    color_ranges = {
        "orange": ((0, 100, 100), (15, 255, 255)),
        "yellow": ((20, 100, 100), (30, 255, 255)),
        "green": ((40, 50, 50), (80, 255, 255)),
    }

    max_pixels = 0
    detected_color = "unknown"

    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_region, np.array(lower), np.array(upper))
        pixel_count = cv2.countNonZero(mask)
        if pixel_count > max_pixels:
            max_pixels = pixel_count
            detected_color = color

    return detected_color

def process_video(input_video_path, output_video_path, model_path):
    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {input_video_path}")
    print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}, Total frames: {total_frames}")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to a higher resolution for better detection
        upscale_size = (1920, 1080)
        resized_frame = cv2.resize(frame, upscale_size, interpolation=cv2.INTER_LINEAR)

        # Perform inference on the resized frame
        results = model(resized_frame, conf=0.20, iou=0.45)

        # Annotate the resized frame with predictions
        annotated_resized_frame = resized_frame.copy()
        for result in results[0].boxes.data:
            x1, y1, x2, y2, confidence, class_id = result.tolist()
            label = model.names[int(class_id)]
            color = detect_color((x1, y1, x2, y2), resized_frame)

            # Format text with class, confidence, and color
            text = f"{label} ({color}) {confidence:.2f}"
            text_position = (int(x1), max(int(y1) - 10, 0))  # Position above the box

            # Draw bounding box and text
            cv2.rectangle(annotated_resized_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(annotated_resized_frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Downscale the annotated frame back to the original size for consistent output
        original_size_frame = cv2.resize(annotated_resized_frame, (frame.shape[1], frame.shape[0]))

        # Write the annotated frame to the output video
        out.write(original_size_frame)

        # Show the live preview
        cv2.imshow('YOLOv8 Cone Detection - Preview', original_size_frame)

        frame_count += 1
        print(f"Processed frame {frame_count}/{total_frames}", end="\r")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nProcessing interrupted by user.")
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("\nProcessing complete. Output saved to:", output_video_path)

# Paths
input_video = 'input/input_video.mp4'  # Path to input video
output_video = 'output/output_video.mp4'  # Path to save the final video
model_weights = 'best.pt'  # Path to the trained model

# Run the function
process_video(input_video, output_video, model_weights)
