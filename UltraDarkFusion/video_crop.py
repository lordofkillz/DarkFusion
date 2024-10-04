import os
import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from collections import defaultdict
import pafy
import logging

# Set up logging instead of using print statements
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def create_kalman_filter():
    """Initialize a Kalman Filter for tracking."""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.R *= 10  # Measurement uncertainty
    kf.P *= 1000  # Initial covariance matrix
    kf.Q = np.eye(4)  # Process uncertainty
    return kf


def update_kalman_filter(kf, measurement):
    """Update Kalman filter with a new measurement."""
    kf.predict()
    kf.update(measurement)
    return kf.x


def iou(box1, box2):
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    return inter_area / float(box1_area + box2_area - inter_area)


def get_video_capture(source):
    """Retrieve video capture from a given source."""
    if isinstance(source, int) or source.isdigit():
        return cv2.VideoCapture(int(source))
    elif "http" in source and "youtube" in source:
        try:
            pafy.set_backend("yt-dlp")
            video = pafy.new(source)
            best = video.getbest(preftype="mp4")
            logging.info(f"Opening YouTube video: {source}")
            return cv2.VideoCapture(best.url)
        except Exception as e:
            logging.error(f"Error opening YouTube video: {e}")
            return None
    else:
        return cv2.VideoCapture(source)


def main(args):
    # Initialize YOLO model with user-provided weights
    model = YOLO(args.weights)
    
    # Initialize ORB detector and BFMatcher
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Load the custom class names from the classes.txt file
    allowed_classes = []
    with open(args.classes, 'r') as file:
        allowed_classes = [line.strip() for line in file.readlines()]

    # Get video capture
    cap = get_video_capture(args.video)
    if cap is None or not cap.isOpened():
        raise ValueError("Error: Unable to open video source")

    # Get video properties
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Prepare the save directory
    crop_dir_name = args.save_location
    if not os.path.exists(crop_dir_name):
        try:
            os.mkdir(crop_dir_name)
        except OSError as e:
            logging.error(f"Error creating directory: {e}")
            return

    idx = 0
    kalman_filters = defaultdict(create_kalman_filter)
    previous_descriptors = {}

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            logging.info("Video frame is empty or video processing has been successfully completed.")
            break

        # Perform detection and tracking using YOLO's track mode
        results = model.track(source=im0, persist=True, imgsz=args.imgsz, half=args.half, show=args.show, stream=True, tracker='bytetrack.yaml')

        # Iterate over the generator results
        for result in results:
            boxes = result.boxes.xyxy.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            track_ids = result.boxes.id.cpu().tolist() if result.boxes.id is not None else [None] * len(boxes)

            current_descriptors = {}

            for box, cls, track_id in zip(boxes, classes, track_ids):
                label = model.names[int(cls)]  # YOLO's internal names
                if label in allowed_classes:  # Filter detections by allowed classes
                    idx += 1
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2
                    start_x = int(center_x - args.crop_width / 2)
                    start_y = int(center_y - args.crop_height / 2)
                    start_x = max(0, min(start_x, w - args.crop_width))
                    start_y = max(0, min(start_y, h - args.crop_height))
                    end_x = min(w, start_x + args.crop_width)
                    end_y = min(h, start_y + args.crop_height)

                    # Crop the object
                    crop_obj = im0[start_y:end_y, start_x:end_x]
                    filename = f"{idx}_{label}_{track_id}.png" if track_id is not None else f"{idx}_{label}.png"
                    cv2.imwrite(os.path.join(crop_dir_name, filename), crop_obj)

                    # Update Kalman Filter for the object
                    kf = kalman_filters[track_id if track_id is not None else idx]
                    measurement = np.array([center_x, center_y])
                    state = update_kalman_filter(kf, measurement)

                    # Extract ORB keypoints and descriptors
                    keypoints, descriptors = orb.detectAndCompute(crop_obj, None)
                    if descriptors is not None:
                        current_descriptors[track_id] = descriptors

                        # Match with previous descriptors
                        if previous_descriptors and track_id in previous_descriptors:
                            matches = bf.match(previous_descriptors[track_id], descriptors)
                            matches = sorted(matches, key=lambda x: x.distance)
                            logging.info(f"Found {len(matches)} matches for class {label} with track ID {track_id}")

            # Update previous descriptors for the next frame (keep history)
            previous_descriptors.update(current_descriptors)

            # Save labels
            label_filename = f"{idx}_{label}_{track_id}.txt" if track_id is not None else f"{idx}_{label}.txt"
            with open(os.path.join(crop_dir_name, label_filename), 'w') as file:
                for inner_box, inner_cls in zip(boxes, classes):
                    if (inner_box[0] >= start_x and inner_box[2] <= end_x and
                        inner_box[1] >= start_y and inner_box[3] <= end_y):
                        inner_label = model.names[int(inner_cls)]
                        if inner_label in allowed_classes:
                            bbox_width = (inner_box[2] - inner_box[0]) / args.crop_width
                            bbox_height = (inner_box[3] - inner_box[1]) / args.crop_height
                            bbox_x_center = ((inner_box[0] + inner_box[2]) / 2 - start_x) / args.crop_width
                            bbox_y_center = ((inner_box[1] + inner_box[3]) / 2 - start_y) / args.crop_height
                            file.write(f"{int(inner_cls)} {bbox_x_center:.6f} {bbox_y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Crop objects from video using YOLO model with integrated tracking and Kalman filtering.")
    parser.add_argument("--weights", type=str, required=True, help="Path to the model weights file (.engine, .onnx, .pt)")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file or YouTube URL")
    parser.add_argument("--classes", type=str, required=True, help="Path to the classes.txt file with allowed class names")
    parser.add_argument("--crop_width", type=int, required=True, help="Width of the crop area")
    parser.add_argument("--crop_height", type=int, required=True, help="Height of the crop area")
    parser.add_argument("--save_location", type=str, required=True, help="Directory to save the cropped images")
    parser.add_argument("--imgsz", type=str, default="(640,640)", help="Inference size (height, width)")
    parser.add_argument("--show", action='store_true', help="Show predictions")
    parser.add_argument("--half", action='store_true', help="Use half precision (only supported on CUDA)")

    args = parser.parse_args()
    args.imgsz = tuple(map(int, args.imgsz.strip('()').split(',')))  # Convert imgsz to tuple
    main(args)

