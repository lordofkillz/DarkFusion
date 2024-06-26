import argparse
import os
import cv2
import pafy
from ultralytics import YOLO

def get_video_capture(source):
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    elif "http" in source and "youtube" in source:
        try:
            pafy.set_backend("yt-dlp")  # Ensure we are using yt-dlp as the backend
            video = pafy.new(source)
            best = video.getbest(preftype="mp4")
            print(f"Opening YouTube video: {source}")
            return cv2.VideoCapture(best.url)
        except Exception as e:
            print(f"Error opening YouTube video: {e}")
            return None
    else:
        return cv2.VideoCapture(source)

def main(args):
    model = YOLO(args.weights)
    names = model.names

    allowed_classes = []
    with open(args.classes, 'r') as file:
        allowed_classes = [line.strip() for line in file.readlines()]

    cap = get_video_capture(args.video)
    if cap is None or not cap.isOpened():
        raise AssertionError("Error opening video source")

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    crop_dir_name = args.save_location
    if not os.path.exists(crop_dir_name):
        os.mkdir(crop_dir_name)

    idx = 0
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        results = model.predict(im0, imgsz=args.imgsz, half=args.half, show=args.show)
        boxes = results[0].boxes.xyxy.cpu().tolist()
        classes = results[0].boxes.cls.cpu().tolist()

        for box, cls in zip(boxes, classes):
            label = names[int(cls)]
            if label in allowed_classes:
                idx += 1
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                start_x = max(0, int(center_x - args.crop_width / 2))
                start_y = max(0, int(center_y - args.crop_height / 2))
                end_x = min(w, start_x + args.crop_width)
                end_y = min(h, start_y + args.crop_height)
                start_x = max(0, end_x - args.crop_width)
                start_y = max(0, end_y - args.crop_height)

                crop_obj = im0[start_y:end_y, start_x:end_x]
                filename = f"{idx}_{label}.png"
                cv2.imwrite(os.path.join(crop_dir_name, filename), crop_obj)

                with open(os.path.join(crop_dir_name, f"{idx}_{label}.txt"), 'w') as file:
                    for inner_box, inner_cls in zip(boxes, classes):
                        if (inner_box[0] >= start_x and inner_box[2] <= end_x and
                            inner_box[1] >= start_y and inner_box[3] <= end_y):
                            inner_label = names[int(inner_cls)]
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
    parser = argparse.ArgumentParser(description="Crop objects from video using YOLO model.")
    parser.add_argument("--weights", type=str, required=True, help="Path to the model weights file (.engine, .onnx, .pt)")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file or YouTube URL")
    parser.add_argument("--classes", type=str, required=True, help="Path to the classes.txt file")
    parser.add_argument("--crop_width", type=int, required=True, help="Width of the crop area")
    parser.add_argument("--crop_height", type=int, required=True, help="Height of the crop area")
    parser.add_argument("--save_location", type=str, required=True, help="Directory to save the cropped images")
    parser.add_argument("--imgsz", type=str, default="(640,640)", help="Inference size (height, width)")
    parser.add_argument("--show", action='store_true', help="Show predictions")
    parser.add_argument("--half", action='store_true', help="Use half precision")

    args = parser.parse_args()
    args.imgsz = tuple(map(int, args.imgsz.strip('()').split(',')))  # Convert imgsz to tuple
    main(args)

