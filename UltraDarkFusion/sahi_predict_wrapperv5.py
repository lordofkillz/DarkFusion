import logging

from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image
import cv2
import os
import random
from PIL import UnidentifiedImageError
from sahi import AutoDetectionModel

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SahiPredictWrapper:
    def __init__(self, model_type, model_path, confidence_threshold, device):
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type=model_type,
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device
        )
    @staticmethod
    def get_unique_color(class_name):
        return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

    def read_class_names(self, file_path):
        with open(file_path, 'r') as file:
            class_names = [line.strip() for line in file.readlines()]
        return class_names

    def process_image(self, image_path, slice_height, slice_width, overlap_height_ratio, overlap_width_ratio, desired_classes):
        try:
            image_np = read_image(image_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            result = get_sliced_prediction(
                image=image_np,
                detection_model=self.detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio
            )

            txt_file_path = os.path.splitext(image_path)[0] + '.txt'
            with open(txt_file_path, 'w') as f:
                for obj in result.object_prediction_list:
                    if obj.category.name in desired_classes:
                        bbox = obj.bbox.to_voc_bbox()
                        color = SahiPredictWrapper.get_unique_color(obj.category.name)

                        cv2.rectangle(image_np, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

                        label = f"{obj.category.name}"
                        cv2.putText(image_np, label, (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                        xc = (bbox[0] + bbox[2]) / 2 / image_np.shape[1]
                        yc = (bbox[1] + bbox[3]) / 2 / image_np.shape[0]
                        w = (bbox[2] - bbox[0]) / image_np.shape[1]
                        h = (bbox[3] - bbox[1]) / image_np.shape[0]

                        f.write(f"{obj.category.id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            # Display the image with a delay of 1ms before moving to the next one
            cv2.imshow("Predictions", image_np)
            cv2.waitKey(1)

        except UnidentifiedImageError:
            logger.info(f"Cannot open image: {image_path}")
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")

    def process_folder(self, folder_path, class_names_file, slice_height, slice_width, overlap_height_ratio, overlap_width_ratio):
        desired_classes = self.read_class_names(class_names_file)
        allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        if not os.path.isdir(folder_path):
            logger.info(f"Image directory not found: {folder_path}")
            return

        for image_file in filter(lambda file: any(file.lower().endswith(ext) for ext in allowed_extensions), os.listdir(folder_path)):
            image_path = os.path.join(folder_path, image_file)
            self.process_image(image_path, slice_height, slice_width, overlap_height_ratio, overlap_width_ratio, desired_classes)
            # Add a small delay between processing of images to give the system time to display the window
            cv2.waitKey(1)  # You can increase this delay if needed

        cv2.destroyAllWindows()
