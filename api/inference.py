import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io, base64, time
from ultralytics import YOLO

class OilSpillDetector():
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.class_names = ['rainbow', 'sheen', 'truecolor']
        self.class_colors = {
            'rainbow': (255, 0, 0),    # Red
            'sheen': (0, 255, 0),      # Green  
            'truecolor': (0, 0, 255)   # Blue
        }
        self.load_model()

    def load_model(self):
        """Load The Model"""
        try:
            self.model = YOLO(self.model_path)
            print(f"Model loaded from {self.model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None

    def is_loaded(self):
        """Check if model is loaded"""
        return self.model is not None

    def predict(self, image, conf_threshold: float = 0.15, return_image: bool = False):
        start_time = time.time()
        results = self.model.predict(
            image,
            conf=conf_threshold,
            verbose=False,
        )

        detections = []
        annotated_image_base64 = None

        if results[0].boxes is not None:
            if return_image:
                if isinstance(image, Image.Image):
                    annotated_image = image.copy()
                else:
                    # assume NumPy BGR array → convert to RGB manually
                    annotated_image = Image.fromarray(image[:, :, ::-1])

                draw = ImageDraw.Draw(annotated_image)
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()

            for box in results[0].boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

                box_area = (x2 - x1) * (y2 - y1)
                image_area = (
                    image.size[0] * image.size[1]
                    if hasattr(image, 'size')
                    else results[0].orig_shape[0] * results[0].orig_shape[1]
                )
                area_percentage = (box_area / float(image_area)) * 100

                class_name = self.class_names[class_id]
                detections.append({
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                    "area_percentage": round(float(area_percentage), 2),
                })

                if return_image:
                    color = self.class_colors.get(class_name, (255, 255, 0))
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    label = f"{class_name}: {confidence:.2f}"
                    bbox = draw.textbbox((x1, y1-25), label, font=font)
                    draw.rectangle(bbox, fill=color)
                    draw.text((x1, y1-25), label, fill=(255, 255, 255), font=font)

            if return_image:
                buffer = io.BytesIO()
                annotated_image.save(buffer, format='JPEG', quality=95)
                annotated_image_base64 = base64.b64encode(buffer.getvalue()).decode()

        result = {
            "detections": detections,
            "total_detections": len(detections),
            "processing_time": round(time.time() - start_time, 3),
        }
        if return_image:
            result["annotated_image"] = annotated_image_base64
        return result

