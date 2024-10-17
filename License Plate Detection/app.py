from ultralytics import YOLOv10
from flask import Flask, request, jsonify
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, VisionEncoderDecoderConfig

app = Flask(__name__)
OCR_ckp_path = "LP-OCR-model/best" #Model weights are available at 
encoder_decoder_config = VisionEncoderDecoderConfig.from_pretrained(OCR_ckp_path)
OCR_model = VisionEncoderDecoderModel.from_pretrained(OCR_ckp_path, config=encoder_decoder_config)
YOLO_ckp_path = "yolov10_detect_big.pt" #Model weights are available at 
YOLO_model = YOLOv10(YOLO_ckp_path)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
OCR_model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400
    
    file = request.files['file']
    if not file.content_type.startswith('image/'):
        return jsonify({'error': 'File is not an image.'}), 400
    
    image = Image.open(file.stream)
    pixel_values = processor(image, return_tensors="pt").pixel_values
    yolo_results = YOLO_model(image) 

    if len(yolo_results) > 0:        
        bbox = yolo_results[0].boxes.xyxy[0]  
        xmin, ymin, xmax, ymax = map(int, bbox)
        cropped_image = image.crop((xmin, ymin, xmax, ymax))
        pixel_values = processor(cropped_image, return_tensors="pt").pixel_values

        with torch.no_grad():
            generated_ids = OCR_model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"Generated text: {generated_text}")
        return jsonify({'License Plate': generated_text})
    
    else:
        print("No license plate detected")
        return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
