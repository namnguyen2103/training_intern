from flask import Flask, request, jsonify
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

app = Flask(__name__)
ckp_path = #Model weights are available at https://drive.google.com/file/d/1Ku2jl6ID9d20eNyBXTCbrRYF2Ekcz2IE
model = VisionEncoderDecoderModel.from_pretrained("OCR-model/best")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided.'}), 400
    
    file = request.files['file']
    if not file.content_type.startswith('image/'):
        return jsonify({'error': 'File is not an image.'}), 400
    
    image = Image.open(file.stream)
    pixel_values = processor(image, return_tensors="pt").pixel_values
    
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return jsonify({'captcha': generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
