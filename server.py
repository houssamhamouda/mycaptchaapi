
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import pytesseract
import numpy as np

app = Flask(__name__)
CORS(app)

# Si nécessaire, configure le chemin vers tesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

@app.route('/', methods=['GET'])
def solve():
    img_b64 = request.args.get('b')
    target = request.args.get('n')
    if not img_b64 or not target:
        return jsonify({'status': 'error', 'reason': 'Missing image or target'})

    try:
        img_bytes = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        h, w = img.shape[:2]
        rows, cols = 3, 3
        box_h, box_w = h // rows, w // cols

        matches = []

        for i in range(rows):
            for j in range(cols):
                x1, y1 = j * box_w, i * box_h
                x2, y2 = x1 + box_w, y1 + box_h
                box = img[y1:y2, x1:x2]

                gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray, config='--psm 6 digits').strip()

                if target in text:
                    matches.append(i * 3 + j)  # index de l'image (0 à 8)

        return jsonify({'status': 'ok', 'matches': matches})
    except Exception as e:
        return jsonify({'status': 'error', 'reason': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
