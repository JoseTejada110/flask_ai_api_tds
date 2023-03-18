from flask import Flask, request, jsonify
import numpy as np
import cv2
from ultralytics import YOLO
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import base64

app = Flask(__name__)
# limiter = Limiter(app,key_func=get_remote_address)
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["20/minute"]
)

@app.route("/predecir_placa", methods=['POST'])
@limiter.limit("20/minute")
def readLicensePlate():
    try:
        if 'image' not in request.files:
            return jsonify({'message': 'La imagen es obligatoria'}), 400
        file = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        car_image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        plate_results = plate_model.predict(source=car_image, conf=0.4, imgsz=640)
        license_plate = ''
        prediction_confidence = 0
        for r in plate_results:
            xB = int(r.boxes.xyxy[0][2])
            xA = int(r.boxes.xyxy[0][0])
            yB = int(r.boxes.xyxy[0][3])
            yA = int(r.boxes.xyxy[0][1])
            cropped_image = car_image[yA:yB, xA:xB]

            # Predict de los caracteres
            character_results = characters_model.predict(source=cropped_image, conf=0.3, imgsz=640)
            
            # Recorriendo los resultados para obtener el número de placa
            for cr in character_results:
                # Ordenando los resultados horizontalmente de izquierda a derecha
                sorted_boxes = sorted(cr.boxes, key=lambda x: x.xyxy[0][0])
                prediction_confidence = str(np.mean(cr.boxes.conf.numpy())*100)
                result_image = base64.b64encode(cr.plot(show_conf=False)).decode('utf-8')
                for i in range(0, len(sorted_boxes)):
                    license_plate+=characters_model.names[int(sorted_boxes[i].cls[0])]

        return jsonify({
            'license_plate': license_plate,
            'prediction_confidence': prediction_confidence,
            'result_image': result_image,
        }), 200
    except:
        return jsonify({
            'message': 'No ha sido posible escanear la placa.\n Esta función se encuentra en beta.',
        }), 500

if __name__ == "__main__":
    plate_model = YOLO("/Users/josetejada/Desktop/flask_api/models/plate_detection_weights.pt")
    characters_model = model = YOLO("/Users/josetejada/Desktop/flask_api/models/characters_detection_weights.pt")
    app.run(debug=True)