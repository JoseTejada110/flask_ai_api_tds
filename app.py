# Importando librerías necesarias
from flask import Flask, request, jsonify
import numpy as np
import cv2
from ultralytics import YOLO
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Inicializando app de Flask
app = Flask(__name__)

# Inicializando limitador de peticiones por minuto
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["20/minute"]
)

# Definiendo endpoint
@app.route("/predecir_placa", methods=['POST'])
@limiter.limit("20/minute")
def readLicensePlate():
    try:
        
        # Validando imagen a predecir
        if 'image' not in request.files:
            return jsonify({'message': 'La imagen es obligatoria'}), 400
        
        # Obteniendo la imagen del body de la petición
        file = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        car_image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Haciendo la inferencia del primer modelo (detectando la placa del vehículo en la imagen)
        plate_results = plate_model.predict(source=car_image, conf=0.4, imgsz=640)
        license_plate = ''
        prediction_confidence = 0
        for r in plate_results:
            # Obteniendo las coordenas de la placa que predijo el modelo
            xB = int(r.boxes.xyxy[0][2])
            xA = int(r.boxes.xyxy[0][0])
            yB = int(r.boxes.xyxy[0][3])
            yA = int(r.boxes.xyxy[0][1])
            # Recortando imagen original para obtener únicamente la placa
            cropped_image = car_image[yA:yB, xA:xB]

            # Haciendo la inferencia sobre la placa detectada
            # por el primer modelo para poder leer los caracteres de la misma
            character_results = characters_model.predict(source=cropped_image, conf=0.3, imgsz=640)
            
            # Recorriendo los resultados del segundo modelo para organizar los caracteres de izquierda a derecha y luego concatenarlos
            for cr in character_results:
                # Ordenando los resultados horizontalmente de izquierda a derecha
                sorted_boxes = sorted(cr.boxes, key=lambda x: x.xyxy[0][0])
                prediction_confidence = str(np.mean(cr.boxes.conf.numpy())*100)
                for i in range(0, len(sorted_boxes)):
                    # Concatenando los resultados
                    license_plate+=characters_model.names[int(sorted_boxes[i].cls[0])]

        # Retornando la respuesta de la placa y la fiabilidad del resultado en formato JSON
        return jsonify({
            'license_plate': license_plate,
            'prediction_confidence': prediction_confidence
        }), 200
    except:
        return jsonify({
            'message': 'No ha sido posible escanear la placa.\n Esta función se encuentra en beta.',
        }), 500

if __name__ == "__main__":
    # Precargando los pesos de los modelos
    plate_model = YOLO("/Users/josetejada/Desktop/flask_api/models/plate_detection_weights.pt")
    characters_model = YOLO("/Users/josetejada/Desktop/flask_api/models/characters_detection_weights.pt")
    app.run(debug=True)
