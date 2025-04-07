from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.geo_utils import *
from datetime import datetime

app = Flask(__name__)
CORS(app)

@app.route("/analyze", methods=["POST"])
def analyze_area():
    data = request.get_json()
    coords = data.get("coordinates", [])
    print("Received coordinates:", coords)

    if not coords or not isinstance(coords, list):
        return jsonify({"error": "Invalid or missing coordinates"}), 400

    area_km2 = calculate_area(coords)
    print("Calculated area in square kilometers:", area_km2)
    image = mock_fetch_satellite_image(coords)
    cropped = crop_image_with_polygon(image, coords)
    cropped_b64 = encode_image_to_base64(cropped)

    return jsonify({
        "coordinates": coords,
        "area_square_miles": area_km2,
        "cropped_image_shape": list(cropped.shape),
        "cropped_image_base64": cropped_b64
    })

@app.route("/map-image", methods=["POST"])
def save_map_image():
    data = request.get_json()
    image_b64 = data.get("image_base64")

    if not image_b64 or not image_b64.startswith("data:image/png;base64,"):
        return jsonify({"error": "Invalid image data"}), 400

    try:
        image_data = base64.b64decode(image_b64.split(",")[1])
        filename = f"map_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        with open(filename, "wb") as f:
            f.write(image_data)
        return jsonify({"message": "Image saved", "filename": filename}), 200
    except Exception as e:
        print("Failed to save image:", e)
        return jsonify({"error": "Failed to save image"}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
