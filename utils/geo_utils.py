import numpy as np
import cv2
from shapely.geometry import Polygon
from shapely.ops import transform
from pyproj import Transformer
import base64
from io import BytesIO
from PIL import Image
import requests
from geopy.distance import geodesic
import math

def mock_fetch_satellite_image(coords):
    # Get center point of polygon
    lats, lons = zip(*coords)
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    # Get a static tile image (no API key needed)
    zoom = 16
    tile_url = f"https://static-maps.yandex.ru/1.x/?ll={center_lon},{center_lat}&z={zoom}&size=450,450&l=sat"

    try:
        response = requests.get(tile_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return np.array(image)
    except Exception as e:
        print("Failed to fetch satellite image:", e)
        # fallback dummy image
        return np.full((450, 450, 3), fill_value=(34, 139, 34), dtype=np.uint8)

def crop_image_with_polygon(image, coords):
    h, w, _ = image.shape
    lats, lons = zip(*coords)
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    scale_lat = lambda lat: int(h - h * ((lat - min_lat) / (max_lat - min_lat + 1e-6)))
    scale_lon = lambda lon: int(w * ((lon - min_lon) / (max_lon - min_lon + 1e-6)))

    pts = np.array([[scale_lon(lon), scale_lat(lat)] for lat, lon in coords], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    return cv2.bitwise_and(image, image, mask=mask)

def calculate_area_mi2(coords):
    from shapely.geometry import Polygon
    from shapely.ops import transform
    from pyproj import Transformer

    polygon = Polygon(coords)
    try:
        transformer = Transformer.from_crs("epsg:4326", "epsg:6933", always_xy=True)
        project = lambda x, y: transformer.transform(x, y)
        area_m2 = transform(project, polygon).area
        area_mi2 = area_m2 * 3.861e-7
    except Exception as e:
        print("Projection error:", e)
        area_mi2 = 0.0
    return round(area_mi2, 3)

def find_circumradius(sides, tol=1e-6, max_iter=1000):
    """
    Find the circumradius R for a cyclic polygon given its side lengths.
    We solve the equation:
      f(R) = sum(2*arcsin(s/(2R))) - 2*pi = 0
    using a binary search.
    """
    # Lower bound: R must be at least half the longest side
    R_low = max(s / 2 for s in sides)
    R_high = R_low * 2  # initial guess for high bound
    
    # Define the function f(R)
    def f(R):
        return sum(2 * math.asin(s / (2 * R)) for s in sides) - 2 * math.pi
    
    # Increase R_high until f(R_high) < 0 (i.e., sum of angles is less than 2*pi)
    while f(R_high) > 0:
        R_high *= 2

    # Binary search for R that gives f(R) close to zero
    for _ in range(max_iter):
        R_mid = (R_low + R_high) / 2
        val = f(R_mid)
        if abs(val) < tol:
            return R_mid
        if val > 0:
            # Sum of angles is too high, increase R
            R_low = R_mid
        else:
            # Sum of angles is too low, decrease R
            R_high = R_mid
    return R_mid

def cyclic_polygon_area(sides):
    """
    Compute the area of a cyclic polygon (vertices on a circle) given its side lengths.
    The area is computed by triangulating the polygon from the center.
    """
    R = find_circumradius(sides)
    area = 0.0
    for s in sides:
        # Compute the central angle for side s
        theta = 2 * math.asin(s / (2 * R))
        # Area of the triangle with central angle theta and radius R
        area += 0.5 * R**2 * math.sin(theta)
    return area

def calculate_area(points):
    
    distances = []
    for i in range(len(points) - 1):
        d = geodesic(points[i], points[i+1]).miles
        distances.append(d)

    print("Distances between consecutive points:")
    for idx, d in enumerate(distances, start=1):
        print(f"Segment {idx}: {d:.2f} miles")

    area = cyclic_polygon_area(distances)

    return round(area, 5)

def encode_image_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')
