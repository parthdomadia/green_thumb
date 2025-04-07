import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet-draw/dist/leaflet.draw.css";
import "leaflet-draw";
import "leaflet-control-geocoder/dist/Control.Geocoder.css";
import "leaflet-control-geocoder";

export default function MapPolygonTool() {
  const mapRef = useRef(null);
  const [polygonCoords, setPolygonCoords] = useState(null);
  const [response, setResponse] = useState(null);
  const [croppedImg, setCroppedImg] = useState(null);
  const [mapImage] = useState(null);

  useEffect(() => {
    const initializeMap = () => {
      const mapContainer = document.getElementById("map");
      if (!mapContainer) return;

      const map = L.map(mapContainer).setView([41.8349, -87.6270], 16);
      mapRef.current = map;

      L.tileLayer("https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", {
        subdomains: ["mt0", "mt1", "mt2", "mt3"],
        attribution: "&copy; Google Maps (Satellite)",
      }).addTo(map);

      L.Control.geocoder({ defaultMarkGeocode: true }).addTo(map);

      const drawnItems = new L.FeatureGroup();
      map.addLayer(drawnItems);

      const drawControl = new L.Control.Draw({
        draw: {
          polygon: true,
          marker: false,
          polyline: false,
          rectangle: false,
          circle: false,
          circlemarker: false,
        },
        edit: { featureGroup: drawnItems },
      });
      map.addControl(drawControl);

      map.on(L.Draw.Event.CREATED, function (event) {
        const layer = event.layer;
        drawnItems.clearLayers();
        drawnItems.addLayer(layer);

        const latlngs = layer.getLatLngs()[0].map((latlng) => [latlng.lat, latlng.lng]);
        setPolygonCoords(latlngs);
      });
    };

    setTimeout(initializeMap, 0);

    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
      }
    };
  }, []);

  const handleAnalyze = async () => {
    if (!polygonCoords) return alert("Please draw a polygon first.");

    const res = await fetch("http://127.0.0.1:8000/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ coordinates: polygonCoords }),
    });
    const data = await res.json();
    setResponse(data);
    if (data.cropped_image_base64) {
      setCroppedImg(`data:image/png;base64,${data.cropped_image_base64}`);
    }
  };


return (
    <div style={{ paddingLeft: "1rem" }}>
        <h1>Deforestation Analyzer</h1>

        <div style={{ width: "600px", height: "400px", border: "1px solid #ccc", overflow: "hidden", position: "relative" }}>
            <div id="map" style={{ height: "100%", width: "100%" }}></div>
        </div>

        <button onClick={handleAnalyze} style={{ marginTop: "1rem", marginRight: "1rem" }}>
            Analyze Vegetation
        </button>

        {response && (
            <div style={{ marginTop: "1rem" }}>
                <h3>Results:</h3>
                <p>Area (Sqaure Miles): {response.area_square_miles}</p>
                <p>Cropped Image Shape: {response.cropped_image_shape.join(" × ")}</p>

                {croppedImg && (
                    <div>
                        <p>Cropped Image:</p>
                        <img src={croppedImg} alt="Cropped" style={{ maxWidth: "100%", border: "1px solid #ccc" }} />
                    </div>
                )}
            </div>
        )}

        {mapImage && (
            <div style={{ marginTop: "1rem" }}>
                <h3>Map Preview:</h3>
                <img src={mapImage} alt="Map Preview" style={{ maxWidth: "100%", border: "1px solid #ccc" }} />
            </div>
        )}
    </div>
);
}
