let map;
let polygonCoords = null;

window.onload = function () {
  map = L.map("map").setView([41.8349, -87.6270], 16);

  L.tileLayer("https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", {
    subdomains: ["mt0", "mt1", "mt2", "mt3"],
    attribution: "&copy; Google Maps (Satellite)",
  }).addTo(map);

  L.Control.geocoder().addTo(map);

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
    edit: {
      featureGroup: drawnItems,
    },
  });
  map.addControl(drawControl);

  map.on(L.Draw.Event.CREATED, function (event) {
    drawnItems.clearLayers();
    drawnItems.addLayer(event.layer);
    polygonCoords = event.layer.getLatLngs()[0].map(latlng => [latlng.lat, latlng.lng]);
  });

  // Function to handle the snapshot button
  document.getElementById("snapshotBtn").onclick = function () {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    const img = new Image();
    img.src = mapImage; // Assuming mapImage is the base64 image of the map
    img.onload = function () {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      const dataURL = canvas.toDataURL("image/png");
      const link = document.createElement("a");
      link.href = dataURL;
      link.download = "map_snapshot.png";
      link.click();
    };
  };

  document.getElementById("analyzeBtn").onclick = async function () {
    if (!polygonCoords) return alert("Please draw a polygon first.");

    const res = await fetch("http://127.0.0.1:8000/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ coordinates: polygonCoords }),
    });

    const data = await res.json();
    const resultDiv = document.getElementById("result");
    resultDiv.innerHTML = `
      <h3>Results:</h3>
      <p>Area (Square Miles): ${data.area_square_miles || data.area_km2}</p>
      <p>Cropped Image Shape: ${data.cropped_image_shape.join(" × ")}</p>
      <img class="result" src="data:image/png;base64,${data.cropped_image_base64}" alt="Cropped Image"/>
    `;
  };
};
