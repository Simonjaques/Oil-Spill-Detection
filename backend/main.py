import os
import io
import math
import base64
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import tensorflow as tf

from PIL import Image, ImageDraw, ImageFilter
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import requests
from dotenv import load_dotenv

# Load credentials from .env file
load_dotenv()

# Store in global variables
AIS_API_KEY = os.getenv("AIS_API_KEY")
COPERNICUS_ID = os.getenv("COPERNICUS_ID")
COPERNICUS_SECRET = os.getenv("COPERNICUS_SECRET")

# Print verification status (returns True if loaded, False if missing)
print(f"AIS_API_KEY loaded: {bool(AIS_API_KEY)}")
print(f"COPERNICUS_ID loaded: {bool(COPERNICUS_ID)}")
print(f"COPERNICUS_SECRET loaded: {bool(COPERNICUS_SECRET)}")

TRAIL_CACHE = {}



# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "backend/oil_spill_mobilenetv2.keras"

CHIP_SIZE = 400
STRIDE = 200
THRESHOLD = 0.50

CDSE_CLIENT_ID = os.getenv("CDSE_CLIENT_ID", "").strip()
CDSE_CLIENT_SECRET = os.getenv("CDSE_CLIENT_SECRET", "").strip()

AIS_API_KEY = os.getenv("AIS_API_KEY", "").strip()

REQUEST_TIMEOUT = 60


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="OceanSight AI",
    description="Satellite-based oil spill detection and environmental intelligence",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODEL
# ============================================================

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    MODEL_LOADED = True
    print("✅ OceanSight MobileNetV2 model loaded.")
except Exception as error:
    model = None
    MODEL_LOADED = False
    print("❌ Could not load model:", error)


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "OceanSight AI backend is running",
        "model_loaded": MODEL_LOADED
    }


@app.get("/model-status")
def model_status():
    return {
        "model_loaded": MODEL_LOADED,
        "model": "MobileNetV2",
        "input_size": "224x224",
        "class_0": "No Oil Spill",
        "class_1": "Oil Spill",
        "preprocessing": "MobileNetV2 preprocess_input [-1,+1]",
        "chip_size": "400x400",
        "stride": 200,
        "threshold": THRESHOLD,
        "boundary_type": "localized SAR anomaly contour"
    }


@app.get("/automatic-status")
def automatic_status():
    return {
        "copernicus_configured": bool(
            CDSE_CLIENT_ID and CDSE_CLIENT_SECRET
        ),
        "ais_configured": bool(AIS_API_KEY)
    }


# ============================================================
# IMAGE HELPERS
# ============================================================

def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


def bytes_to_image(contents: bytes) -> Image.Image:
    try:
        return Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {error}"
        )


# ============================================================
# MODEL PREPROCESSING
# ============================================================

def preprocess_for_model(chip: np.ndarray) -> tf.Tensor:

    resized = tf.image.resize(
        chip,
        (224, 224)
    )

    processed = (
        tf.keras.applications.mobilenet_v2
        .preprocess_input(resized)
    )

    return tf.expand_dims(
        processed,
        axis=0
    )


# ============================================================
# SINGLE CHIP PREDICTION
# ============================================================

def predict_chip(chip: np.ndarray) -> float:

    if not MODEL_LOADED or model is None:
        raise RuntimeError(
            "MobileNetV2 model is not loaded."
        )

    batch = preprocess_for_model(
        chip
    )

    prediction = model.predict(
        batch,
        verbose=0
    )

    probability = float(
        np.asarray(prediction).reshape(-1)[0]
    )

    return max(
        0.0,
        min(
            1.0,
            probability
        )
    )


# ============================================================
# SAR SCENE SCANNER
#
# EXACT TRAINING/SCANNING PIPELINE
#
# 400x400 chip
# 200 pixel stride
# 224x224 model input
# MobileNetV2 preprocess_input [-1,+1]
# ============================================================

def scan_sar_scene(image: Image.Image):

    image_array = np.array(
        image.convert("RGB")
    )

    height, width = image_array.shape[:2]

    results = []

    # Handle images smaller than one chip separately.
    if width < CHIP_SIZE or height < CHIP_SIZE:

        probability = predict_chip(
            image_array
        )

        return image_array, [
            {
                "x": 0,
                "y": 0,
                "probability": probability,
                "oil": probability >= THRESHOLD
            }
        ]

    for y in range(
        0,
        height - CHIP_SIZE + 1,
        STRIDE
    ):

        for x in range(
            0,
            width - CHIP_SIZE + 1,
            STRIDE
        ):

            chip = image_array[
                y:y + CHIP_SIZE,
                x:x + CHIP_SIZE
            ]

            if (
                chip.shape[0] != CHIP_SIZE
                or
                chip.shape[1] != CHIP_SIZE
            ):
                continue

            probability = predict_chip(
                chip
            )

            results.append(
                {
                    "x": x,
                    "y": y,
                    "probability": probability,
                    "oil": probability >= THRESHOLD
                }
            )

    return image_array, results


# ============================================================
# OIL-POSITIVE MODEL REGIONS
# ============================================================

def get_oil_results(results):

    return [
        item
        for item in results
        if item["oil"]
    ]


# ============================================================
# LOCAL SAR ANOMALY DETECTION
#
# This creates the irregular red boundary requested by the UI.
#
# IMPORTANT:
# MobileNetV2 decides whether the 400x400 chip is an
# oil-spill candidate.
#
# This secondary image-processing step localizes the dark
# SAR anomaly INSIDE that positive chip so the UI can show
# an irregular red outline rather than a giant square.
#
# It is NOT claimed to be pixel-level trained segmentation.
# ============================================================

def create_local_spill_mask(
    chip: np.ndarray
) -> np.ndarray:

    gray = np.array(
        Image.fromarray(
            chip.astype(np.uint8)
        ).convert("L"),
        dtype=np.float32
    )

    # Local background estimation.
    blurred = np.array(
        Image.fromarray(
            gray.astype(np.uint8)
        ).filter(
            ImageFilter.GaussianBlur(
                radius=12
            )
        ),
        dtype=np.float32
    )

    darkness = blurred - gray

    # Adaptive threshold.
    darkness_std = float(
        np.std(darkness)
    )

    adaptive_delta = max(
        5.0,
        darkness_std * 0.45
    )

    dark_threshold = float(
        np.percentile(
            gray,
            45
        )
    )

    mask = (
        (darkness >= adaptive_delta)
        &
        (gray <= dark_threshold)
    )

    # Remove very small isolated noise using a simple
    # neighborhood count.
    padded = np.pad(
        mask.astype(np.uint8),
        1,
        mode="constant"
    )

    neighborhood = np.zeros_like(
        mask,
        dtype=np.uint8
    )

    for dy in range(3):
        for dx in range(3):
            neighborhood += padded[
                dy:dy + mask.shape[0],
                dx:dx + mask.shape[1]
            ]

    mask = (
        mask
        &
        (neighborhood >= 3)
    )

    return mask.astype(bool)


# ============================================================
# CONNECTED COMPONENTS
# ============================================================

def connected_components(
    mask: np.ndarray,
    min_area: int = 20
):

    height, width = mask.shape

    visited = np.zeros(
        mask.shape,
        dtype=bool
    )

    components = []

    for y in range(height):

        for x in range(width):

            if not mask[y, x]:
                continue

            if visited[y, x]:
                continue

            stack = [(x, y)]
            visited[y, x] = True

            points = []

            while stack:

                px, py = stack.pop()

                points.append(
                    (px, py)
                )

                for dx, dy in (
                    (-1, -1),
                    (0, -1),
                    (1, -1),
                    (-1, 0),
                    (1, 0),
                    (-1, 1),
                    (0, 1),
                    (1, 1)
                ):

                    nx = px + dx
                    ny = py + dy

                    if nx < 0 or nx >= width:
                        continue

                    if ny < 0 or ny >= height:
                        continue

                    if visited[ny, nx]:
                        continue

                    if not mask[ny, nx]:
                        continue

                    visited[ny, nx] = True

                    stack.append(
                        (nx, ny)
                    )

            if len(points) >= min_area:
                components.append(
                    points
                )

    return components


# ============================================================
# DRAW IRREGULAR RED SPILL OUTLINES
# ============================================================

def draw_component_outline(
    draw: ImageDraw.ImageDraw,
    points,
    offset_x,
    offset_y
):

    point_set = set(
        points
    )

    boundary = []

    for x, y in points:

        neighbours = (
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1)
        )

        is_boundary = any(
            neighbour not in point_set
            for neighbour in neighbours
        )

        if is_boundary:

            boundary.append(
                (
                    x + offset_x,
                    y + offset_y
                )
            )

    if not boundary:
        return

    # Draw individual boundary pixels with a small radius.
    # This gives a continuous red contour around the
    # irregular dark SAR anomaly.
    for x, y in boundary:

        draw.ellipse(
            [
                x - 1,
                y - 1,
                x + 1,
                y + 1
            ],
            fill=(255, 0, 0)
        )


def create_annotated_image(
    image: Image.Image,
    oil_results
):

    annotated = image.copy().convert("RGB")

    draw = ImageDraw.Draw(
        annotated
    )

    localized_regions = []

    for item in oil_results:

        x = int(
            item["x"]
        )

        y = int(
            item["y"]
        )

        chip = np.array(
            image.convert("RGB")
        )[
            y:y + CHIP_SIZE,
            x:x + CHIP_SIZE
        ]

        if (
            chip.shape[0] != CHIP_SIZE
            or
            chip.shape[1] != CHIP_SIZE
        ):
            continue

        mask = create_local_spill_mask(
            chip
        )

        components = connected_components(
            mask,
            min_area=20
        )

        # Keep meaningful components only.
        components = [
            component
            for component in components
            if len(component) >= 20
        ]

        # Limit to strongest/largest anomalies.
        components = sorted(
            components,
            key=len,
            reverse=True
        )[:5]

        for component in components:

            # Ignore components that occupy almost the
            # entire chip. Those are usually background.
            component_ratio = (
                len(component)
                /
                float(CHIP_SIZE * CHIP_SIZE)
            )

            if component_ratio > 0.45:
                continue

            draw_component_outline(
                draw,
                component,
                x,
                y
            )

            xs = [
                p[0]
                for p in component
            ]

            ys = [
                p[1]
                for p in component
            ]

            localized_regions.append(
                {
                    "x": x + min(xs),
                    "y": y + min(ys),
                    "width": max(xs) - min(xs) + 1,
                    "height": max(ys) - min(ys) + 1,
                    "area_pixels": len(component)
                }
            )

    # If the model says oil but the dark-anomaly localization
    # finds nothing useful, draw a thin red outline around
    # the most likely darker part of the chip instead of
    # showing nothing.
    if oil_results and not localized_regions:

        for item in oil_results:

            x = int(
                item["x"]
            )

            y = int(
                item["y"]
            )

            chip = np.array(
                image.convert("L")
            )[
                y:y + CHIP_SIZE,
                x:x + CHIP_SIZE
            ]

            if (
                chip.shape[0] != CHIP_SIZE
                or
                chip.shape[1] != CHIP_SIZE
            ):
                continue

            threshold = np.percentile(
                chip,
                20
            )

            ys, xs = np.where(
                chip <= threshold
            )

            if len(xs) == 0:
                continue

            min_x = int(
                np.percentile(
                    xs,
                    5
                )
            )

            max_x = int(
                np.percentile(
                    xs,
                    95
                )
            )

            min_y = int(
                np.percentile(
                    ys,
                    5
                )
            )

            max_y = int(
                np.percentile(
                    ys,
                    95
                )
            )

            draw.rectangle(
                [
                    x + min_x,
                    y + min_y,
                    x + max_x,
                    y + max_y
                ],
                outline=(255, 0, 0),
                width=4
            )

            localized_regions.append(
                {
                    "x": x + min_x,
                    "y": y + min_y,
                    "width": max_x - min_x + 1,
                    "height": max_y - min_y + 1,
                    "area_pixels":
                        (
                            max_x - min_x + 1
                        )
                        *
                        (
                            max_y - min_y + 1
                        )
                }
            )

    return annotated, localized_regions


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(
    image: Image.Image
):

    width, height = image.size

    image_array, results = scan_sar_scene(
        image
    )

    if not results:

        raise RuntimeError(
            "No valid SAR regions could be created."
        )

    oil_results = get_oil_results(
        results
    )

    probabilities = [
        item["probability"]
        for item in results
    ]

    max_probability = max(
        probabilities
    )

    mean_probability = float(
        np.mean(
            probabilities
        )
    )

    spill_detected = (
        len(oil_results) > 0
    )

    if spill_detected:

        confidence = max_probability

    else:

        confidence = 1.0 - max_probability

    # Create localized red spill boundaries.
    annotated, localized_regions = (
        create_annotated_image(
            image,
            oil_results
        )
    )

    # Candidate area is based on the localized regions,
    # NOT the full 400x400 chip area.
    candidate_area_pixels = sum(
        int(
            region["area_pixels"]
        )
        for region in localized_regions
    )

    return {

        "spill_detected":
            bool(spill_detected),

        "confidence":
            float(confidence),

        "oil_probability":
            float(max_probability),

        "mean_oil_probability":
            mean_probability,

        "candidate_boundary":
            bool(
                spill_detected
                and
                len(localized_regions) > 0
            ),

        "candidate_count":
            len(localized_regions),

        "candidate_area_pixels":
            candidate_area_pixels,

        "model_positive_chips":
            len(oil_results),

        "total_chips":
            len(results),

        "oil_chips":
            len(oil_results),

        "original_image":
            image_to_data_url(
                image
            ),

        "annotated_image":
            image_to_data_url(
                annotated
            ),

        "localized_regions":
            localized_regions,

        "chip_results":
            [
                {
                    "x": item["x"],
                    "y": item["y"],
                    "probability":
                        float(
                            item["probability"]
                        ),
                    "oil":
                        bool(
                            item["oil"]
                        )
                }
                for item in results
            ],

        "note":
            (
                "MobileNetV2 classifies 400x400 SAR chips. "
                "Red outlines localize dark SAR anomalies "
                "inside oil-positive chips. The outlines "
                "are candidate spill regions, not trained "
                "pixel-level segmentation."
            )
    }


# ============================================================
# REGION UTILITIES
# ============================================================

def validate_bounds(
    north: float,
    south: float,
    east: float,
    west: float
):

    if north <= south:

        raise HTTPException(
            status_code=400,
            detail=(
                "North boundary must be greater "
                "than south boundary."
            )
        )

    if east <= west:

        raise HTTPException(
            status_code=400,
            detail=(
                "East boundary must be greater "
                "than west boundary."
            )
        )

    if not (
        -90 <= south <= 90
        and
        -90 <= north <= 90
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid latitude."
        )

    if not (
        -180 <= west <= 180
        and
        -180 <= east <= 180
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid longitude."
        )


def region_center(
    north,
    south,
    east,
    west
):

    return (
        (north + south) / 2.0,
        (east + west) / 2.0
    )


def region_area_km2(
    north,
    south,
    east,
    west
):

    center_lat = (
        north + south
    ) / 2.0

    lat_km = (
        111.32
        *
        abs(
            north - south
        )
    )

    lon_km = (
        111.32
        *
        math.cos(
            math.radians(
                center_lat
            )
        )
        *
        abs(
            east - west
        )
    )

    return abs(
        lat_km * lon_km
    )


def build_region(
    north,
    south,
    east,
    west
):

    center_lat, center_lon = (
        region_center(
            north,
            south,
            east,
            west
        )
    )

    return {

        "north":
            north,

        "south":
            south,

        "east":
            east,

        "west":
            west,

        "center_lat":
            center_lat,

        "center_lon":
            center_lon,

        "area_km2":
            region_area_km2(
                north,
                south,
                east,
                west
            )
    }


# ============================================================
# LAND / OCEAN CHECK
# ============================================================

def check_land_region(
    north,
    south,
    east,
    west
):

    center_lat, center_lon = region_center(
        north,
        south,
        east,
        west
    )

    try:

        response = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={
                "lat": center_lat,
                "lon": center_lon,
                "format": "json",
                "zoom": 10
            },
            headers={
                "User-Agent":
                    "OceanSightAI/2.0"
            },
            timeout=10
        )

        if response.status_code != 200:
            return False

        data = response.json()

        address = data.get(
            "address",
            {}
        )

        land_fields = [
            "country",
            "state",
            "state_district",
            "county",
            "city",
            "town",
            "village",
            "municipality",
            "suburb"
        ]

        for field in land_fields:

            if address.get(field):
                return True

        return False

    except Exception as error:

        print(
            "Land check unavailable:",
            error
        )

        # Do not block the scan if the external
        # geocoding service is unavailable.
        return False


# ============================================================
# ENVIRONMENTAL DATA
# ============================================================

def get_environment(
    lat,
    lon
):

    environment = {

        "wind": {
            "speed_kmh": None,
            "direction_deg": None
        },

        "current": {
            "speed_ms": None,
            "direction_deg": None
        }
    }

    # --------------------------------------------------------
    # WIND
    # --------------------------------------------------------

    try:

        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": (
                    "wind_speed_10m,"
                    "wind_direction_10m"
                )
            },
            timeout=15
        )

        if response.status_code == 200:

            data = response.json()

            current = data.get(
                "current",
                {}
            )

            environment["wind"] = {

                "speed_kmh":
                    current.get(
                        "wind_speed_10m"
                    ),

                "direction_deg":
                    current.get(
                        "wind_direction_10m"
                    )
            }

    except Exception as error:

        print(
            "Wind data unavailable:",
            error
        )


    # --------------------------------------------------------
    # OCEAN CURRENT
    # --------------------------------------------------------

    try:

        response = requests.get(
            "https://marine-api.open-meteo.com/v1/marine",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": (
                    "ocean_current_velocity,"
                    "ocean_current_direction"
                ),
                "timezone": "GMT"
            },
            timeout=20
        )

        if response.status_code == 200:

            data = response.json()

            current = data.get(
                "current",
                {}
            )

            environment["current"] = {

                "speed_ms":
                    current.get(
                        "ocean_current_velocity"
                    ),

                "direction_deg":
                    current.get(
                        "ocean_current_direction"
                    )
            }

        else:

            print(
                "Ocean current API returned:",
                response.status_code,
                response.text[:500]
            )

    except Exception as error:

        print(
            "Ocean current data unavailable:",
            error
        )


    return environment


# ============================================================
# COPERNICUS DATA SPACE AUTHENTICATION
# ============================================================

def get_copernicus_token():

    if not (
        CDSE_CLIENT_ID
        and
        CDSE_CLIENT_SECRET
    ):

        return None

    token_url = (
        "https://identity.dataspace.copernicus.eu/"
        "auth/realms/CDSE/"
        "protocol/openid-connect/token"
    )

    try:

        response = requests.post(
            token_url,
            data={
                "client_id":
                    CDSE_CLIENT_ID,

                "client_secret":
                    CDSE_CLIENT_SECRET,

                "grant_type":
                    "client_credentials"
            },
            timeout=20
        )

        response.raise_for_status()

        token = response.json().get(
            "access_token"
        )

        return token

    except Exception as error:

        print(
            "Copernicus authentication failed:",
            error
        )

        return None


# ============================================================
# SENTINEL-1 CATALOG SEARCH
#
# Current Copernicus Data Space Catalog API.
#
# Uses:
#   https://sh.dataspace.copernicus.eu/catalog/v1/search
#
# Collection:
#   sentinel-1-grd
#
# BBOX:
#   selected map rectangle
# ============================================================

def search_sentinel1(
    north,
    south,
    east,
    west
):

    token = get_copernicus_token()

    if not token:

        return {

            "status":
                "credentials_required",

            "message":
                (
                    "Copernicus Data Space credentials "
                    "are required for automatic "
                    "Sentinel-1 retrieval."
                )
        }

    now = datetime.now(
        timezone.utc
    )

    start = now - timedelta(
        days=30
    )

    bbox = [
        west,
        south,
        east,
        north
    ]

    catalog_url = (
        "https://sh.dataspace.copernicus.eu/"
        "catalog/v1/search"
    )

    params = {

        "bbox":
            ",".join(
                str(value)
                for value in bbox
            ),

        "datetime":
            (
                start.isoformat()
                .replace(
                    "+00:00",
                    "Z"
                )
                +
                "/"
                +
                now.isoformat()
                .replace(
                    "+00:00",
                    "Z"
                )
            ),

        "collections":
            "sentinel-1-grd",

        "limit":
            10
    }

    try:

        response = requests.get(
            catalog_url,
            params=params,
            headers={
                "Authorization":
                    f"Bearer {token}"
            },
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        features = data.get(
            "features",
            []
        )

        if not features:

            return {

                "status":
                    "no_sentinel1",

                "message":
                    (
                        "No recent Sentinel-1 GRD "
                        "acquisition was found for "
                        "the selected region."
                    )
            }

        # Most recent result first.
        features = sorted(
            features,
            key=lambda item:
                item.get(
                    "properties",
                    {}
                ).get(
                    "datetime",
                    ""
                ),
            reverse=True
        )

        selected = features[0]

        properties = selected.get(
            "properties",
            {}
        )

        return {

            "status":
                "success",

            "product":
                {

                    "id":
                        properties.get(
                            "id"
                        )
                        or
                        selected.get(
                            "id"
                        ),

                    "name":
                        properties.get(
                            "title"
                        ),

                    "datetime":
                        properties.get(
                            "datetime"
                        ),

                    "collection":
                        "sentinel-1-grd"
                },

            "catalog_feature":
                selected,

            "center_lat":
                (north + south) / 2.0,

            "center_lon":
                (east + west) / 2.0
        }

    except requests.HTTPError as error:

        body = ""

        try:
            body = response.text[:1000]
        except Exception:
            pass

        print(
            "Sentinel-1 catalog HTTP error:",
            error,
            body
        )

        return {

            "status":
                "error",

            "message":
                (
                    "Sentinel-1 catalog request failed: "
                    f"{error}"
                )
        }

    except Exception as error:

        print(
            "Sentinel-1 catalog search failed:",
            error
        )

        return {

            "status":
                "error",

            "message":
                (
                    "Sentinel-1 catalog search failed: "
                    f"{error}"
                )
        }


# ============================================================
# SENTINEL-1 IMAGE RETRIEVAL
#
# Current Copernicus Data Space Process API:
#
# https://sh.dataspace.copernicus.eu/process/v1
#
# Sentinel-1 GRD
# VV polarization
# ============================================================

def retrieve_sentinel1_image(
    north,
    south,
    east,
    west
):

    token = get_copernicus_token()

    if not token:
        return None

    bbox = [
        west,
        south,
        east,
        north
    ]

    process_url = (
        "https://sh.dataspace.copernicus.eu/"
        "process/v1"
    )

    now = datetime.now(
        timezone.utc
    )

    start = now - timedelta(
        days=30
    )

    payload = {

        "input": {

            "bounds": {

                "bbox":
                    bbox
            },

            "data": [

                {

                    "type":
                        "sentinel-1-grd",

                    "dataFilter": {

                        "timeRange": {

                            "from":
                                start.isoformat()
                                .replace(
                                    "+00:00",
                                    "Z"
                                ),

                            "to":
                                now.isoformat()
                                .replace(
                                    "+00:00",
                                    "Z"
                                )
                        },

                        "mosaickingOrder":
                            "mostRecent"
                    },

                    "processing": {

                        "orthorectify":
                            "true"
                    }
                }
            ]
        },

        "output": {

            "width":
                1024,

            "height":
                1024,

            "responses": [

                {

                    "identifier":
                        "default",

                    "format": {

                        "type":
                            "image/png"
                    }
                }
            ]
        },

        "evalscript":
            """
            //VERSION=3

            function setup() {

                return {

                    input: ["VV"],

                    output: {

                        bands: 1,

                        sampleType: "AUTO"
                    }
                };
            }

            function evaluatePixel(sample) {

                return [sample.VV];
            }
            """
    }

    try:

        response = requests.post(
            process_url,
            json=payload,
            headers={
                "Authorization":
                    f"Bearer {token}",

                "Content-Type":
                    "application/json"
            },
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        content_type = (
            response.headers
            .get(
                "content-type",
                ""
            )
            .lower()
        )

        if (
            "image" not in content_type
            and
            not response.content.startswith(
                b"\x89PNG"
            )
        ):

            print(
                "Unexpected Sentinel-1 response:",
                content_type,
                response.text[:1000]
            )

            return None

        return response.content

    except requests.HTTPError as error:

        print(
            "Sentinel-1 Process API HTTP error:",
            error
        )

        try:

            print(
                response.text[:1500]
            )

        except Exception:
            pass

        return None

    except Exception as error:

        print(
            "Sentinel-1 image retrieval failed:",
            error
        )

        return None


# ============================================================
# DRIFT ESTIMATE
# ============================================================

def calculate_drift(
    wind_speed_kmh,
    wind_direction_deg,
    current_speed_ms,
    current_direction_deg
):

    if (
        wind_speed_kmh is None
        and
        current_speed_ms is None
    ):

        return None

    wind_speed_ms = (
        float(
            wind_speed_kmh or 0
        )
        /
        3.6
    )

    wind_dir = math.radians(
        float(
            wind_direction_deg or 0
        )
    )

    current_dir = math.radians(
        float(
            current_direction_deg or 0
        )
    )

    wind_u = (
        wind_speed_ms
        *
        math.sin(
            wind_dir
        )
    )

    wind_v = (
        wind_speed_ms
        *
        math.cos(
            wind_dir
        )
    )

    current_u = (
        float(
            current_speed_ms or 0
        )
        *
        math.sin(
            current_dir
        )
    )

    current_v = (
        float(
            current_speed_ms or 0
        )
        *
        math.cos(
            current_dir
        )
    )

    total_u = (
        0.03 * wind_u
        +
        current_u
    )

    total_v = (
        0.03 * wind_v
        +
        current_v
    )

    speed_ms = math.sqrt(
        total_u ** 2
        +
        total_v ** 2
    )

    distance_km = (
        speed_ms
        *
        3600
        /
        1000
    )

    direction_deg = (
        math.degrees(
            math.atan2(
                total_u,
                total_v
            )
        )
        +
        360
    ) % 360

    return {

        "distance_km":
            distance_km,

        "direction_deg":
            direction_deg,

        "forecast_hours":
            1,

        "method":
            (
                "short-term wind/current "
                "vector estimate"
            )
    }


# ============================================================
# AIS
#
# AIS is contextual evidence only.
# It does NOT establish causation.
# ============================================================

# Update your get_ais function signature to make arguments optional:
def get_ais(north: float = None, south: float = None, east: float = None, west: float = None, lat: float = None, lon: float = None, radius: float = 50.0):
    # Your existing code inside get_ais...
    vessels_list = []
    
    for mmsi, history in TRAIL_CACHE.items():
        if history:
            latest = history[-1]
            trail_coords = [[point["lat"], point["lon"]] for point in history]
            
            vessels_list.append({
                "mmsi": mmsi,
                "name": latest.get("name", f"Vessel {mmsi}"),
                "lat": latest["lat"],
                "lon": latest["lon"],
                "trail": trail_coords
            })
            
    return {"vessels": vessels_list}

def get_simulated_manual_ais(north, south, east, west):
    """
    Simulated AIS data for Manual Scan demonstration.
    Clearly marked as simulated — not live vessel data.
    """

    lat_range = north - south
    lon_range = east - west

    vessels = [
        {
            "mmsi": "636019842",
            "name": "DEMO OCEAN TRADER",
            "type": "Cargo Vessel",
            "speed": 12.4,
            "heading": 78,
            "destination": "CHENNAI",
            "lat": south + lat_range * 0.72,
            "lon": west + lon_range * 0.25,
            "trail": [
                [south + lat_range * 0.60, west + lon_range * 0.10],
                [south + lat_range * 0.64, west + lon_range * 0.15],
                [south + lat_range * 0.68, west + lon_range * 0.20],
                [south + lat_range * 0.72, west + lon_range * 0.25]
            ]
        },
        {
            "mmsi": "636027531",
            "name": "DEMO SEA HAWK",
            "type": "Tanker",
            "speed": 9.7,
            "heading": 145,
            "destination": "COLOMBO",
            "lat": south + lat_range * 0.42,
            "lon": west + lon_range * 0.68,
            "trail": [
                [south + lat_range * 0.60, west + lon_range * 0.82],
                [south + lat_range * 0.54, west + lon_range * 0.77],
                [south + lat_range * 0.48, west + lon_range * 0.72],
                [south + lat_range * 0.42, west + lon_range * 0.68]
            ]
        },
        {
            "mmsi": "636031274",
            "name": "DEMO MARINE STAR",
            "type": "Service Vessel",
            "speed": 6.2,
            "heading": 225,
            "destination": "OFFSHORE",
            "lat": south + lat_range * 0.25,
            "lon": west + lon_range * 0.45,
            "trail": [
                [south + lat_range * 0.15, west + lon_range * 0.55],
                [south + lat_range * 0.18, west + lon_range * 0.52],
                [south + lat_range * 0.22, west + lon_range * 0.48],
                [south + lat_range * 0.25, west + lon_range * 0.45]
            ]
        }
    ]

    return {"vessels": vessels}

# ============================================================
# SIMULATED AIS — MANUAL SCAN DEMONSTRATION ONLY
# ============================================================

def get_simulated_manual_ais(
    north: float,
    south: float,
    east: float,
    west: float
):
    """
    Simulated AIS data used ONLY for the Manual Scan showcase.

    This does not represent real vessels or live AIS data.
    All vessel positions are generated inside the selected box.
    """

    lat_span = max(abs(north - south), 0.01)
    lon_span = max(abs(east - west), 0.01)

    def point(lat_fraction, lon_fraction):
        lat = south + (lat_span * lat_fraction)
        lon = west + (lon_span * lon_fraction)

        return [
            round(lat, 6),
            round(lon, 6)
        ]

    vessel_definitions = [

        {
            "mmsi": "636019842",
            "name": "DEMO OCEAN TRADER",
            "type": "Cargo Vessel",
            "speed_knots": 12.4,
            "heading": 78,
            "destination": "CHENNAI",
            "path": [
                (0.18, 0.12),
                (0.23, 0.20),
                (0.29, 0.29),
                (0.35, 0.39),
                (0.42, 0.49),
                (0.48, 0.59),
                (0.54, 0.68)
            ]
        },

        {
            "mmsi": "636027531",
            "name": "DEMO SEA HAWK",
            "type": "Tanker",
            "speed_knots": 9.7,
            "heading": 145,
            "destination": "COLOMBO",
            "path": [
                (0.82, 0.78),
                (0.75, 0.73),
                (0.68, 0.67),
                (0.61, 0.61),
                (0.54, 0.54),
                (0.47, 0.48),
                (0.40, 0.42)
            ]
        },

        {
            "mmsi": "636031274",
            "name": "DEMO MARINE STAR",
            "type": "Service Vessel",
            "speed_knots": 6.2,
            "heading": 225,
            "destination": "OFFSHORE",
            "path": [
                (0.22, 0.78),
                (0.28, 0.71),
                (0.34, 0.64),
                (0.40, 0.57),
                (0.46, 0.50),
                (0.52, 0.43),
                (0.58, 0.36)
            ]
        }

    ]

    vessels = []

    for vessel in vessel_definitions:

        trail = [
            point(lat_fraction, lon_fraction)
            for lat_fraction, lon_fraction
            in vessel["path"]
        ]

        current = trail[-1]

        vessels.append({

            "mmsi":
                vessel["mmsi"],

            "name":
                vessel["name"],

            "type":
                vessel["type"],

            "lat":
                current[0],

            "lon":
                current[1],

            "speed_knots":
                vessel["speed_knots"],

            "heading":
                vessel["heading"],

            "destination":
                vessel["destination"],

            "trail":
                trail

        })

    return {

        "configured":
            True,

        "simulated":
            True,

        "provider":
            "SIMULATED AIS — MANUAL DEMONSTRATION",

        "vessel_count":
            len(vessels),

        "vessels":
            vessels
    }

# ============================================================
# RISK
# ============================================================

def calculate_risk(
    spill_detected,
    oil_probability,
    wind_speed_kmh=None,
    candidate_count=0
):

    if not spill_detected:

        return {

            "level":
                "Low",

            "score":
                0,

            "reason":
                "No oil-spill candidate detected."
        }

    score = 0

    if oil_probability >= 0.85:

        score += 3

    elif oil_probability >= 0.65:

        score += 2

    else:

        score += 1

    if wind_speed_kmh is not None:

        if wind_speed_kmh >= 30:

            score += 2

        elif wind_speed_kmh >= 15:

            score += 1

    if candidate_count >= 4:

        score += 2

    elif candidate_count >= 2:

        score += 1

    if score >= 5:

        level = "High"

    elif score >= 3:

        level = "Moderate"

    else:

        level = "Low"

    return {

        "level":
            level,

        "score":
            score,

        "reason":
            "Based on model confidence, candidate extent and environmental conditions."
    }


# ============================================================
# COMMON RESPONSE BUILDER
# ============================================================

def build_analysis_response(
    analysis,
    region,
    mode,
    environment=None,
    satellite=None,
    ais=None,
    drift=None
):

    if environment is None:

        environment = {

            "wind": {

                "speed_kmh":
                    None,

                "direction_deg":
                    None
            },

            "current": {

                "speed_ms":
                    None,

                "direction_deg":
                    None
            }
        }

    if ais is None:

        ais = {

            "configured":
                False,

            "vessel_count":
                None,

            "vessels":
                []
        }

    wind = environment.get(
        "wind",
        {}
    )

    risk = calculate_risk(
        analysis[
            "spill_detected"
        ],

        analysis[
            "oil_probability"
        ],

        wind.get(
            "speed_kmh"
        ),

        analysis[
            "candidate_count"
        ]
    )

    return {

        "status":
            "success",

        "mode":
            mode,

        # ----------------------------------------------------
        # MODEL RESULT
        # ----------------------------------------------------

        "spill_detected":
            analysis[
                "spill_detected"
            ],

        "confidence":
            analysis[
                "confidence"
            ],

        "oil_probability":
            analysis[
                "oil_probability"
            ],

        "mean_oil_probability":
            analysis[
                "mean_oil_probability"
            ],

        # ----------------------------------------------------
        # RED BOUNDARY
        # ----------------------------------------------------

        "candidate_boundary":
            analysis[
                "candidate_boundary"
            ],

        "candidate_count":
            analysis[
                "candidate_count"
            ],

        "candidate_area_pixels":
            analysis[
                "candidate_area_pixels"
            ],

        "localized_regions":
            analysis[
                "localized_regions"
            ],

        "model_positive_chips":
            analysis[
                "model_positive_chips"
            ],

        "total_chips":
            analysis[
                "total_chips"
            ],

        "oil_chips":
            analysis[
                "oil_chips"
            ],

        # ----------------------------------------------------
        # SAR IMAGES
        # ----------------------------------------------------

        "original_image":
            analysis[
                "original_image"
            ],

        "annotated_image":
            analysis[
                "annotated_image"
            ],

        # Compatibility fields.
        "image":
            analysis[
                "original_image"
            ],

        "annotated_sar":
            analysis[
                "annotated_image"
            ],

        # ----------------------------------------------------
        # EXTRA MODEL DATA
        # ----------------------------------------------------

        "chip_results":
            analysis[
                "chip_results"
            ],

        "note":
            analysis[
                "note"
            ],

        # ----------------------------------------------------
        # GEOGRAPHY
        # ----------------------------------------------------

        "region":
            region,

        "coordinates":
            {

                "latitude":
                    region[
                        "center_lat"
                    ],

                "longitude":
                    region[
                        "center_lon"
                    ]
            },

        "georeferenced":
            mode == "Automatic",

        # ----------------------------------------------------
        # ENVIRONMENT
        # ----------------------------------------------------

        "environment":
            environment,

        # Frontend-friendly aliases.
        "wind":
            environment.get(
                "wind"
            ),

        "current":
            environment.get(
                "current"
            ),

        # ----------------------------------------------------
        # AIS
        # ----------------------------------------------------

        "ais":
            ais,

        "vessels":
            ais.get(
                "vessels",
                []
            ),

        # ----------------------------------------------------
        # DRIFT
        # ----------------------------------------------------

        "drift":
            drift,

        # ----------------------------------------------------
        # RISK
        # ----------------------------------------------------

        "risk":
            risk,

        # ----------------------------------------------------
        # ALERT
        # ----------------------------------------------------

        "alert":
            (
                "Alert condition"
                if analysis[
                    "spill_detected"
                ]
                else
                "No alert"
            ),

        # ----------------------------------------------------
        # SATELLITE
        # ----------------------------------------------------

        "satellite":
            satellite
    }


# ============================================================
# LAND RESPONSE
# ============================================================

def land_response():

    return {

        "status":
            "land",

        "message":
            (
                "LAND REGION SELECTED — "
                "Please select an ocean/marine region "
                "for oil-spill monitoring."
            )
    }


# ============================================================
# MANUAL SCAN
# ============================================================

@app.post("/manual-scan")
async def manual_scan(
    north: float,
    south: float,
    east: float,
    west: float,
    file: UploadFile = File(...)
):

    validate_bounds(
        north,
        south,
        east,
        west
    )

    if not MODEL_LOADED:

        raise HTTPException(
            status_code=500,
            detail="MobileNetV2 model is not loaded."
        )

    # --------------------------------------------------------
    # LAND CHECK
    # --------------------------------------------------------

    if check_land_region(
        north,
        south,
        east,
        west
    ):

        return land_response()

    # --------------------------------------------------------
    # READ SAR IMAGE
    # --------------------------------------------------------

    contents = await file.read()

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Uploaded SAR image is empty."
        )

    image = bytes_to_image(
        contents
    )

    # --------------------------------------------------------
    # MODEL ANALYSIS
    # --------------------------------------------------------

    try:

        analysis = analyze_image(
            image
        )

    except Exception as error:

        print(
            "Manual analysis failed:",
            error
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "SAR analysis failed: "
                f"{error}"
            )
        )

    # --------------------------------------------------------
    # REGION
    # --------------------------------------------------------

    region = build_region(
        north,
        south,
        east,
        west
    )

    center_lat = region[
        "center_lat"
    ]

    center_lon = region[
        "center_lon"
    ]

    # --------------------------------------------------------
    # ENVIRONMENT
    # --------------------------------------------------------

    environment = get_environment(
        center_lat,
        center_lon
    )

    wind = environment.get(
        "wind",
        {}
    )

    # --------------------------------------------------------
    # DRIFT
    # --------------------------------------------------------

    drift = calculate_drift(
        wind.get(
            "speed_kmh"
        ),

        wind.get(
            "direction_deg"
        ),

        None,

        None
    )

    # --------------------------------------------------------
    # AIS
    # --------------------------------------------------------

    ais = get_simulated_manual_ais(
        north,
        south,
        east,
        west
    )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return build_analysis_response(

        analysis=
            analysis,

        region=
            region,

        mode=
            "Manual",

        environment=
            environment,

        satellite=
            None,

        ais=
            ais,

        drift=
            drift
    )


# ============================================================
# AUTOMATIC SCAN
# ============================================================

@app.post("/automatic-scan")
async def automatic_scan(
    north: float,
    south: float,
    east: float,
    west: float
):

    validate_bounds(
        north,
        south,
        east,
        west
    )

    if not MODEL_LOADED:

        raise HTTPException(
            status_code=500,
            detail="MobileNetV2 model is not loaded."
        )

    # --------------------------------------------------------
    # LAND CHECK
    # --------------------------------------------------------

    if check_land_region(
        north,
        south,
        east,
        west
    ):

        return land_response()

    # --------------------------------------------------------
    # COPERNICUS CREDENTIAL CHECK
    # --------------------------------------------------------

    if not (
        CDSE_CLIENT_ID
        and
        CDSE_CLIENT_SECRET
    ):

        return {

            "status":
                "credentials_required",

            "message":
                (
                    "Copernicus Data Space credentials "
                    "are required for automatic "
                    "Sentinel-1 retrieval."
                ),

            "setup":
                {

                    "required":
                        [
                            "CDSE_CLIENT_ID",
                            "CDSE_CLIENT_SECRET"
                        ],

                    "service":
                        "Copernicus Data Space Ecosystem"
                }
        }

    # --------------------------------------------------------
    # SEARCH SENTINEL-1
    # --------------------------------------------------------

    search_result = search_sentinel1(
        north,
        south,
        east,
        west
    )

    if search_result[
        "status"
    ] == "credentials_required":

        return search_result

    if search_result[
        "status"
    ] == "no_sentinel1":

        return search_result

    if search_result[
        "status"
    ] == "error":

        raise HTTPException(
            status_code=502,
            detail=search_result[
                "message"
            ]
        )

    # --------------------------------------------------------
    # RETRIEVE SAR
    # --------------------------------------------------------

    png = retrieve_sentinel1_image(
        north,
        south,
        east,
        west
    )

    if png is None:

        raise HTTPException(
            status_code=502,
            detail=(
                "Sentinel-1 acquisition was found, "
                "but the SAR image could not be "
                "retrieved from the Copernicus "
                "Data Space Process API."
            )
        )

    image = bytes_to_image(
        png
    )

    # --------------------------------------------------------
    # MODEL ANALYSIS
    # --------------------------------------------------------

    try:

        analysis = analyze_image(
            image
        )

    except Exception as error:

        print(
            "Automatic SAR analysis failed:",
            error
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Automatic SAR analysis failed: "
                f"{error}"
            )
        )

    # --------------------------------------------------------
    # REGION
    # --------------------------------------------------------

    region = build_region(
        north,
        south,
        east,
        west
    )

    center_lat = region[
        "center_lat"
    ]

    center_lon = region[
        "center_lon"
    ]

    # --------------------------------------------------------
    # AIS
    # MANUAL SCAN USES SIMULATED AIS FOR DEMONSTRATION
    # --------------------------------------------------------

    ais = get_simulated_manual_ais(
        north,
        south,
        east,
        west
    )
    # --------------------------------------------------------
    # ENVIRONMENT
    # --------------------------------------------------------

    environment = get_environment(
        center_lat,
        center_lon
    )

    # Current provider is not configured yet.
    # Never fabricate current values.

    environment[
        "current"
    ] = {

        "speed_ms":
            None,

        "direction_deg":
            None
    }

    wind = environment.get(
        "wind",
        {}
    )

    current = environment.get(
        "current",
        {}
    )

    # --------------------------------------------------------
    # DRIFT
    # --------------------------------------------------------

    drift = calculate_drift(

        wind.get(
            "speed_kmh"
        ),

        wind.get(
            "direction_deg"
        ),

        current.get(
            "speed_ms"
        ),

        current.get(
            "direction_deg"
        )
    )

    # --------------------------------------------------------
    # AIS
    # --------------------------------------------------------

    ais = get_ais(
        center_lat,
        center_lon
    )

    

    # --------------------------------------------------------
    # SATELLITE METADATA
    # --------------------------------------------------------

    product = search_result.get(
        "product",
        {}
    )

    satellite = {

        "id":
            product.get(
                "id"
            ),

        "name":
            product.get(
                "name"
            ),

        "datetime":
            product.get(
                "datetime"
            ),

        "collection":
            "Sentinel-1 GRD"
    }

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return build_analysis_response(

        analysis=
            analysis,

        region=
            region,

        mode=
            "Automatic",

        environment=
            environment,

        satellite=
            satellite,

        ais=
            ais,

        drift=
            drift
    )


# ============================================================
# DIRECT PREDICTION
#
# Kept for compatibility/testing.
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    if not MODEL_LOADED:

        raise HTTPException(
            status_code=500,
            detail="MobileNetV2 model is not loaded."
        )

    contents = await file.read()

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    image = bytes_to_image(
        contents
    )

    image_array = np.array(
        image
    )

    probability = predict_chip(
        image_array
    )

    spill = (
        probability >= THRESHOLD
    )

    return {

        "prediction":
            probability,

        "oil_probability":
            probability,

        "spill_detected":
            bool(spill),

        "confidence":
            (
                probability
                if spill
                else
                1.0 - probability
            ),

        "class_0":
            "No Oil Spill",

        "class_1":
            "Oil Spill",

        "preprocessing":
            (
                "MobileNetV2 "
                "preprocess_input [-1,+1]"
            ),

        "threshold":
            THRESHOLD
    }