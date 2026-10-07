“AI-Driven Marine Oil Spill Detection, Source Attribution, and Spatiotemporal Risk Assessment Using SAR, AIS, GIS, and Environmental Data”


An AI-powered marine monitoring system for detecting oil spills from satellite imagery and combining **SAR data, AIS vessel information, GIS mapping, and weather/ocean conditions** to analyze spill location, possible sources, movement, and environmental risk.



Overview

Oil spills pose a serious threat to marine ecosystems, coastal regions, fisheries, and maritime operations. Early detection and accurate assessment are essential for effective response.

This project proposes an neat pipeline that goes beyond simple oil-spill detection. The system combines **satellite image analysis with maritime vessel data, geographic information, and environmental conditions** to provide a more complete understanding of an oil-spill event.

Core Pipeline

```text
Satellite Imagery
       ↓
SAR Image Processing
       ↓
AI-Based Oil Spill Detection
       ↓
Spill Localization & Area Estimation
       ↓
AIS Vessel Data
       ↓
Possible Source / Vessel Correlation
       ↓
GIS Mapping
       ↓
Weather & Ocean Current Analysis
       ↓
Spill Drift Prediction
       ↓
Coastal & Sensitive-Area Impact
       ↓
Risk Assessment & Alerts
```

---

Objectives

* Detect oil spills using satellite imagery.
* Utilize **SAR imagery** for reliable detection under conditions such as cloud cover.
* Classify satellite images into **Oil Spill / No Oil Spill**.
* Estimate the approximate size and severity of detected spills.
* Locate detected spills geographically using GIS.
* Integrate **AIS vessel data** to identify vessels operating near the spill.
* Analyze historical vessel tracks for possible source correlation.
* Incorporate weather and ocean conditions into spill analysis.
* Predict the possible movement and spread of an oil spill.
* Identify potentially affected coastal and environmentally sensitive regions.
* Generate an overall marine risk assessment.
* Provide alerts for significant spill events.

---

Current AI Model

The initial implementation focuses on binary oil-spill classification using satellite images.

Model

**MobileNetV2**

The model is used as a transfer-learning-based image classification model for distinguishing between:

* `Class 0` → No Oil Spill
* `Class 1` → Oil Spill

### Dataset

Current dataset:

| Class        |    Images |
| ------------ | --------: |
| No Oil Spill |     3,695 |
| Oil Spill    |     1,843 |
| **Total**    | **5,548** |

Images are resized to:

```text
224 × 224 × 3
```

### Data Augmentation

The training pipeline includes:

* Random Flip
* Random Rotation
* Random Zoom
* Random Contrast

Class weights are also used to handle the imbalance between the two classes.

### Example Result

The trained model has demonstrated approximately **93% training accuracy**, with individual test predictions achieving high confidence in suitable cases.

> Note: Model performance will be evaluated further using appropriate validation and test metrics as the project develops.

---

# 🛰️ SAR Satellite Integration

Synthetic Aperture Radar (SAR) imagery is an important component of the planned system.

Unlike conventional optical satellite imagery, SAR can operate through clouds and during both day and night, making it particularly useful for marine monitoring.

The planned workflow is:

```text
SAR Satellite Image
        ↓
Preprocessing
        ↓
AI-Based Detection
        ↓
Oil Spill Region
        ↓
Geographic Coordinates
```

SAR-based detection will improve the system's usefulness for real-world marine monitoring.

---

AIS Vessel Analysis

Automatic Identification System (AIS) data will be integrated with detected spill locations.

AIS can provide information such as:

* Vessel position
* Vessel identity
* Vessel speed
* Vessel heading
* Timestamp
* Vessel movement history

The system can search for vessels operating near the detected spill.

Historical Track Analysis

Instead of simply displaying nearby vessels, the system can analyze their previous movements:

```text
Oil Spill Location
       ↓
Define Search Radius
       ↓
Retrieve Nearby Vessels
       ↓
Analyze Historical AIS Tracks
       ↓
Identify Vessels That Passed Near Area
       ↓
Generate Possible Source Correlation
```

This analysis is intended to identify **possible correlations**, not to definitively determine which vessel caused a spill.

---

GIS Mapping

Geographic Information System (GIS) technology will be used to visualize and analyze the detected event.

The GIS layer can display:

* Oil-spill location
* Spill boundary
* Estimated spill area
* Nearby vessels
* Vessel tracks
* Coastlines
* Ports
* Fishing areas
* Marine protected areas
* Environmentally sensitive regions
* Predicted spill trajectory

### Example Concept

```text
             Vessel Track
                  ↓
        🚢 ──────────────
                    \
                     \
              🛢️ SPILL AREA
                 ███████
               ███████████
                    ↓
             Predicted Drift
                    ↓
             🌊 → → → → 🏝️
                         Coast
```

---

Weather & Ocean Analysis

Environmental conditions strongly influence how an oil spill moves.

The system will incorporate relevant environmental information such as:

* Wind speed
* Wind direction
* Ocean currents
* Wave conditions
* Other available environmental parameters

These factors can be combined with the detected spill location to estimate the likely direction of movement.

---

Spill Drift Prediction

One of the major planned features is predicting where the oil spill may move over time.

### Concept

```text
Current Spill
      ↓
Wind Data
      +
Ocean Current Data
      ↓
Drift Model
      ↓
6 Hour Prediction
      ↓
12 Hour Prediction
      ↓
24 Hour Prediction
```

The predicted trajectory can then be displayed on the GIS map.

This can help identify areas that may potentially be affected in the future.

---

Spill Size & Severity Estimation

The system will be extended to estimate the approximate geographical area covered by the detected spill.

Possible severity levels:

| Level       | Description                                             |
| ----------- | ------------------------------------------------------- |
| 🟢 Low      | Small detected spill                                    |
| 🟡 Moderate | Medium-scale spill                                      |
| 🟠 High     | Large spill or increasing spread                        |
| 🔴 Critical | Large spill with significant environmental/coastal risk |

The severity assessment can consider factors such as:

* Spill area
* Distance from coastline
* Predicted movement
* Environmental conditions
* Nearby sensitive regions

---

Coastal & Environmental Impact

The predicted spill trajectory can be compared against geographic features.

The system can identify whether the predicted movement approaches:

* Coastal regions
* Ports
* Fishing zones
* Marine protected areas
* Ecologically sensitive regions

Example:

```text
Spill Detected
      ↓
Predict Movement
      ↓
Overlay With GIS Layers
      ↓
Does Trajectory Intersect Sensitive Area?
      ↓
Risk Assessment
```

This changes the system from simply **detecting an oil spill** to helping understand its potential environmental impact.

---

# ⚠️ Marine Risk Assessment

A future risk-scoring module can combine multiple parameters:

```text
Spill Size
     +
Distance From Coast
     +
Spill Movement
     +
Weather Conditions
     +
Ocean Currents
     +
Sensitive Areas
     +
Vessel Activity
     ↓
Overall Risk Score
```

Example output:

```text
╔════════════════════════════════╗
║       OIL SPILL ALERT          ║
╠════════════════════════════════╣
║ Risk Level       : HIGH        ║
║ Spill Area       : XX km²      ║
║ Coast Distance   : XX km       ║
║ Nearby Vessels   : XX          ║
║ Predicted Drift  : Northeast   ║
║ Coastal Risk     : HIGH        ║
╚════════════════════════════════╝
```

---

# 🛰️ Multi-Temporal Monitoring

Instead of analyzing only one satellite image, future versions can compare images captured at different times.

```text
T₀ → Initial Detection
 ↓
T₁ → Spill Expansion
 ↓
T₂ → Spill Movement
 ↓
T₃ → Current Condition
```

This allows the system to study:

* Spill growth
* Movement
* Reduction
* Changes in shape
* Overall evolution over time

---

Automated Alert System

A future alert module can automatically generate notifications when a significant spill is detected.

An alert could contain:

```text
OIL SPILL DETECTED

Location:
Latitude: XX.XXXX
Longitude: XX.XXXX

Estimated Area:
XX km²

Severity:
HIGH

Predicted Movement:
Northeast

Potential Coastal Impact:
Within XX hours

Nearby Vessel Activity:
XX vessels
```

This could eventually be extended to email, SMS, or a dedicated monitoring dashboard.

---

Final System Architecture

```text
                    ┌──────────────────────┐
                    │  Satellite Imagery   │
                    │   Optical / SAR      │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ AI Oil Spill Model   │
                    │    MobileNetV2       │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Spill Localization   │
                    │ & Area Estimation    │
                    └──────────┬───────────┘
                               ↓
              ┌────────────────┴────────────────┐
              ↓                                 ↓
     ┌──────────────────┐             ┌──────────────────┐
     │   AIS Dataset    │             │ Weather / Ocean  │
     │ Vessel Tracking  │             │    Conditions    │
     └────────┬─────────┘             └────────┬─────────┘
              ↓                                ↓
     ┌──────────────────┐             ┌──────────────────┐
     │ Vessel & Source  │             │ Drift Prediction │
     │    Analysis      │             │                  │
     └────────┬─────────┘             └────────┬─────────┘
              └────────────────┬───────────────┘
                               ↓
                    ┌──────────────────────┐
                    │     GIS Mapping      │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Environmental Impact │
                    │   & Risk Assessment  │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Alerts / Dashboard   │
                    └──────────────────────┘
```

---

Technologies

### AI / Machine Learning

* Python
* TensorFlow
* MobileNetV2
* CNN / Transfer Learning
* NumPy
* Pandas
* Scikit-learn

Satellite & Geospatial

* SAR Satellite Data
* GIS
* Geospatial Data Processing
* Satellite Image Analysis

Maritime Data

* AIS Data
* Vessel Tracking
* Historical Vessel Trajectory Analysis

Environmental Data

* Weather Data
* Wind Information
* Ocean Current Data
* Marine Conditions

Development

* Google Colab
* Google Drive
* GitHub

---

Project Structure

```text
Oil-Spill-Detection/
│
├── Dataset/
│   ├── Class_0/
│   └── Class_1/
│
├── Models/
│   └── oil_spill_mobilenetv2.keras
│
├── Results/
│   ├── Training/
│   ├── Predictions/
│   └── Evaluation/
│
├── Test_Images/
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   └── prediction.ipynb
│
├── GIS/
│
├── AIS/
│
├── Weather/
│
├── README.md
└── requirements.txt
```

---

Development Roadmap

Phase 1 — AI Detection ✅

* [x] Dataset preparation
* [x] Image preprocessing
* [x] Data augmentation
* [x] MobileNetV2 implementation
* [x] Oil Spill / No Oil Spill classification
* [x] Model training
* [x] Test prediction

Phase 2 — Satellite & Geospatial Integration 

* [ ] SAR dataset integration
* [ ] Spill localization
* [ ] Spill area estimation
* [ ] GIS visualization

Phase 3 — Maritime Intelligence 🔄

* [ ] AIS data integration
* [ ] Nearby vessel detection
* [ ] Historical vessel track analysis
* [ ] Possible source correlation

Phase 4 — Environmental Analysis 🔄

* [ ] Weather data integration
* [ ] Wind analysis
* [ ] Ocean current integration
* [ ] Spill drift prediction

Phase 5 — Risk & Response 🚀

* [ ] Coastal impact prediction
* [ ] Sensitive-area analysis
* [ ] Spill severity classification
* [ ] Risk scoring
* [ ] Multi-temporal monitoring
* [ ] Automated alerts
* [ ] Monitoring dashboard

---

Expected Outcome

The final system aims to provide an integrated platform capable of:

**Detecting → Locating → Investigating → Predicting → Assessing → Alerting**

rather than treating oil-spill detection as an isolated image-classification problem.

The project ultimately aims to support faster identification of marine oil spills and provide useful information for **environmental monitoring, maritime safety, and emergency response planning**.

---

Project Status

**Currently under development**

The initial AI-based oil-spill classification model has been developed. Integration of SAR imagery, AIS vessel data, GIS mapping, environmental conditions, drift prediction, and risk assessment is being developed as subsequent stages of the project.
