# Interactive Basal Stimulation Guidance — Proof of Concept (PoC)

A lightweight, camera-only Proof of Concept (PoC) demonstrating real-time visual guidance and kinematic feedback for somatic massage strokes used in basal stimulation training. 

Developed as the initial validation prototype for an interactive Spatial Augmented Reality (SAR) medical simulation system at the **Faculty of Informatics and Information Technologies, Slovak University of Technology in Bratislava (FIIT STU)**.

---

## Overview

In nursing education and basal stimulation therapy, the effectiveness of somatic strokes relies heavily on **anatomical direction** and **controlled hand movement velocity**. 

This PoC verifies whether standard, low-cost RGB webcam tracking is sufficient to evaluate therapeutic hand motion in real time without requiring physical sensors, markers, or specialized depth hardware. Using a live video feed, the application tracks the user's hand, renders a directional guidance arrow on screen, and dynamically modulates the arrow's color based on instantaneous movement speed and directional alignment.

### Visual Feedback States

As the user moves their **index fingertip** along the on-screen guidance arrow, the system evaluates velocity along the arrow's vector and provides immediate, progressive color feedback:

| Indicator | State | Kinematic Condition |
| :--- | :--- | :--- |
| **Green** | **Optimal** | Movement follows the target direction within the therapeutic speed range. |
| **Yellow** | **Too Slow** | Hand velocity falls below the minimum target threshold. |
| **Red** | **Too Fast** | Hand velocity exceeds the maximum therapeutic threshold (risk of overstimulation). |

---

## Tech Stack & Dependencies

* **Language:** Python 3.9+
* **Computer Vision & Rendering:** [OpenCV](https://opencv.org/) (`opencv-python`)
* **Hand Tracking:** [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) (`mediapipe`) — extracts 21 3D skeletal hand landmarks per frame.
* **Vector & Kinematic Math:** [NumPy](https://numpy.org/) (`numpy`)

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Set Up a Virtual Environment (Recommended)

```bash
python -m venv venv
# On Linux / macOS / WSL2:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install opencv-python mediapipe numpy
```

> **Note:** Ensure the MediaPipe `hand_landmarker.task` model bundle is placed in the project root directory or update the model path configured in your script.

### 4. Run the Application

```bash
python main.py
```

* Ensure a standard USB or built-in webcam is connected and accessible.
* Position your hand in front of the camera and trace the central guidance arrow with your **index fingertip**.
* Press `q` or `ESC` to exit the application.

---

## How It Works

1. **Video Capture:** OpenCV captures live RGB frames from the webcam.
2. **Landmark Detection:** MediaPipe Hand Landmarker detects the hand and extracts 21 key joint coordinates, isolating the index fingertip landmark (`INDEX_FINGER_TIP`, ID `8`).
3. **Directional Projection & Velocity Estimation:** NumPy computes the displacement vector between consecutive frames ($\Delta \mathbf{x} / \Delta t$) and projects it onto the unit vector of the on-screen guidance arrow to obtain directional speed.
4. **Progressive Rendering:** The guidance arrow is progressively filled and color-coded (Green / Yellow / Red) in real time based on the calculated velocity thresholds.

---

## Project Roadmap (Towards Full SAR System)

This PoC serves as the baseline tracking module for a full-scale **Projector-Camera (ProCam)** medical simulation setup:

- [x] **Phase 1 (Current):** 2D screen-based PoC validating markerless hand tracking and velocity feedback via standard RGB webcam.
- [ ] **Phase 2:** Integration of an RGB-D depth sensor and overhead DLP projector for automated ProCam calibration.
- [ ] **Phase 3:** 3D surface reconstruction and dynamic projection mapping directly onto physical anatomical manikins (e.g., *Lifecast Senior Vivien*).
- [ ] **Phase 4:** Usability and psychomotor skill evaluation with nursing students and educators.

---

## Authors & Affiliation

* **Maroš Dulovec** — Student, FIIT STU (`xdulovecm@stuba.sk`)
* **Juraj Vincúr** — Assistant Professor, HRILab, FIIT STU (`juraj.vincur@stuba.sk`)