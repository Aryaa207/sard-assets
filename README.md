# Face Detection & IMU Dashboard

> Real-time face detection with military-style tactical HUD and pan-tilt servo tracking, plus a 9-DOF IMU dashboard with Kalman filtering and 3D visualization — all running on a Raspberry Pi 4B.

---

## Hardware

| Component | Role |
|---|---|
| Raspberry Pi 4B | Main compute |
| Arducam 64MP OWLSight | Camera sensor |
| LSM9DS1 9-DOF IMU | Accelerometer + Gyroscope + Magnetometer |
| PCA9685 16-Ch PWM Board | Servo driver (I2C) |
| 2× MG90S Micro Servos | Pan/tilt gimbal |
| YuNet ONNX Model | Face detection neural network |

---

## Repository Structure

```
.
├── face_detection/
│   ├── setup.sh                  # Automated dependency installer + YuNet model downloader
│   ├── detect.py                 # v0 — Base face detection with basic HUD overlay
│   ├── detection_version1.py     # v1 — Military tactical HUD (phosphor green, scanlines, reticles)
│   ├── detection_version2.py     # v2 — Redesigned to ATAK-style dark UI (amber/white accents)
│   ├── detection_v3.py           # v3 — Fixed BGR/RGB color pipeline (eliminated blue-face bug)
│   ├── detection_v4.py           # v4 — Added PCA9685 servo pan-tilt tracking with PID control
│   ├── Detection_v5.py           # v5 — Servo tuning: MG90S pulse range, safe startup, lower PID gains
│   ├── Detection_v6.py           # v6 — Added real-time debug output for PID tracking diagnostics
│   └── Detection_v7.py           # v7 — Unicode encoding fixes for Raspberry Pi terminal
│
├── imu_dashboard/
│   ├── app.py                    # Flask backend — LSM9DS1 reader + WebSocket server + Kalman filter
│   ├── templates/
│   │   └── index.html            # Frontend — Three.js 3D drone + Chart.js telemetry plots
│   ├── imu_dashboard.py          # Earlier single-file version (Flask + embedded HTML)
│   ├── AppV5.py                  # v5 — Unified single-script dashboard (embedded HTML, one terminal)
│   └── AppV6.py                  # v6 — Final version (reserved word fix, production-ready)
│
└── README.md
```

---

## Face Detection System

### What It Does
Real-time face detection using OpenCV's YuNet model running on a Raspberry Pi 4B with the Arducam 64MP OWLSight camera. Detected faces are tracked by a pan-tilt servo gimbal using PID control, and the feed is rendered through a Pygame-based military tactical HUD.

### Version History

| Version | File | What Changed |
|---|---|---|
| v0 | `detect.py` | Initial optimized pipeline — YuNet on downsampled 320×240, FPS overlay, autofocus |
| v1 | `detection_version1.py` | Full military HUD: phosphor green tint, animated scanlines, corner-bracket reticles, subject IDs with confidence %, threat level indicator, classified watermark |
| v2 | `detection_version2.py` | Redesigned from green to realistic ATAK/C2 aesthetic — dark slate panels, amber/white accents, cleaner typography |
| v3 | `detection_v3.py` | Fixed color pipeline — picamera2 `BGR888` actually outputs RGB, removed incorrect `cvtColor` that caused blue faces |
| v4 | `detection_v4.py` | Integrated PCA9685 servo board for pan-tilt tracking. PID controller locks onto highest-confidence face. Gimbal status panel added to HUD |
| v5 | `Detection_v5.py` | Servo tuning for MG90S: corrected pulse range (600–2400µs), safe slow-sweep startup, reduced PID gains, wider dead zone |
| v6 | `Detection_v6.py` | Added per-frame debug output (face position, pixel error, PID deltas, servo angles) for tracking diagnostics |
| v7 | `Detection_v7.py` | Fixed Unicode em-dash encoding errors that crashed on Raspberry Pi's latin-1 terminal |

### Quick Start

```bash
# On your Raspberry Pi:
chmod +x face_detection/setup.sh
./face_detection/setup.sh

# Run the latest version
cd face_detection
python3 Detection_v7.py
```

### Dependencies
- Python 3.9+
- OpenCV (`opencv-python` with GUI support, NOT headless)
- Pygame
- picamera2
- adafruit-circuitpython-pca9685 (for servo versions)
- adafruit-circuitpython-motor
- pigpio (alternative servo driver)

---

## IMU Dashboard

### What It Does
Reads 9-axis data from an LSM9DS1 IMU sensor over I2C, applies a Kalman filter for attitude estimation, and streams the results to a browser-based dashboard featuring a 3D drone model (Three.js) that rotates in real time and live telemetry plots (Chart.js) showing raw vs filtered roll, pitch, and yaw.

### Version History

| Version | File | What Changed |
|---|---|---|
| Early | `imu_dashboard.py` | Initial Flask app with embedded HTML, basic 3D visualization |
| Split | `app.py` + `templates/index.html` | Separated backend and frontend, added WebSocket streaming, gyro calibration |
| v5 | `AppV5.py` | Unified single-script version — HTML embedded in Python, one terminal to run, one URL to open |
| v6 | `AppV6.py` | Fixed JavaScript reserved word conflict (`top` → `topPlate`), final production version |

### Quick Start

```bash
# Install dependencies
pip3 install flask flask-sock adafruit-circuitpython-lsm9ds1 --break-system-packages

# Run (single command)
cd imu_dashboard
python3 AppV6.py

# Open in browser
# http://localhost:5000
```

### Dashboard Features
- **3D Drone Visualization** — Procedural quadrotor model, quaternion-driven orientation from Kalman output
- **Live Telemetry Plots** — Raw vs Kalman-filtered roll/pitch/yaw, accelerometer and gyroscope readouts
- **Military ATAK Aesthetic** — Dark panels, amber accents, monospace typography, tactical styling
- **Automatic Gyro Calibration** — 2-second bias calibration on startup

---

## Wiring Reference

### Face Detection (PCA9685 Servo Control)
```
Raspberry Pi          PCA9685
─────────────         ───────────────
GPIO 2 (SDA)  ──────► SDA
GPIO 3 (SCL)  ──────► SCL
3.3V          ──────► VCC
GND           ──────► GND

PCA9685 Terminal Block (external 5V power):
V+  ◄──── Arduino 5V (or 5V wall adapter)
GND ◄──── Common ground (shared with Pi GND)

PCA9685 Channels:
Channel 1 ──────► Tilt servo signal
Channel 2 ──────► Pan servo signal
```

### IMU Dashboard (LSM9DS1)
```
Raspberry Pi          LSM9DS1
─────────────         ───────────────
GPIO 2 (SDA)  ──────► SDA
GPIO 3 (SCL)  ──────► SCL
3.3V          ──────► VIN
GND           ──────► GND
```

---

## Tech Stack

- **AI Model**: YuNet (face detection, ONNX format)
- **Computer Vision**: OpenCV 4.x
- **Display**: Pygame (camera HUD), Flask + Three.js + Chart.js (IMU dashboard)
- **Sensor Fusion**: Custom Kalman filter (2-state: angle + gyro bias)
- **Servo Control**: PCA9685 via I2C, PID controller
- **Camera**: picamera2 (libcamera backend)
- **Language**: Python 3

---

## License

MIT
