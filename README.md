## REAL-TIME AUTOMATED TRAFFIC VIOLATION DETECTION USING COMPUTER VISION AND IoT FOR SMART CITIES

**Mithun S, Vignesh A, and G. Padmapriya**\
Department of Computing Technologies, SRM Institute of Science and Technology, Kattankulathur, Tamil Nadu, India\
Email: gpadmapriya@srmist.edu.in

---

## Abstract

Urban road crashes claim over 1.19 million lives annually, with traditional enforcement systems struggling to scale effectively. This paper presents a Dockerized microservices-based traffic violation detection system that processes live CCTV feeds to identify signal-line crossing, illegal parking, and wrong-way driving in real-time. Leveraging OpenCV preprocessing, a transfer-learned MobileNet v1 classifier (94% accuracy), and rule-based violation logic, the system achieves an F₁-score of 0.91 with latency under 150 ms. A PyQt5 GUI enables real-time monitoring and record management, supported by a BCNF-normalized SQLite database. Evaluated under diverse conditions, this solution aligns with UN Sustainable Development Goals 9 and 11, enhancing smart city safety and sustainability.

**Index Terms**—Traffic violation detection, computer vision, microservices, IoT, MobileNet, PyQt5, real-time monitoring, smart city, sustainable transportation.

---

## I. Introduction

Road crashes, a pressing global issue, result in over 1.19 million deaths yearly \[1\]. Traditional traffic enforcement, reliant on human oversight, suffers from limited coverage, inconsistency, and fatigue. Vision-based automation offers a scalable alternative by utilizing existing CCTV networks for continuous, objective monitoring. This study introduces a real-time system targeting three key violations—signal, parking, and directional—using a modular, Dockerized architecture, a lightweight MobileNet v1 classifier, and a user-friendly PyQt5 interface. Key contributions include high accuracy (F₁-score 0.91), low latency (&lt;150 ms), and robust smart city applicability.

---

## II. Methodology

### A. System Architecture

The system employs a Dockerized microservices pipeline:

- **Video Ingestion**: Buffers live CCTV feeds.
- **Preprocessing**: Applies grayscale conversion, Gaussian blur, and background subtraction.
- **Detection & Tracking**: Uses contour detection and IoU-based tracking.
- **Classification**: Employs MobileNet v1 for vehicle identification.
- **Violation Logic**: Detects violations via rule-based algorithms.
- **Database**: Logs events in a BCNF-normalized SQLite schema.
- **GUI**: Provides real-time visualization and management via PyQt5.

### B. Detection and Classification

Frames undergo preprocessing to isolate moving objects, followed by contour-based detection and tracking. MobileNet v1 classifies vehicles with 94% accuracy in 20 ms. Violation logic identifies:

- **Signal Violation**: Crossing stop lines on red.
- **Parking Violation**: Stationary vehicles in no-parking zones.
- **Directional Violation**: Movement against traffic flow.

---

## III. Results

### A. Experimental Setup

- **Hardware**: Intel i7-10750H, NVIDIA RTX 2060, 16 GB RAM.
- **Software**: Ubuntu 22.04, Python 3.12, OpenCV 4.7, TensorFlow 2.12.
- **Dataset**: 5,000 frames across varied conditions.

### B. Performance Metrics

| **Violation** | **Precision** | **Recall** | **F₁-Score** |
| --- | --- | --- | --- |
| Signal | 0.94 | 0.91 | 0.93 |
| Parking | 0.92 | 0.89 | 0.90 |
| Directional | 0.90 | 0.87 | 0.88 |
| **Overall** | **0.92** | **0.89** | **0.91** |

End-to-end latency averages 142 ms, supporting 15 FPS operation.

### C. Environmental Robustness

| **Condition** | **F₁-Score** | **Latency (ms)** |
| --- | --- | --- |
| Daylight | 0.94 | 138 |
| Night | 0.87 | 145 |
| Rain | 0.89 | 149 |
| Dense Traffic | 0.85 | 156 |

---

## IV. Discussion and Conclusion

The system excels in real-time violation detection, achieving high accuracy and low latency. Limitations include sensitivity to glare and occlusions, addressable via future enhancements like multi-camera fusion and adaptive models. Deployment costs range from $1,500-$2,500 per intersection, with scalability supporting 10-15 cameras per server. This work advances smart city safety, aligning with sustainable urban goals.

**References**\
\[1\] World Health Organization, "Global status report on road safety 2023," 2023.\
\[2\] P. Sharma and A. Gupta, "Automated Traffic Violation Detection using YOLOv3 and OpenCV," IEEE Internet of Things Journal, 2019.
