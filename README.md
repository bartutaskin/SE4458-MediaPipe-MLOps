# Hand Tracking with MediaPipe and MLflow

This project is developed as part of the **SE 4458 - SYSTEMS ARCHITECTURE FOR LARGE-SCALE SYSTEMS** course. It demonstrates real-time hand tracking using **MediaPipe**, integrated with **MLflow** for basic MLOps functionalities such as logging parameters and performance metrics (FPS).

## 📌 Project Structure

├── .gitignore # Standard Git ignore file 
├── HandTracking.py # Main script for hand tracking and MLflow integration 
├── HandTrackingModule.py # (Optional/for extensions) Utility functions for modular tracking 
├── requirements.txt # Required Python libraries


## 🎯 Features

- Real-time hand tracking with **MediaPipe**
- Webcam feed processing with **OpenCV**
- Live **FPS** metric calculation
- **MLflow** integration to log:
  - Hyperparameters (e.g., number of hands, confidence thresholds)
  - FPS as performance metric

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/bartutaskin/SE4458-MediaPipe-MLOps.git
cd SE4458-MediaPipe-MLOps
```
### 2. Set Up Environment

```bash
pip install -r requirements.txt
```

### 3. Run the Project

```bash
python HandTracking.py
python HandTracking.py --max_num_hands 1 --min_detection_confidence 0.7 --min_tracking_confidence 0.7
```

### 3. Run the Project

```bash
mlflow ui
```

## Example Output

The application opens a webcam feed window with hand landmarks drawn in real time and current FPS shown in the top-left corner. To stop the stream, press Q.
