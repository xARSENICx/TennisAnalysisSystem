# Tennis Video Analysis Suite

This repository contains two main systems for advanced tennis video analysis:

1. **Ball Tracking**: Detects and tracks players and the ball, analyzes court lines, and computes match statistics from video.
2. **Player Pose & Shot Classification**: Uses pose estimation and deep learning to classify tennis shots and visualize player movement.

---

## Table of Contents
- [Ball Tracking](#ball-tracking)
  - [Features](#features)
  - [Output Demo](#output-demo)
  - [How It Works](#how-it-works)
  - [Setup and Installation](#setup-and-installation)
  - [How to Run](#how-to-run)
  - [Project Structure](#project-structure)
- [Player Pose & Shot Classification](#player-pose--shot-classification)
  - [Overview](#overview)
  - [Features](#features-1)
  - [Project Structure](#project-structure-1)
  - [Installation & Requirements](#installation--requirements)
  - [Usage](#usage)
  - [Models](#models)
  - [Notes](#notes)
  - [References](#references)

---

# Ball Tracking

This project leverages computer vision to analyze tennis matches from video footage. It automatically detects players and the ball, tracks their movements, and calculates advanced statistics like player speed, ball speed for each shot, and the number of shots per player.

## Features

- **Player Detection & Tracking:** Uses YOLOv8 to detect and track players on the court throughout the video.
- **Ball Detection & Tracking:** Employs a fine-tuned YOLOv5 model to accurately detect and track the tennis ball.
- **Court Line Detection:** A ResNet-based model identifies the court lines to establish a frame of reference.
- **Shot Detection:** Identifies when a shot is played by analyzing the ball's trajectory.
- **Statistical Analysis:**
    - Calculates the speed of each shot in km/h.
    - Measures the running speed of each player in km/h.
    - Counts the total number of shots for each player.
    - Displays real-time and average stats on the output video.
- **Mini-Court Visualization:** Projects player and ball positions onto a 2D mini-court for a tactical overview.

## Output Demo

Here is a screenshot from a sample output video, showing the trackers and the statistics overlay:

![Screenshot](Ball%20Tracking/output_videos/screenshot.jpeg)

## How It Works

1.  **Video Input:** The program reads an input video file (`.mp4`).
2.  **Object Detection:** It processes each frame to detect players and the ball using the YOLO models.
3.  **Court Recognition:** The court line detector finds the key points of the court in the first frame.
4.  **Coordinate Transformation:** Player and ball coordinates are converted to a standardized mini-court coordinate system.
5.  **Metrics Calculation:** The system calculates distances, speeds, and shots based on the movement of objects between frames.
6.  **Video Output:** The original video is annotated with bounding boxes, a mini-court, and a stats board, then saved as a new video file.

## Setup and Installation

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/tennis_analysis.git
cd tennis_analysis
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
# On Windows, use: venv\Scripts\activate
```

### 3. Install Dependencies

Install all the required Python packages using the `requirements.txt` file.

```bash
pip install -r Ball\ Tracking/requirements.txt
```

### 4. Download Pre-trained Models

You need to download the pre-trained models and place them in the `Ball Tracking/models/` directory.

-   **Trained YOLOv5 model (for ball detection):**
    -   **Link:** [Google Drive](https://drive.google.com/file/d/1UZwiG1jkWgce9lNhxJ2L0NVjX1vGM05U/view?usp=sharing)
    -   **Save as:** `models/yolo5_last.pt`
-   **Trained Court Keypoint model:**
    -   **Link:** [Google Drive](https://drive.google.com/file/d/1QrTOF1ToQ4plsSZbkBs3zOLkVt3MBlta/view?usp=sharing)
    -   **Save as:** `models/keypoints_model.pth`

## How to Run

1.  **Add Input Video:** Place your input video file in the `Ball Tracking/input_videos/` directory. The script is currently set to read `input_videos/input_video.mp4`.
2.  **Execute the Script:** Run the `main.py` script from the root of the project directory.

```bash
python Ball\ Tracking/main.py
```

3.  **Check the Output:** The processed video, `output_video.avi`, will be saved in the `Ball Tracking/output_videos/` directory.

## Project Structure

```
Ball Tracking/
├── input_videos/       # Place input videos here
├── models/             # Contains trained model files (.pt, .pth)
├── output_videos/      # Where the final processed videos are saved
├── trackers/           # Modules for player and ball tracking
├── court_line_detector/  # Module for detecting court lines
├── utils/              # Helper functions for video processing, math, etc.
├── main.py             # Main script to run the analysis
├── requirements.txt    # Project dependencies
└── README.md           # This file
```

---

# Player Pose & Shot Classification

![Shot Detection Example](Player%20Pose/image.png)

This project provides a complete pipeline for analyzing tennis videos using deep learning and computer vision. It enables automatic detection, classification, and visualization of tennis shots (e.g., forehand, backhand, serve, volley, etc.) from raw video footage. The system leverages pose estimation, feature extraction, and recurrent neural networks (RNNs) to classify shots and visualize player movements.

## Overview

The pipeline is designed to process tennis match or practice videos and automatically:

1. **Detect and track the player** using pose estimation (MoveNet).
2. **Extract human pose features** for each frame or shot.
3. **Annotate and save features** for training and evaluation.
4. **Train deep learning models** (single-frame and RNN-based) to classify shot types.
5. **Apply trained models** to new videos for shot detection and classification.
6. **Visualize** the extracted features and classification results.

This enables coaches, analysts, and enthusiasts to gain insights into player technique, shot distribution, and match patterns.

## Features

- **Pose Estimation**: Uses MoveNet (TensorFlow Lite) to extract 17 keypoints per frame.
- **Region of Interest (RoI) Tracking**: Focuses on the player, dynamically updating the area of interest.
- **Shot Feature Extraction**: Associates pose data with annotated shot types.
- **Batch Processing**: Automates feature extraction for large datasets.
- **Deep Learning Models**: Supports both single-frame and RNN-based classifiers.
- **Shot Classification**: Detects and classifies shots in real time or from video files.
- **Visualization**: Animates and exports pose sequences for analysis or presentation.

## Project Structure

```
Player Pose/
├── image.png                        # Shot Detection Example (screengrab from output video)
├── 4.tflite                         # MoveNet pose estimation model (TensorFlow Lite)
├── tennis_rnn_rafa_diy.keras        # Example RNN model (Keras)
├── tennis_rnn_rafa_diy_balanced.keras # Another RNN model (Keras)
├── track_and_classify_with_rnn.py   # RNN-based shot classification on video
├── track_and_classify_frame_by_frame.py # Single-frame shot classification
├── extract_shots_as_features.py     # Extracts pose features for annotated shots
├── extract_human_pose.py            # Extracts and tracks human pose from video
├── batch_extract_shots.py           # Batch feature extraction for datasets
├── visualize_features.py            # Visualizes pose features and creates GIFs
├── SingleFrameShotClassifier.ipynb  # Jupyter notebook for single-frame model
├── RNNShotClassifier.ipynb          # Jupyter notebook for RNN model
└── ...
```

## Installation & Requirements

**Python 3.7+** is recommended.

Install the required packages:

```bash
pip install tensorflow numpy pandas opencv-python imageio tqdm matplotlib seaborn
```

**Note:** You may need to install additional dependencies for Jupyter notebooks and GPU support.

## Usage

### 1. Extract Human Pose

Display and debug pose estimation on a video:

```bash
python Player\ Pose/extract_human_pose.py <video_file> [--debug]
```

- Shows detected keypoints and RoI on each frame.

### 2. Annotate and Extract Shot Features

Extract pose features for each annotated shot in a video:

```bash
python Player\ Pose/extract_shots_as_features.py <video_file> <annotation_csv> <output_dir> [--show] [--debug]
```

- `<annotation_csv>` should contain shot frame indices and types.
- Outputs CSV files with pose features for each shot.

### 3. Batch Feature Extraction

Process an entire directory of videos and annotations:

```bash
python Player\ Pose/batch_extract_shots.py <videos_dir> <annotations_dir> <output_dir> [--show] [--debug] [--continue-on-error]
```

- Matches videos and annotations by filename.
- Calls `extract_shots_as_features.py` for each pair.

### 4. Train Shot Classifiers

Use the provided Jupyter notebooks:

- **SingleFrameShotClassifier.ipynb**: Trains a classifier using pose features from single frames.
- **RNNShotClassifier.ipynb**: Trains a recurrent model (e.g., GRU) on sequences of pose features.

You can customize the training data, model architecture, and evaluation inside the notebooks.

### 5. Shot Classification on Videos

#### RNN-based (sequence) classification:

```bash
python Player\ Pose/track_and_classify_with_rnn.py <video_file> <model_file> [--evaluate <annotation_csv>] [--left-handed] [--threshold <float>] [-f <frame>]
```

- Processes the video with a sliding window of 30 frames (1 second).
- Outputs a video with overlaid shot probabilities and counts.

#### Single-frame classification:

```bash
python Player\ Pose/track_and_classify_frame_by_frame.py <video_file> <model_file> [--evaluate <annotation_csv>] [-f <frame>]
```

- Classifies each frame independently using a single-frame model.

### 6. Visualize Features

Animate and export pose sequences from CSV files:

```bash
python Player\ Pose/visualize_features.py <csv_file(s)> [--gif <output.gif>]
```

- Shows pose animation for each shot.
- Optionally exports as a GIF.

## Models

- **image.png**: Shot Detection Example (screengrab from output video)
- **4.tflite**: MoveNet pose estimation model (TensorFlow Lite).
- **tennis_rnn_rafa_diy.keras**: Example RNN model for shot classification.
- **tennis_rnn_rafa_diy_balanced.keras**: Another RNN model (possibly with balanced classes).

You can train your own models using the provided notebooks and your dataset.

## Notes

- The system expects videos to be annotated with shot types and frame indices for training.
- The pose extraction and classification are designed for single-player, court-level tennis videos.
- The code is modular: you can swap models, add new shot types, or adapt to other sports with similar pose-based analysis.

## References

- [MoveNet: Ultra fast and accurate pose detection model](https://www.tensorflow.org/lite/models/pose_estimation/overview)
- [OpenCV](https://opencv.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)

**Contact:**  
For questions or contributions, please open an issue or submit a pull request.