# Tennis Analysis

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

![Screenshot](output_videos/screenshot.jpeg)

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
pip install -r requirements.txt
```

### 4. Download Pre-trained Models

You need to download the pre-trained models and place them in the `models/` directory.

-   **Trained YOLOv5 model (for ball detection):**
    -   **Link:** [Google Drive](https://drive.google.com/file/d/1UZwiG1jkWgce9lNhxJ2L0NVjX1vGM05U/view?usp=sharing)
    -   **Save as:** `models/yolo5_last.pt`
-   **Trained Court Keypoint model:**
    -   **Link:** [Google Drive](https://drive.google.com/file/d/1QrTOF1ToQ4plsSZbkBs3zOLkVt3MBlta/view?usp=sharing)
    -   **Save as:** `models/keypoints_model.pth`

## How to Run

1.  **Add Input Video:** Place your input video file in the `input_videos/` directory. The script is currently set to read `input_videos/input_video.mp4`.
2.  **Execute the Script:** Run the `main.py` script from the root of the project directory.

```bash
python main.py
```

3.  **Check the Output:** The processed video, `output_video.avi`, will be saved in the `output_videos/` directory.

## Project Structure

```
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

## Author

**[Ayush Sah](https://github.com/xArsenicx)**

## TODO

- [ ] Add pose detection model using OpenPose
- [ ] Add inference from ball tracking into the RNN being used for pose detection
- [ ] Work on the website to finish the BTP project.
