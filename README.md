# YOLO-Based Object Trajectory Prediction with MLP

This project builds a simple computer vision pipeline for object trajectory prediction.

The pipeline uses a pretrained YOLO model to detect and track people in a video. The tracking results are converted into center-point trajectory data. Then, a small MLP model is trained to predict the next center point based on the previous 10 center points.

## Project Overview

The main workflow is:

1. Run YOLO object detection and tracking on a walking video
2. Save frame-level tracking results to CSV
3. Select the main tracked person
4. Convert the center-point trajectory into sequence data
5. Train an MLP model for short-term trajectory prediction
6. Generate prediction results and demo video
7. Evaluate prediction error

## Project Structure

```text
YOLO-Based Object Trajectory Prediction with MLP/
├── main.ipynb
├── model.py
├── train.py
├── predict.py
├── requirements.txt
├── .gitignore
├── video_data/
│   └── video.mp4
├── models/
│   └── trajectory_model.pth
└── results/
    ├── tracking_results.csv
    ├── main_person_track.csv
    ├── prediction_results.csv
    ├── all_person_prediction.csv
    └── all_person_prediction_scored.csv
```

## Model

The trajectory prediction model is a small MLP.

The input contains the previous 10 center points:

```text
10 frames × 2 coordinates = 20 input features
```

The output is the predicted next center point:

```text
predicted_center_x, predicted_center_y
```

The model architecture is:

```text
Linear(20 → 64)
ReLU
Linear(64 → 32)
ReLU
Linear(32 → 2)
```

## Dataset

The input video is stored in:

```text
video_data/video.mp4
```

YOLO tracking generates:

```text
results/tracking_results.csv
```

The main person trajectory is saved as:

```text
results/main_person_track.csv
```

The CSV contains frame-level trajectory data such as:

```text
frame
track_id
center_x
center_y
box_area
confidence
```

## Prediction Task

The MLP learns the following mapping:

```text
previous 10 center points → next center point
```

This is a short-term trajectory prediction task. The model does not predict long-term human intention. It only estimates the next position based on recent motion history.

## Evaluation

Prediction error is measured by comparing the predicted center point with the actual YOLO tracking center point.

For the main person trajectory, the result was:

```text
Video size: 1080 × 1920
Video diagonal: 2202.91 pixels
Mean error percent: 6.16%
Mean score: 93.84 / 100
Median score: 96.35 / 100
```

The score is calculated by normalizing the pixel error using the video diagonal:

```text
error_percent = pixel_error / video_diagonal × 100
score = 100 - error_percent
```

This is not classification accuracy. It is a normalized trajectory prediction score.

## Prediction Result and Limitation

In this project, track ID 13 was selected as the main person trajectory for MLP-based prediction. The model was trained to use the previous 10 center points to predict the next center point.

For the main person trajectory, the prediction result was strong. The video size was 1080 × 1920, with a video diagonal of 2202.91 pixels. The mean normalized error was 6.16% of the video diagonal, corresponding to a prediction score of 93.84 / 100. The median score was 96.35 / 100.

However, when the same MLP model was applied to other tracked people, the prediction error became much larger. This shows an important limitation of the current model: the MLP learned the motion pattern of the selected main person well, but its generalization ability to different people and different motion directions is limited.

Therefore, the final demo focuses mainly on the main tracked person rather than claiming stable prediction for all detected people.

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the notebook:

```text
main.ipynb
```

The notebook includes:

1. YOLO detection
2. YOLO tracking
3. CSV generation
4. Trajectory cleaning
5. MLP training
6. Prediction visualization
7. Error evaluation

To run the prediction script:

```bash
python predict.py
```

## Notes

- YOLO is used as a pretrained object detection and tracking model.
- The MLP is trained from the extracted trajectory CSV data.
- The saved PyTorch model is stored as a `state_dict` in:

```text
models/trajectory_model.pth
```

- The current MLP is suitable for short-term center-point prediction, not long-term human intention prediction.
