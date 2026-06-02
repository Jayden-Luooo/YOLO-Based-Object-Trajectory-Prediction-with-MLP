import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from model import TrajectoryMLP


# Set paths
video_path = "video_data/video.mp4"
csv_path = "results/main_person_track.csv"
model_path = "models/trajectory_model.pth"
save_path = "results/prediction_results.csv"

# Set parameters
sequence_length = 10
input_size = sequence_length * 2
device = "cuda" if torch.cuda.is_available() else "cpu"

# Get video size
cap = cv2.VideoCapture(video_path)
video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
cap.release()

# Load tracking data
df = pd.read_csv(csv_path)

# Normalize center coordinates
df["center_x_norm"] = df["center_x"] / video_width
df["center_y_norm"] = df["center_y"] / video_height

positions = df[["center_x_norm", "center_y_norm"]].values

# Create sequence data
X = []
y = []
target_frames = []

for i in range(len(positions) - sequence_length):
    X.append(positions[i:i + sequence_length].flatten())
    y.append(positions[i + sequence_length])
    target_frames.append(df["frame"].iloc[i + sequence_length])

X = torch.tensor(np.array(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.float32)

# Use the same time-based split as training
split_idx = int(len(X) * 0.8)

X_test = X[split_idx:].to(device)
y_test = y[split_idx:]

target_frames = np.array(target_frames)
target_frames_test = target_frames[split_idx:]

# Load trained model
model = TrajectoryMLP(
    input_size=input_size,
    hidden_size=64,
    output_size=2
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Make predictions
with torch.inference_mode():
    y_pred = model(X_test).cpu().numpy()

y_true = y_test.numpy()

# Convert normalized coordinates back to pixel coordinates
y_pred_pixel = y_pred.copy()
y_true_pixel = y_true.copy()

y_pred_pixel[:, 0] = y_pred_pixel[:, 0] * video_width
y_pred_pixel[:, 1] = y_pred_pixel[:, 1] * video_height

y_true_pixel[:, 0] = y_true_pixel[:, 0] * video_width
y_true_pixel[:, 1] = y_true_pixel[:, 1] * video_height

# Calculate prediction error in pixels
error = np.sqrt(
    (y_true_pixel[:, 0] - y_pred_pixel[:, 0]) ** 2 +
    (y_true_pixel[:, 1] - y_pred_pixel[:, 1]) ** 2
)

# Save prediction results
Path("results").mkdir(exist_ok=True)

pred_df = pd.DataFrame({
    "frame": target_frames_test,
    "true_x": y_true_pixel[:, 0],
    "true_y": y_true_pixel[:, 1],
    "pred_x": y_pred_pixel[:, 0],
    "pred_y": y_pred_pixel[:, 1],
    "error": error
})

pred_df.to_csv(save_path, index=False)

print(f"Saved prediction results to {save_path}")
print(f"Mean error: {pred_df['error'].mean():.2f} pixels")
print(f"Max error: {pred_df['error'].max():.2f} pixels")
