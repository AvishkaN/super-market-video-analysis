import argparse
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors
import pandas as pd
import datetime
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from scipy.spatial.distance import cosine
from filterpy.kalman import KalmanFilter
import threading

track_history = defaultdict(list)
track_features = defaultdict(list)
kalman_filters = {}

current_region = None
counting_regions = [
    {   
        "name": "Region 0",
        "polygon": Polygon([(0, 150), (2080, 150), (2080, 500), (0, 500)]),
        "counts": 0,
        "dragging": False,
        "region_color": (255, 0, 0),
        "text_color": (255, 255, 255),
    },
    {
        "name": "Region 1",
        "polygon": Polygon([(298, 400), (548, 400), (548, 650), (298, 650)]),  
        "counts": 0,
        "dragging": False,
        "region_color": (255, 0, 0),
        "text_color": (255, 255, 255),
    },
    {
        "name": "Region 2",
        "polygon": Polygon([(550, 400), (800, 400), (800, 650), (550, 650)]), 
        "counts": 0,
        "dragging": False,
        "region_color": (0, 255, 0),
        "text_color": (0, 0, 0),
    },
]

# Initialize entry and exit times
def initialize_entry_exit_times():
    times = {} 
    for region in counting_regions:
        times[f"{region['name']} entry_time"] = None 
        times[f"{region['name']} exit_time"] = None 
    return times

# Dictionary to store entry and exit times
entry_exit_times = defaultdict(initialize_entry_exit_times)

# Save entry and exit times to an Excel file
def save_entry_exit_times_to_excel(entry_exit_times):
    data = [                       
        {"ID": key, **value}
        for key, value in entry_exit_times.items()
    ]  
    df = pd.DataFrame(data)
    
    save_dir = increment_path(Path("Time_Excel_Files") / "exp", exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / "entry_exit_times_region_wise.xlsx"

    df.to_excel(file_path, index=False)
    print(f"Entry and exit times have been saved to '{file_path}'")

# Check if the center of the bounding box is within the polygon region
def is_within_region(bbox, polygon):
    # Assuming bbox is a tuple or list of (xmin, ymin, xmax, ymax)
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2
    point = Point(x_center, y_center)
    return polygon.contains(point)  

def mouse_callback(event, x, y, flags, param):
    global current_region

    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            if region["polygon"].contains(Point((x, y))):
                current_region = region
                current_region["dragging"] = True
                current_region["offset_x"] = x
                current_region["offset_y"] = y

    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region["dragging"]:
            dx = x - current_region["offset_x"]
            dy = y - current_region["offset_y"]
            current_region["polygon"] = Polygon(
                [(p[0] + dx, p[1] + dy) for p in current_region["polygon"].exterior.coords]
            )
            current_region["offset_x"] = x
            current_region["offset_y"] = y

    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region["dragging"]:
            current_region["dragging"] = False

# Define a feature extraction model
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Use all layers except the final classification layer

    def forward(self, x):
        features = self.backbone(x)
        return features.view(features.size(0), -1)

# Initialize the feature extractor
feature_extractor = FeatureExtractor().to("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor.eval()  # Set the model to evaluation mode

# Define a function to extract features
def extract_features(image, model):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = preprocess(image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        features = model(image)
    return features.cpu().numpy()

# Initialize a Kalman filter for a new track
def initialize_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([0., 0., 0., 0.])  # Initial state
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])  # State transition matrix
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])  # Measurement function
    kf.P *= 1000.  # Covariance matrix
    kf.R = np.array([[10, 0],
                     [0, 10]])  # Measurement noise
    kf.Q = np.eye(4)  # Process noise
    return kf

# Predict the next position with the Kalman filter
def predict_kalman(track_id):
    kalman = kalman_filters.get(track_id)
    if kalman:
        kalman.predict()
        return kalman.x[:2]
    return None

# Update the Kalman filter with the detected position
def update_kalman(track_id, detection):
    kalman = kalman_filters.get(track_id)
    if kalman:
        kalman.update(detection)
    else:
        kalman = initialize_kalman_filter()
        kalman.x[:2] = detection
        kalman_filters[track_id] = kalman

# Compute the cosine similarity between the current and previous features
def compute_cosine_similarity(current_features, previous_features):
    if not previous_features:
        return float('inf')
    similarities = [cosine(current_features, feature) for feature in previous_features]
    return min(similarities)

def run(weights="model/PeopleDetector.pt", 
        source=None, 
        device="cpu", 
        view_img=False, 
        save_img=False, 
        exist_ok=False, 
        classes=None, 
        line_thickness=2, 
        track_thickness=2, 
        region_thickness=2):
    vid_frame_count = 0

    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    model = YOLO(weights)
    model.to("cuda" if device == "0" else "cpu")

    names = model.model.names

    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    save_dir = increment_path(Path("super-market-video-analysis_output") / "exp", exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (frame_width, frame_height))

    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1

        results = model.track(frame, persist=True, classes=classes, show=False, conf=0.3, iou=0.5, tracker="bytetrack.yaml")

        for result in results:
            for det in result.boxes:
                try:
                    track_id = int(det.id.cpu().numpy().item())
                    bbox = det.xyxy.cpu().numpy().astype(int)[0]

                    roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    if roi.size > 0:
                        current_features = extract_features(roi, feature_extractor)
                        previous_features = track_features[track_id]
                        similarity = compute_cosine_similarity(current_features, previous_features)

                        if similarity > 0.5:
                            continue  # Ignore the current detection as it is not similar to the previous detections

                        track_features[track_id].append(current_features)

                    detection_center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                    update_kalman(track_id, detection_center)

                    for region in counting_regions:
                        within_region = is_within_region(bbox, region["polygon"])

                        if within_region:
                            if entry_exit_times[track_id][f"{region['name']} entry_time"] is None:
                                entry_time = datetime.datetime.now()
                                entry_exit_times[track_id][f"{region['name']} entry_time"] = entry_time
                                save_entry_exit_times_to_excel(entry_exit_times)
                        else:
                            if entry_exit_times[track_id][f"{region['name']} entry_time"] is not None and entry_exit_times[track_id][f"{region['name']} exit_time"] is None:
                                exit_time = datetime.datetime.now()
                                entry_exit_times[track_id][f"{region['name']} exit_time"] = exit_time
                                save_entry_exit_times_to_excel(entry_exit_times)
                except (KeyError, TypeError, ValueError) as e:
                    print(f"Error processing track: {det}, Error: {e}")

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            annotator = Annotator(frame, line_width=line_thickness, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                annotator.box_label(box, f'{track_id} {names[cls]}', color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

                kalman_prediction = predict_kalman(track_id)
                if kalman_prediction is not None:
                    bbox_center = kalman_prediction

                track = track_history[track_id]
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)

                for region in counting_regions:
                    if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                        region["counts"] += 1
                        if track_id not in region.get("tracked_ids", set()):
                            if "tracked_ids" not in region:
                                region["tracked_ids"] = set()
                            region["tracked_ids"].add(track_id)

        for region in counting_regions:
            region_label = str(region["counts"])
            region_color = region["region_color"]
            region_text_color = region["text_color"]

            polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

            text_size, _ = cv2.getTextSize(region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness)
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), region_color, -1)
            cv2.putText(frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, thickness=line_thickness, lineType=cv2.LINE_AA)

            cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)

            counter_text = " | ".join([f"{region['name']}: {region['counts']}" for region in counting_regions])
            text_size, _ = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2)
            text_x = frame_width - text_size[0] - 10
            text_y = text_size[1] + 10
            cv2.rectangle(
                frame,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame, counter_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

        if view_img:
            if vid_frame_count == 1:
                cv2.namedWindow("super-market-video-analysis Region Counter Movable")
                cv2.setMouseCallback("super-market-video-analysis Region Counter Movable", mouse_callback)
            cv2.imshow("super-market-video-analysis Region Counter Movable", frame)

        if save_img:
            video_writer.write(frame)

        for region in counting_regions:
            region["counts"] = 0

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    videocapture.release()
    video_writer.release()
    cv2.destroyAllWindows()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="initial weights path")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--source", type=str, required=True, help="video file path")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-img", action="store_true", help="save results")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--line-thickness", type=int, default=2, help="bounding box thickness")
    parser.add_argument("--track-thickness", type=int, default=2, help="Tracking line thickness")
    parser.add_argument("--region-thickness", type=int, default=4, help="Region thickness")

    return parser.parse_args()

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
