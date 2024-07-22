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
import threading

track_history = defaultdict(list)

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
    x_center = (bbox[0][0] + bbox[0][2]) / 2
    y_center = (bbox[0][1] + bbox[0][3]) / 2
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

def run(opt):
    weights = opt['weights']
    device = opt['device']
    source = opt['source']
    view_img = opt['view_img']
    save_img = opt['save_img']
    exist_ok = opt['exist_ok']
    classes = opt['classes']
    line_thickness = opt['line_thickness']
    track_thickness = opt['track_thickness']
    region_thickness = opt['region_thickness']

    # Load the model
    model = YOLO(weights)
    device = torch.device('cuda' if torch.cuda.is_available() and device == '0' else 'cpu')
    model.to(device)

    print("Model loaded successfully")

    vid_frame_count = 0

    # Check source path
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")
    print(f"Processing video: {source}")

    # Setup Model
    model = YOLO(weights)
    model.to("cuda" if device == "0" else "cpu")

    # Extract classes names
    names = model.model.names

    # Video setup
    videocapture = cv2.VideoCapture(source)
    if not videocapture.isOpened():
        print(f"Error: Cannot open video file {source}")
        return

    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    # Output setup
    save_dir = increment_path(Path("super-market-video-analysis_output") / "exp", exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (frame_width, frame_height))

    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            print("Finished processing the video")
            break
        vid_frame_count += 1

        # Extract the results
        results = model.track(frame, persist=True, classes=classes, show=False, conf=0.3, iou=0.5, tracker="bytetrack.yaml")

        for result in results:
            for det in result.boxes:
                try:
                    track_id = int(det.id.cpu().numpy().item())
                    bbox = det.xyxy.cpu().numpy()

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
            cv2.rectangle(frame, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, counter_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
    parser.add_argument("--sources", type=str, required=True, nargs='+', help="video file paths")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-img", action="store_true", help="save results")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--line-thickness", type=int, default=2, help="bounding box thickness")
    parser.add_argument("--track-thickness", type=int, default=2, help="Tracking line thickness")
    parser.add_argument("--region-thickness", type=int, default=4, help="Region thickness")

    return parser.parse_args()

def main(opt):
    threads = []
    for source in opt.sources:
        thread_opt = argparse.Namespace(
            weights="model/PeopleDetector.pt",
            device=opt.device,
            source=source,
            view_img=opt.view_img,
            save_img=opt.save_img,
            exist_ok=opt.exist_ok,
            classes=opt.classes,
            line_thickness=opt.line_thickness,
            track_thickness=opt.track_thickness,
            region_thickness=opt.region_thickness
        )
        thread = threading.Thread(target=run, args=(vars(thread_opt),))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
