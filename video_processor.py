from typing import Dict, Iterable, List, Optional, Set

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

import supervision as sv
from settings import *

class VideoProcessor:
    def __init__(self,
        source_weights_path: str,
        source_video_path: str,
        target_video_path: str,
        polygons: List[List[tuple]],
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7

    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path

        self.model = YOLO(source_weights_path)
        self.tracker = sv.ByteTrack()

        self.video_info = sv.VideoInfo.from_video_path(source_video_path)

        self.polygons = polygons

        self.zones = [
            sv.PolygonZone(polygon=polygon) for polygon in polygons
        ]

        self.zone_annotators = [
            sv.PolygonZoneAnnotator(
                zone=zone,
                color=colors.by_idx(index),
                thickness=4,
                text_thickness=8,
                text_scale=4
            )
            for index, zone in enumerate(self.zones)
        ]
        
        self.box_annotators = [
            sv.BoxAnnotator(
                color=colors.by_idx(index),
                thickness=4,
            )
            for index in range(len(self.polygons))
        ]
        
        self.trace_annotators = [
            sv.TraceAnnotator(
                color=colors.by_idx(index), 
                position=sv.Position.CENTER, 
                trace_length=100, 
                thickness=2
            ) for index in range(len(self.polygons))
        ]
        
        self.color_annotators = [
            sv.ColorAnnotator(
                color=colors.by_idx(index)
            )
            for index in range(len(self.polygons))
        ]

        self.label_annotator = sv.LabelAnnotator(
            text_color=sv.Color.BLACK
        )

        
    def process_video(self):
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )
        with sv.VideoSink(self.target_video_path, self.video_info) as sink:
            for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                annotated_frame = self.process_frame(frame)
                sink.write_frame(annotated_frame)

    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        annotated_frame = frame.copy()
        labels = [
            f"{tracker_id}" for tracker_id in detections.tracker_id
        ]
        annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)
        
        for zone, zone_annotator, box_annotator, color_annotator, trace_annotator in zip(self.zones, self.zone_annotators, self.box_annotators, self.color_annotators, self.trace_annotators):
            mask = zone.trigger(detections=detections)
            detections_filtered = detections[mask]
            # annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections_filtered)
            annotated_frame = color_annotator.annotate(scene=annotated_frame, detections=detections_filtered)
            annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections_filtered)
            annotated_frame = zone_annotator.annotate(scene=annotated_frame)
        
        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        result = self.model(frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = self.tracker.update_with_detections(detections) #detection with tracker_id

        detections_in_zones = [] #tạo list rỗng chứa các detection trong các zone

        for zone in self.zones:
            mask = zone.trigger(detections=detections) #duyệt và thêm các detection trong từng zone vào list
            detections_in_one_zone = detections[mask]
            detections_in_zones.append(detections_in_one_zone)
        detections = sv.Detections.merge(detections_in_zones)
        return self.annotate_frame(frame, detections) #chỉ annotate các detection trong zone