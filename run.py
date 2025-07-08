from video_processor import VideoProcessor
import numpy as np

source_weights_path = '/kaggle/input/traffic_analysis/pytorch/default/1/traffic_analysis.pt'
source_video_path = '/kaggle/input/crossroads/City busy traffic intersection time lapse  _ Free Full HD Video - no copyright.mp4'
target_video_path = '/kaggle/working/output.mp4'
confidence_threshold = 0.3
iou_threshold = 0.7

polygons =  [
                np.array([(0, 171), (704, 171), (704, 392), (0, 392)]), 
                np.array([(0, 693), (708, 693), (708, 914), (0, 914)]),
                np.array([(1239, 177), (1920, 177), (1920, 393), (1239, 393)]),
                np.array([(1244, 692), (1920, 692), (1920, 914), (1244, 914)])
            ]

processor = VideoProcessor(source_weights_path, source_video_path, target_video_path, polygons, confidence_threshold, iou_threshold)
processor.process_video()