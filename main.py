import pyrealsense2 as rs
import numpy as np
import cv2
import os
from pathlib import Path

# Create output directory for storing frames
output_dir = Path("captured_frames")
output_dir.mkdir(exist_ok=True)

pipe = rs.pipeline()
config = rs.config()

# Enable streams - depth and color streams work together
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# Infrared stream 1 (left IR camera)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)

profile = pipe.start(config)

try:
    for i in range(0, 100):
        frames = pipe.wait_for_frames()

        # Get color frame
        color_frame = frames.get_color_frame()
        if color_frame:
            color_image = np.asanyarray(color_frame.get_data())
            cv2.imwrite(str(output_dir / f"color_{i:04d}.png"), color_image)

        # Get depth frame
        depth_frame = frames.get_depth_frame()
        if depth_frame:
            depth_image = np.asanyarray(depth_frame.get_data())
            # Save raw depth data
            cv2.imwrite(str(output_dir / f"depth_{i:04d}.png"), depth_image)
            # Save colorized depth for visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )
            cv2.imwrite(str(output_dir / f"depth_colormap_{i:04d}.png"), depth_colormap)

        # Get infrared frame
        infrared_frame = frames.get_infrared_frame(1)
        if infrared_frame:
            infrared_image = np.asanyarray(infrared_frame.get_data())
            cv2.imwrite(str(output_dir / f"infrared_{i:04d}.png"), infrared_image)

        print(f"Saved frame {i+1}/100")

finally:
    pipe.stop()
    print(f"\nAll frames saved to {output_dir.absolute()}")
