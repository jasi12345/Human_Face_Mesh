# Realtime Face Mesh Detection

# Overview

This project implements real-time face mesh detection using OpenCV and MediaPipe. It processes an input video file, detects facial landmarks, and overlays a transparent face mesh visualization.

# Features

Uses MediaPipe's FaceMesh model to detect facial landmarks.

Overlays face mesh tessellation, contours, and irises.

Displays both the original video and the face mesh overlay.

Supports processing static images and video streams.

# Requirements

Ensure you have the following dependencies installed before running the script:

pip install opencv-python mediapipe numpy

# Usage

Place your video file in a known directory and update the video=cv2.VideoCapture(r"path_to_video.mp4") line accordingly.

Run the script using:

python face_mesh.py

Press Esc to exit the video window.

