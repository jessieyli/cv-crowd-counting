# Crowd Counting: Detecting and Tracking People with YOLOv3 and HOG

Adam Wang, Jessie Li, Joshua Tan

CSCI 1430 Computer Vision Final Project

## Installation

`pip install -r requirements.txt`

## Example of running detection and tracking

General:
`python main.py -if <path to folder containing frames of video> -t -o <path to output video>`

Example:
`python main.py -if mot/ -t -o result.avi`

## Transfer learning

Refer to https://github.com/AdamWang00/yolov3-tf2 for our transfer learning model and weights.