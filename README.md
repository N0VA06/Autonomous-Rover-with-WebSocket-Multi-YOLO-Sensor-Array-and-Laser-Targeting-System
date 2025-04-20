# Autonomous Rover

A fully autonomous reconnaissance rover equipped with real-time camouflage detection, object classification, and laser-based threat targeting capabilities.
## Overview and Demo Results
![image](https://github.com/user-attachments/assets/2466cae7-bd0b-43f5-94b4-baeb694645b6)
![image](https://github.com/user-attachments/assets/95d21418-5480-4cfa-a957-f458fa4d548a)
![image](https://github.com/user-attachments/assets/3966c12a-d219-47cc-a5ca-d04006b56c28)
![Screenshot 2025-04-20 165309](https://github.com/user-attachments/assets/6c8b1546-42b2-4fd7-ba83-c1b8d337395b)

## Demo Videos

### Analytics System Demo
[![Analytics Demo](https://img.youtube.com/vi/9rGjHxXuEnQ/0.jpg)](https://youtu.be/9rGjHxXuEnQ)

### Autonomous Navigation & Obstacle Avoidance Demo
[![Rover Navigation Demo](https://img.youtube.com/vi/do77U-R3pFQ/0.jpg)](https://youtu.be/do77U-R3pFQ)

## System Overview

This project integrates several advanced technologies:

- **Sensor Array:** Multiple ultrasonic sensors for environmental mapping and obstacle detection
- **Computer Vision:** Triple YOLOv8 deep learning models with KNN filtering for object classification and camouflage detection
- **Navigation:** A* path planning algorithm for autonomous route finding
- **Targeting System:** Dual-axis servo-based laser module for precise targeting
- **Communication:** WebSocket-based analytics for real-time data processing
- **Hardware:** Raspberry Pi 5 with 256GB external SSD running NAS system

## Performance

- 95.38% detection accuracy for standard uniforms
- 83% detection accuracy for camouflaged personnel
- Mean laser targeting error of 12cm at 50 meters

## Applications

Designed for military surveillance and reconnaissance scenarios where autonomous operation and target identification are required.

## Technologies

- YOLOv8 & YOLOv12
- WebSocket communications
- A* path planning
- Ultrasonic sensor array
- Computer vision
- Raspberry Pi computing
