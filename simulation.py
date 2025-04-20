import cv2
import numpy as np
import time
from collections import deque
import math

class RoverNavigation:
    def __init__(self, camera_id=0, resolution=(640, 480), goal_coordinates=(100, 100)):
        """
        Initialize the autonomous rover navigation system.
        
        Args:
            camera_id: Camera device ID
            resolution: Camera resolution
            goal_coordinates: Target destination coordinates in 2D grid
        """
        # Camera setup
        self.resolution = resolution
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.initial_goal_coordinates = goal_coordinates  # Store initial goal in grid coordinates
        self.goal_coordinates = goal_coordinates          # Current goal in grid coordinates
        self.grid_size = (120, 120)  # Define grid_size first

        # Add these new parameters for global positioning
        self.grid_to_world_transform = None              # Transform from grid to world coordinates
        self.world_to_grid_transform = None              # Transform from world to grid coordinates
        self.world_goal_position = None                  # Goal in world coordinates
        self.initialize_coordinate_transforms()

        # Navigation parameters
        self.resolution = resolution
        self.goal_coordinates = goal_coordinates
        self.grid_size = (120, 120)  # Increased grid resolution
        self.occupancy_grid = np.zeros(self.grid_size)
        self.confidence_grid = np.zeros(self.grid_size)  # Confidence in obstacle detection
        self.traversability_grid = np.ones(self.grid_size)  # Ease of traversal (1 = easy, 0 = impossible)
        
        # Movement states
        self.current_position = (self.grid_size[0]//2, self.grid_size[1]-10)  # Start at bottom center
        self.current_orientation = 0  # Degrees, 0 = forward
        self.previous_positions = deque(maxlen=20)  # Store previous positions to detect getting stuck
        
        # Object detection model
        self.detection_model = None
        try:
            import torch
            # Try to import YOLO model
            self.detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
            self.detection_model.eval()
            print("Loaded YOLOv5 model successfully")
        except Exception as e:
            print(f"Warning: Could not load object detection model: {e}")
            print("Using simple object detection fallback")
            self.detection_model = None
        
        # Pathfinding parameters
        self.path = []
        self.current_path_index = 0
        self.replanning_count = 0
        
        # Classification parameters with confidence thresholds
        self.static_objects = {
            "wall": 0.6, "rock": 0.65, "tree": 0.7, "building": 0.75, 
            "fence": 0.6, "pole": 0.6, "bench": 0.6
        }
        self.dynamic_objects = {
            "person": 0.7, "animal": 0.65, "car": 0.75, "bicycle": 0.7,
            "motorcycle": 0.7, "truck": 0.75, "dog": 0.65, "cat": 0.65
        }
        
        # Movement commands
        self.cmd_forward = "FORWARD"
        self.cmd_reverse = "REVERSE"
        self.cmd_left = "LEFT"
        self.cmd_right = "RIGHT"
        self.cmd_stop = "STOP"
        self.current_command = self.cmd_stop
        
        # Rover parameters (4-wheel chained with 2 DOF)
        self.wheel_radius = 0.1  # meters
        self.wheel_base = 0.3  # meters
        self.max_velocity = 0.5  # m/s
        self.max_angular_velocity = 0.8  # rad/s
        
        # Statistics
        self.start_time = time.time()
        self.frame_count = 0
        self.detection_times = deque(maxlen=30)
        self.planning_times = deque(maxlen=30)
        
        # Dynamic obstacle tracking
        self.tracked_obstacles = {}
        self.next_obstacle_id = 0
        
        # Pathfinding method
        self.use_potential_fields = False
        self.attractive_gain = 0.5
        self.repulsive_gain = 20.0
        self.potential_field = np.zeros(self.grid_size)

    def initialize_coordinate_transforms(self):
        """Initialize transforms between grid and world coordinate systems"""
        grid_width, grid_height = self.grid_size
        world_width, world_height = 12.0, 12.0  # in meters
        
        # Scale factors
        self.grid_to_world_scale_x = world_width / grid_width
        self.grid_to_world_scale_y = world_height / grid_height
        
        # Convert initial grid goal to world coordinates
        goal_x, goal_y = self.initial_goal_coordinates
        world_goal_x = goal_x * self.grid_to_world_scale_x
        world_goal_y = goal_y * self.grid_to_world_scale_y
        
        # Store world goal
        self.world_goal_position = (world_goal_x, world_goal_y)
        
        print(f"Initialized world goal at ({world_goal_x:.2f}m, {world_goal_y:.2f}m)")

    def capture_frame(self):
        """Capture a frame from the camera"""
        ret, frame = self.camera.read()
        if not ret:
            print("Failed to capture frame")
            return None
        return frame
    
    def detect_objects(self, frame):
        """
        Detect objects in the current frame using the detection model.
        
        Returns:
            List of detected objects with class, confidence, and bounding box
        """
        detection_start = time.time()
        
        if self.detection_model is None:
            # Simple detection fallback - detect edges and contours
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours in the edge map
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            objects = []
            for i, contour in enumerate(contours):
                # Filter small contours
                if cv2.contourArea(contour) < 500:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Simple classification based on shape and location
                if h > 2*w:  # Tall objects are likely people or trees
                    obj_class = "person" if y > frame.shape[0]//2 else "tree"
                elif w > 2*h:  # Wide objects may be walls
                    obj_class = "wall"
                else:
                    obj_class = "rock"
                
                objects.append({
                    'class': obj_class,
                    'confidence': 0.7,  # Fixed confidence for simple detector
                    'bbox': (x, y, x + w, y + h)
                })
        else:
            # Use YOLO for detection
            results = self.detection_model(frame)
            detections = results.pandas().xyxy[0]  # Convert to pandas DataFrame
            
            objects = []
            for _, detection in detections.iterrows():
                obj = {
                    'class': detection['name'],
                    'confidence': detection['confidence'],
                    'bbox': (detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax'])
                }
                objects.append(obj)
        
        self.detection_times.append(time.time() - detection_start)
        return objects
    
    def classify_objects(self, detected_objects):
        """
        Classify detected objects as static or dynamic based on class and confidence.
        
        Args:
            detected_objects: List of detected objects
            
        Returns:
            List of classified objects with type information
        """
        classified_objects = []
        
        for obj in detected_objects:
            obj_class = obj['class']
            confidence = obj['confidence']
            
            # Check if this class is in our static objects dictionary with sufficient confidence
            if obj_class in self.static_objects and confidence >= self.static_objects[obj_class]:
                obj_type = "static"
                classified_objects.append({**obj, 'type': obj_type})
            
            # Check if this class is in our dynamic objects dictionary with sufficient confidence
            elif obj_class in self.dynamic_objects and confidence >= self.dynamic_objects[obj_class]:
                obj_type = "dynamic"
                classified_objects.append({**obj, 'type': obj_type})
                
                # Track this dynamic object for velocity estimation
                self.track_dynamic_object(obj)
        
        return classified_objects

    def update_occupancy_grid(self, classified_objects, frame):
        """
        Update the 2D occupancy grid based on detected objects.
        Also updates the goal position in grid coordinates based on world goal.
        
        Args:
            classified_objects: List of classified objects
            frame: Current camera frame
        """
        # Decay confidence in previous detections
        self.confidence_grid *= 0.9
        
        # Calculate perspective transform to map camera view to 2D grid
        h, w = frame.shape[:2]
        
        # Improved perspective transform - assumes camera mounted at a height and looking forward
        # These points need calibration for specific rover and camera setup
        src_points = np.float32([
            [w*0.1, h*0.8],    # Bottom left
            [w*0.9, h*0.8],    # Bottom right
            [w*0.3, h*0.55],   # Middle left
            [w*0.7, h*0.55],   # Middle right
        ])
        
        dst_width = self.grid_size[0]
        dst_depth = self.grid_size[1] // 2  # Visible area is half the grid
        
        dst_points = np.float32([
            [dst_width*0.1, dst_depth*0.9],  # Bottom left
            [dst_width*0.9, dst_depth*0.9],  # Bottom right
            [dst_width*0.2, dst_depth*0.1],  # Top left
            [dst_width*0.8, dst_depth*0.1],  # Top right
        ])
        
        # Calculate transforms
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Create a new confidence mask for this frame
        new_confidence = np.zeros_like(self.confidence_grid)
        
        # Map each object to grid space
        for obj in classified_objects:
            bbox = obj['bbox']
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            confidence = obj['confidence']
            
            # Get bottom center of object (assuming it's on the ground)
            # For more accuracy, use multiple points along the bottom of the bounding box
            base_points = []
            for i in range(5):
                x_point = x1 + (x2 - x1) * i / 4
                base_points.append([x_point, y2])
            
            object_points = np.array([base_points], dtype=np.float32)
            
            # Transform to grid coordinates
            try:
                grid_points = cv2.perspectiveTransform(object_points, transform_matrix)
                
                # Calculate inflation size based on object class and size
                if obj['type'] == 'static':
                    inflation_base = max(int((x2 - x1 + y2 - y1) / 15), 3)
                else:
                    inflation_base = max(int((x2 - x1 + y2 - y1) / 12), 4)
                    
                    # Use velocity information if available
                    for tracked_id, tracked_obj in self.tracked_obstacles.items():
                        if tracked_obj['class'] == obj['class']:
                            # Check if bounding boxes overlap
                            if (x1 < tracked_obj['bbox'][2] and x2 > tracked_obj['bbox'][0] and
                                y1 < tracked_obj['bbox'][3] and y2 > tracked_obj['bbox'][1]):
                                # Calculate predicted position based on velocity
                                vx, vy = tracked_obj['velocity']
                                if abs(vx) > 5 or abs(vy) > 5:  # Only if moving significantly
                                    # Predict position in next few seconds
                                    for t in range(1, 4):
                                        # Transform predicted screen position to grid
                                        future_x = x1 + vx * t
                                        future_y = y2 + vy * t
                                        future_point = np.array([[[future_x, future_y]]], dtype=np.float32)
                                        try:
                                            future_grid = cv2.perspectiveTransform(future_point, transform_matrix)
                                            grid_x, grid_y = int(future_grid[0][0][0]), int(future_grid[0][0][1])
                                            
                                            # Add to confidence grid with decay over time
                                            if (0 <= grid_x < self.grid_size[0] and 
                                                0 <= grid_y < self.grid_size[1]):
                                                decay = 0.7 ** t  # Confidence decreases with time
                                                inflation = max(inflation_base - t, 2)
                                                for dx in range(-inflation, inflation + 1):
                                                    for dy in range(-inflation, inflation + 1):
                                                        nx, ny = grid_x + dx, grid_y + dy
                                                        if (0 <= nx < self.grid_size[0] and 
                                                            0 <= ny < self.grid_size[1]):
                                                            dist = np.sqrt(dx**2 + dy**2)
                                                            if dist <= inflation:
                                                                weight = (1 - dist/inflation) * confidence * decay
                                                                new_confidence[ny, nx] = max(
                                                                    new_confidence[ny, nx], weight)
                                        except:
                                            pass  # Skip if transformation fails
                
                # Add each transformed point to the grid
                for point in grid_points[0]:
                    grid_x, grid_y = int(point[0]), int(point[1])
                    
                    if 0 <= grid_x < self.grid_size[0] and 0 <= grid_y < self.grid_size[1]:
                        # Add inflation radius based on object type and size
                        inflation = inflation_base
                        
                        for dx in range(-inflation, inflation + 1):
                            for dy in range(-inflation, inflation + 1):
                                nx, ny = grid_x + dx, grid_y + dy
                                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                                    dist = np.sqrt(dx**2 + dy**2)
                                    if dist <= inflation:
                                        # Weight by distance from center and confidence
                                        weight = (1 - dist/inflation) * confidence
                                        new_confidence[ny, nx] = max(new_confidence[ny, nx], weight)
            except:
                pass  # Skip if transformation fails
        
        # Update main confidence grid with new detections
        self.confidence_grid = np.maximum(self.confidence_grid, new_confidence)
        
        # Convert confidence to occupancy (1 = occupied, 0 = free)
        self.occupancy_grid = (self.confidence_grid > 0.6).astype(np.float32)
        
        # Update traversability grid based on occupancy and confidence
        self.traversability_grid = 1.0 - self.confidence_grid
        
        # Ensure completely occupied cells are not traversable
        self.traversability_grid[self.occupancy_grid > 0.9] = 0.0
        
        # Add extra safety margin around occupied cells with distance transform
        dist_transform = cv2.distanceTransform((1 - self.occupancy_grid).astype(np.uint8), 
                                            cv2.DIST_L2, 3)
        
        # Normalize distance transform to [0, 1] and use it to modify traversability
        if dist_transform.max() > 0:
            normalized_dist = dist_transform / dist_transform.max()
            safety_factor = np.clip(normalized_dist * 5, 0, 1)  # Cells within 1/5 of max distance get penalty
            self.traversability_grid *= safety_factor
            
        # Update potential field if enabled
        if self.use_potential_fields:
            self.update_potential_field()
            
        # After updating the occupancy grid, update the goal position
        self.update_goal_in_grid()


    def track_dynamic_object(self, obj):
        """
        Track dynamic objects across frames to estimate movement.
        
        Args:
            obj: Object dictionary with bbox and class
        """
        obj_class = obj['class']
        bbox = obj['bbox']
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        obj_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        # Check if this object matches any tracked object
        matched = False
        for obj_id, tracked in list(self.tracked_obstacles.items()):
            if tracked['class'] == obj_class:
                # Check if centers are close enough
                tracked_center_x = (tracked['bbox'][0] + tracked['bbox'][2]) / 2
                tracked_center_y = (tracked['bbox'][1] + tracked['bbox'][3]) / 2
                tracked_size = (tracked['bbox'][2] - tracked['bbox'][0]) * (tracked['bbox'][3] - tracked['bbox'][1])
                
                # Calculate distance between centers
                distance = np.sqrt((center_x - tracked_center_x)**2 + (center_y - tracked_center_y)**2)
                size_ratio = max(obj_size / tracked_size, tracked_size / obj_size)
                
                # If centers are close and size is similar, it's the same object
                if distance < 50 and size_ratio < 2.0:
                    # Update velocity based on movement
                    dx = center_x - tracked_center_x
                    dy = center_y - tracked_center_y
                    timestamp = time.time()
                    dt = timestamp - tracked['timestamp']
                    
                    if dt > 0:
                        velocity_x = dx / dt
                        velocity_y = dy / dt
                        
                        # Smooth velocity with previous estimate
                        if 'velocity' in tracked:
                            alpha = 0.7  # Smoothing factor
                            velocity_x = alpha * velocity_x + (1 - alpha) * tracked['velocity'][0]
                            velocity_y = alpha * velocity_y + (1 - alpha) * tracked['velocity'][1]
                        
                        self.tracked_obstacles[obj_id] = {
                            'class': obj_class,
                            'bbox': bbox,
                            'timestamp': timestamp,
                            'velocity': (velocity_x, velocity_y),
                            'updates': tracked['updates'] + 1
                        }
                    matched = True
                    break
        
        # If no match, create a new tracked object
        if not matched:
            self.tracked_obstacles[self.next_obstacle_id] = {
                'class': obj_class,
                'bbox': bbox,
                'timestamp': time.time(),
                'velocity': (0, 0),
                'updates': 1
            }
            self.next_obstacle_id += 1
        
        # Remove stale tracks
        current_time = time.time()
        for obj_id in list(self.tracked_obstacles.keys()):
            if current_time - self.tracked_obstacles[obj_id]['timestamp'] > 2.0:
                del self.tracked_obstacles[obj_id]
    
        """
        Update the 2D occupancy grid based on detected objects.
        
        Args:
            classified_objects: List of classified objects
            frame: Current camera frame
        """
        # Decay confidence in previous detections
        self.confidence_grid *= 0.9
        
        # Calculate perspective transform to map camera view to 2D grid
        h, w = frame.shape[:2]
        
        # Improved perspective transform - assumes camera mounted at a height and looking forward
        # These points need calibration for specific rover and camera setup
        src_points = np.float32([
            [w*0.1, h*0.8],    # Bottom left
            [w*0.9, h*0.8],    # Bottom right
            [w*0.3, h*0.55],   # Middle left
            [w*0.7, h*0.55],   # Middle right
        ])
        
        dst_width = self.grid_size[0]
        dst_depth = self.grid_size[1] // 2  # Visible area is half the grid
        
        dst_points = np.float32([
            [dst_width*0.1, dst_depth*0.9],  # Bottom left
            [dst_width*0.9, dst_depth*0.9],  # Bottom right
            [dst_width*0.2, dst_depth*0.1],  # Top left
            [dst_width*0.8, dst_depth*0.1],  # Top right
        ])
        
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Create a new confidence mask for this frame
        new_confidence = np.zeros_like(self.confidence_grid)
        
        # Map each object to grid space
        for obj in classified_objects:
            bbox = obj['bbox']
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            confidence = obj['confidence']
            
            # Get bottom center of object (assuming it's on the ground)
            # For more accuracy, use multiple points along the bottom of the bounding box
            base_points = []
            for i in range(5):
                x_point = x1 + (x2 - x1) * i / 4
                base_points.append([x_point, y2])
            
            object_points = np.array([base_points], dtype=np.float32)
            
            # Transform to grid coordinates
            try:
                grid_points = cv2.perspectiveTransform(object_points, transform_matrix)
                
                # Calculate inflation size based on object class and size
                if obj['type'] == 'static':
                    inflation_base = max(int((x2 - x1 + y2 - y1) / 15), 3)
                else:
                    inflation_base = max(int((x2 - x1 + y2 - y1) / 12), 4)
                    
                    # Use velocity information if available
                    for tracked_id, tracked_obj in self.tracked_obstacles.items():
                        if tracked_obj['class'] == obj['class']:
                            # Check if bounding boxes overlap
                            if (x1 < tracked_obj['bbox'][2] and x2 > tracked_obj['bbox'][0] and
                                y1 < tracked_obj['bbox'][3] and y2 > tracked_obj['bbox'][1]):
                                # Calculate predicted position based on velocity
                                vx, vy = tracked_obj['velocity']
                                if abs(vx) > 5 or abs(vy) > 5:  # Only if moving significantly
                                    # Predict position in next few seconds
                                    for t in range(1, 4):
                                        # Transform predicted screen position to grid
                                        future_x = x1 + vx * t
                                        future_y = y2 + vy * t
                                        future_point = np.array([[[future_x, future_y]]], dtype=np.float32)
                                        try:
                                            future_grid = cv2.perspectiveTransform(future_point, transform_matrix)
                                            grid_x, grid_y = int(future_grid[0][0][0]), int(future_grid[0][0][1])
                                            
                                            # Add to confidence grid with decay over time
                                            if (0 <= grid_x < self.grid_size[0] and 
                                                0 <= grid_y < self.grid_size[1]):
                                                decay = 0.7 ** t  # Confidence decreases with time
                                                inflation = max(inflation_base - t, 2)
                                                for dx in range(-inflation, inflation + 1):
                                                    for dy in range(-inflation, inflation + 1):
                                                        nx, ny = grid_x + dx, grid_y + dy
                                                        if (0 <= nx < self.grid_size[0] and 
                                                            0 <= ny < self.grid_size[1]):
                                                            dist = np.sqrt(dx**2 + dy**2)
                                                            if dist <= inflation:
                                                                weight = (1 - dist/inflation) * confidence * decay
                                                                new_confidence[ny, nx] = max(
                                                                    new_confidence[ny, nx], weight)
                                        except:
                                            pass  # Skip if transformation fails
                
                # Add each transformed point to the grid
                for point in grid_points[0]:
                    grid_x, grid_y = int(point[0]), int(point[1])
                    
                    if 0 <= grid_x < self.grid_size[0] and 0 <= grid_y < self.grid_size[1]:
                        # Add inflation radius based on object type and size
                        inflation = inflation_base
                        
                        for dx in range(-inflation, inflation + 1):
                            for dy in range(-inflation, inflation + 1):
                                nx, ny = grid_x + dx, grid_y + dy
                                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                                    dist = np.sqrt(dx**2 + dy**2)
                                    if dist <= inflation:
                                        # Weight by distance from center and confidence
                                        weight = (1 - dist/inflation) * confidence
                                        new_confidence[ny, nx] = max(new_confidence[ny, nx], weight)
            except:
                pass  # Skip if transformation fails
        
        # Update main confidence grid with new detections
        self.confidence_grid = np.maximum(self.confidence_grid, new_confidence)
        
        # Convert confidence to occupancy (1 = occupied, 0 = free)
        self.occupancy_grid = (self.confidence_grid > 0.6).astype(np.float32)
        
        # Update traversability grid based on occupancy and confidence
        self.traversability_grid = 1.0 - self.confidence_grid
        
        # Ensure completely occupied cells are not traversable
        self.traversability_grid[self.occupancy_grid > 0.9] = 0.0
        
        # Add extra safety margin around occupied cells with distance transform
        dist_transform = cv2.distanceTransform((1 - self.occupancy_grid).astype(np.uint8), 
                                              cv2.DIST_L2, 3)
        
        # Normalize distance transform to [0, 1] and use it to modify traversability
        if dist_transform.max() > 0:
            normalized_dist = dist_transform / dist_transform.max()
            safety_factor = np.clip(normalized_dist * 5, 0, 1)  # Cells within 1/5 of max distance get penalty
            self.traversability_grid *= safety_factor
            
        # Update potential field if enabled
        if self.use_potential_fields:
            self.update_potential_field()
    
    def update_goal_in_grid(self):
        """
        Update the goal position in grid coordinates based on the fixed world goal.
        This ensures the goal remains consistent even as the camera/perspective changes.
        """
        # Get current rover position and orientation
        rover_x, rover_y = self.current_position
        rover_orientation_rad = np.radians(self.current_orientation)
        
        # Convert rover position to world coordinates
        rover_world_x = rover_x * self.grid_to_world_scale_x
        rover_world_y = rover_y * self.grid_to_world_scale_y
        
        # Get world goal
        goal_world_x, goal_world_y = self.world_goal_position
        
        # Calculate relative position of goal to rover
        rel_x = goal_world_x - rover_world_x
        rel_y = goal_world_y - rover_world_y
        
        # Apply rotation based on rover orientation to get the goal in rover's frame
        # We need to rotate the goal's position based on the rover's orientation
        rotated_rel_x = rel_x * np.cos(rover_orientation_rad) + rel_y * np.sin(rover_orientation_rad)
        rotated_rel_y = -rel_x * np.sin(rover_orientation_rad) + rel_y * np.cos(rover_orientation_rad)
        
        # Convert relative position back to grid coordinates
        # The rover is at the center bottom of the visible grid
        base_grid_x = self.grid_size[0] // 2  # Center X
        base_grid_y = self.grid_size[1] - 10  # Near bottom
        
        grid_goal_x = int(base_grid_x + (rotated_rel_x / self.grid_to_world_scale_x))
        grid_goal_y = int(base_grid_y - (rotated_rel_y / self.grid_to_world_scale_y))  # Y is inverted in grid
        
        # Ensure goal is within grid bounds
        grid_goal_x = max(0, min(self.grid_size[0] - 1, grid_goal_x))
        grid_goal_y = max(0, min(self.grid_size[1] - 1, grid_goal_y))
        
        # Update goal in grid coordinates
        self.goal_coordinates = (grid_goal_x, grid_goal_y)
        
        # For debugging
        # print(f"Updated goal in grid: {self.goal_coordinates}, based on world goal: {self.world_goal_position}")
        
    def gradient_descent_pathfinding(self):
        """
        Use gradient descent on the potential field to find a path.
        
        Returns:
            List of grid coordinates representing the path to the goal
        """
        # Start at current position
        path = [self.current_position]
        current = self.current_position
        max_steps = 200  # Prevent infinite loops
        
        # Parameters
        step_size = 1.0
        convergence_distance = 2.0  # Distance to goal to consider path complete
        
        # Calculate gradient manually instead of using Sobel
        # This avoids OpenCV compatibility issues
        potential_padded = np.pad(self.potential_field, 1, mode='edge')
        
        for _ in range(max_steps):
            # Calculate distance to goal
            goal_distance = np.sqrt((current[0] - self.goal_coordinates[0])**2 + 
                                    (current[1] - self.goal_coordinates[1])**2)
            
            # If close enough to goal, end path
            if goal_distance < convergence_distance:
                path.append(self.goal_coordinates)
                break
                
            # Get current position as integers for array indexing
            cx, cy = int(current[0]), int(current[1])
            
            # Check boundaries
            if cx < 0 or cx >= self.grid_size[0] or cy < 0 or cy >= self.grid_size[1]:
                break
                
            # Calculate gradient manually
            # Use padded array to safely handle edges
            px, py = cx + 1, cy + 1  # Convert to padded coordinates
            
            # Calculate x gradient using central difference
            grad_x = (potential_padded[py, px+1] - potential_padded[py, px-1]) / 2.0
            
            # Calculate y gradient using central difference
            grad_y = (potential_padded[py+1, px] - potential_padded[py-1, px]) / 2.0
            
            # Normalize gradient
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            if gradient_magnitude < 1e-6:  # Prevent division by zero
                break
                
            grad_x /= gradient_magnitude
            grad_y /= gradient_magnitude
            
            # Move in the direction of negative gradient (downhill)
            next_x = current[0] - step_size * grad_x
            next_y = current[1] - step_size * grad_y
            
            # Ensure we stay within grid bounds
            next_x = np.clip(next_x, 0, self.grid_size[0] - 1)
            next_y = np.clip(next_y, 0, self.grid_size[1] - 1)
            
            next_position = (int(next_x), int(next_y))
            
            # If we hit an obstacle or are stuck, end path
            if (self.occupancy_grid[int(next_y), int(next_x)] > 0.8 or
                next_position == current):
                break
                
            # Add to path and continue
            if next_position != path[-1]:  # Avoid duplicates
                path.append(next_position)
            
            current = (next_x, next_y)
        
        # Simplify path by removing redundant waypoints
        simplified_path = self.simplify_path(path)
        
        return simplified_path
    
    def simplify_path(self, path):
        """
        Simplify a path by removing unnecessary waypoints.
        Use line-of-sight optimization.
        
        Args:
            path: List of waypoints
            
        Returns:
            Simplified path
        """
        if len(path) < 3:
            return path
            
        simplified = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            # Try to connect current point with furthest visible point
            for i in range(len(path) - 1, current_idx, -1):
                if self.is_line_free(simplified[-1], path[i]):
                    simplified.append(path[i])
                    current_idx = i
                    break
            else:
                # If no point is visible, add the next point
                current_idx += 1
                if current_idx < len(path):
                    simplified.append(path[current_idx])
        
        return simplified
    
    def is_line_free(self, start, end):
        """
        Check if a straight line between two points is free of obstacles.
        
        Args:
            start: Starting point (x, y)
            end: Ending point (x, y)
            
        Returns:
            True if the line is free, False otherwise
        """
        # Bresenham's line algorithm
        x0, y0 = int(start[0]), int(start[1])
        x1, y1 = int(end[0]), int(end[1])
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while x0 != x1 or y0 != y1:
            if 0 <= x0 < self.grid_size[0] and 0 <= y0 < self.grid_size[1]:
                # Check if point is in an obstacle
                if self.occupancy_grid[y0, x0] > 0.6:
                    return False
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        return True
    
    def a_star_pathfinding(self):
        """
        A* algorithm for pathfinding with improved heuristic.
        
        Returns:
            List of grid coordinates representing the path to the goal
        """
        planning_start = time.time()
        
        # Define heuristic function (Euclidean distance with traversability weight)
        def heuristic(a, b):
            # Base Euclidean distance
            base_dist = np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
            return base_dist
        
        # Initialize open and closed sets
        start = self.current_position
        goal = self.goal_coordinates
        
        # If start or goal are in obstacles, find closest free cell
        if (self.occupancy_grid[start[1], start[0]] > 0.8 or 
            self.traversability_grid[start[1], start[0]] < 0.2):
            # Find closest free cell to start
            free_cells = np.where(self.traversability_grid > 0.5)
            if len(free_cells[0]) > 0:
                distances = np.sqrt((free_cells[1] - start[0])**2 + (free_cells[0] - start[1])**2)
                closest_idx = np.argmin(distances)
                start = (free_cells[1][closest_idx], free_cells[0][closest_idx])
        
        if (self.occupancy_grid[goal[1], goal[0]] > 0.8 or 
            self.traversability_grid[goal[1], goal[0]] < 0.2):
            # Find closest free cell to goal
            free_cells = np.where(self.traversability_grid > 0.5)
            if len(free_cells[0]) > 0:
                distances = np.sqrt((free_cells[1] - goal[0])**2 + (free_cells[0] - goal[1])**2)
                closest_idx = np.argmin(distances)
                goal = (free_cells[1][closest_idx], free_cells[0][closest_idx])
        
        open_set = {start}
        closed_set = set()
        
        # Dictionary to store g and f scores
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        # Dictionary to store path
        came_from = {}
        
        # A* search with timeout
        timeout = 0.5  # seconds
        start_time = time.time()
        
        while open_set and time.time() - start_time < timeout:
            # Find node with lowest f_score
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            
            # Check if we reached the goal
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                
                self.planning_times.append(time.time() - planning_start)
                return path[::-1]  # Reverse to get path from start to goal
            
            # Move current node from open to closed set
            open_set.remove(current)
            closed_set.add(current)
            
            # Check neighbors (8-connected grid)
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Skip if out of bounds or in closed set
                if (neighbor[0] < 0 or neighbor[0] >= self.grid_size[0] or
                    neighbor[1] < 0 or neighbor[1] >= self.grid_size[1] or
                    neighbor in closed_set):
                    continue
                
                # Get traversability cost
                neighbor_traversability = self.traversability_grid[neighbor[1], neighbor[0]]
                
                # Skip if obstacle
                if neighbor_traversability < 0.1:
                    continue
                
                # Calculate movement cost (diagonal movement costs more)
                movement_cost = 1.4 if dx != 0 and dy != 0 else 1.0
                
                # Higher cost for difficult terrain
                terrain_cost = movement_cost / max(neighbor_traversability, 0.1)
                
                # Calculate g_score for this neighbor
                tentative_g_score = g_score[current] + terrain_cost
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue
                
                # This path is the best so far
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
        
        # Timeout or no path found - try a different approach
        if time.time() - start_time >= timeout:
            print("A* search timed out, falling back to gradient descent")
            if self.use_potential_fields:
                return self.gradient_descent_pathfinding()
        
        # No path found with A* or timeout - fall back to straight line with obstacle avoidance
        print("No path found, falling back to direct path")
        self.planning_times.append(time.time() - planning_start)
        return self.fallback_direct_path(start, goal)
    
    def fallback_direct_path(self, start, goal):
        """
        Create a simple direct path when A* fails.
        Uses simple obstacle avoidance.
        
        Args:
            start: Starting position
            goal: Goal position
            
        Returns:
            Path as list of waypoints
        """
        # Get intermediate points with obstacle checking
        points = [start]
        current = start
        
        # Maximum number of iterations to prevent infinite loops
        max_iterations = 100
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            # Calculate direction to goal
            dx = goal[0] - current[0]
            dy = goal[1] - current[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # If we're close to the goal, add it and finish
            if distance < 3:
                points.append(goal)
                break
            
            # Normalize direction
            if distance > 0:
                dx /= distance
                dy /= distance
            
            # Step size (smaller steps in complex environments)
            step_size = 3
            
            # Check for obstacles in the path
            next_x = int(current[0] + dx * step_size)
            next_y = int(current[1] + dy * step_size)
            next_pos = (next_x, next_y)
            
            # Make sure we're in bounds
            next_x = max(0, min(next_x, self.grid_size[0] - 1))
            next_y = max(0, min(next_y, self.grid_size[1] - 1))
            next_pos = (next_x, next_y)
            
            # If the direct path is blocked, try to go around the obstacle
            if self.traversability_grid[next_y, next_x] < 0.3:
                # Try different angles to find a clear path
                found_clear_path = False
                for angle_offset in [30, -30, 60, -60, 90, -90]:
                    # Convert direction angle to radians
                    base_angle = np.arctan2(dy, dx)
                    test_angle = base_angle + np.radians(angle_offset)
                    
                    # Calculate new direction
                    test_dx = np.cos(test_angle)
                    test_dy = np.sin(test_angle)
                    
                    # Check position
                    test_x = int(current[0] + test_dx * step_size)
                    test_y = int(current[1] + test_dy * step_size)
                    
                    # Make sure we're in bounds
                    test_x = max(0, min(test_x, self.grid_size[0] - 1))
                    test_y = max(0, min(test_y, self.grid_size[1] - 1))
                    
                    # If this direction is clear, use it
                    if self.traversability_grid[test_y, test_x] > 0.5:
                        next_x, next_y = test_x, test_y
                        next_pos = (next_x, next_y)
                        found_clear_path = True
                        break
                
                # If no clear path found, make a small random move
                if not found_clear_path:
                    # Try to move toward areas with higher traversability
                    best_traversability = 0
                    best_pos = current
                    
                    for nx in range(max(0, current[0]-3), min(self.grid_size[0], current[0]+4)):
                        for ny in range(max(0, current[1]-3), min(self.grid_size[1], current[1]+4)):
                            if (nx, ny) != current:
                                traversability = self.traversability_grid[ny, nx]
                                if traversability > best_traversability:
                                    best_traversability = traversability
                                    best_pos = (nx, ny)
                    
                    next_pos = best_pos
            
            # Add next position to path if it's different from current
            if next_pos != current:
                current = next_pos
                points.append(current)
            else:
                # We're stuck, break the loop
                break
        
        return points
    
    def get_movement_command(self):
        """
        Determine the next movement command based on the current path.
        Returns:
            Movement command string
        """
        # Update goal position in grid before planning
        self.update_goal_in_grid()
        
        # Ensure we haven't reached the goal in world coordinates
        # Calculate distance to goal in world coordinates
        rover_world_x = self.current_position[0] * self.grid_to_world_scale_x
        rover_world_y = self.current_position[1] * self.grid_to_world_scale_y
        goal_world_x, goal_world_y = self.world_goal_position
        
        world_dist_to_goal = np.sqrt((rover_world_x - goal_world_x)**2 + (rover_world_y - goal_world_y)**2)
        
        # If we're close enough to the goal in world coordinates, stop
        if world_dist_to_goal < 0.5:  # 0.5 meters
            return self.cmd_stop
        
        # Store current position for stuck detection
        self.previous_positions.append(self.current_position)
        
        # Check if we're stuck (same position for several iterations)
        if len(self.previous_positions) >= 10:
            recent_positions = list(self.previous_positions)[-10:]
            unique_positions = set(recent_positions)
            if len(unique_positions) <= 2:  # We're oscillating or stuck
                print("Detected stuck condition, forcing replanning")
                self.path = []  # Force replanning
                self.replanning_count += 1
        
        # If no path or we reached the end of current path, recalculate
        if not self.path or self.current_path_index >= len(self.path):
            prev_path = self.path[:] if self.path else []
            
            # Choose pathfinding method
            if self.use_potential_fields and self.replanning_count % 3 == 0:
                self.path = self.gradient_descent_pathfinding()
            else:
                self.path = self.a_star_pathfinding()
                
            # Store path history for visualization
            if prev_path:
                self.path_history.append(prev_path)
                if len(self.path_history) > 5:
                    self.path_history.pop(0)
            
            self.current_path_index = 0
            
            # If still no path, try to move away from obstacles
            if not self.path:
                # Check surroundings for obstacles
                obstacles = []
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        nx, ny = self.current_position[0] + dx, self.current_position[1] + dy
                        if (0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1] and 
                            self.occupancy_grid[ny, nx] > 0.7):
                            obstacles.append((dx, dy))
                
                # Move away from obstacles if possible
                if obstacles:
                    avg_dx = -sum(dx for dx, _ in obstacles) / len(obstacles)
                    avg_dy = -sum(dy for _, dy in obstacles) / len(obstacles)
                    
                    # Determine direction
                    if abs(avg_dx) > abs(avg_dy):
                        return self.cmd_right if avg_dx > 0 else self.cmd_left
                    else:
                        return self.cmd_forward if avg_dy < 0 else self.cmd_reverse
                
                # If stuck, try to turn around
                return self.cmd_right
        
        # Get next waypoint
        next_waypoint = self.path[self.current_path_index]
        dx = next_waypoint[0] - self.current_position[0]
        dy = next_waypoint[1] - self.current_position[1]
        
        # If we're already at this waypoint, advance to next
        if dx == 0 and dy == 0:
            self.current_path_index += 1
            if self.current_path_index < len(self.path):
                next_waypoint = self.path[self.current_path_index]
                dx = next_waypoint[0] - self.current_position[0]
                dy = next_waypoint[1] - self.current_position[1]
            else:
                # End of path reached
                return self.cmd_stop
        
        # Determine direction to move based on orientation
        target_orientation = np.degrees(np.arctan2(dy, dx)) % 360
        orientation_diff = (target_orientation - self.current_orientation) % 360
        
        # If we're roughly pointing in the right direction, move forward
        if orientation_diff < 15 or orientation_diff > 345:
            self.current_position = next_waypoint
            self.current_path_index += 1
            self.current_command = self.cmd_forward
            return self.cmd_forward
        # Turn left
        elif 15 <= orientation_diff < 180:
            self.current_orientation = (self.current_orientation + 15) % 360
            self.current_command = self.cmd_left
            return self.cmd_left
        # Turn right
        else:
            self.current_orientation = (self.current_orientation - 15) % 360
            self.current_command = self.cmd_right
            return self.cmd_right


    def execute_command(self, command):
        """
        Execute movement command by calculating wheel velocities for the rover.
        
        Args:
            command: Movement command string
        
        Returns:
            Tuple of (left_velocity, right_velocity) for differential drive
        """
        # For 4-wheel chained rover with 2 DOF (differential drive)
        left_velocity = 0
        right_velocity = 0
        
        if command == self.cmd_forward:
            left_velocity = self.max_velocity
            right_velocity = self.max_velocity
        elif command == self.cmd_reverse:
            left_velocity = -self.max_velocity
            right_velocity = -self.max_velocity
        elif command == self.cmd_left:
            left_velocity = -self.max_velocity / 2
            right_velocity = self.max_velocity / 2
        elif command == self.cmd_right:
            left_velocity = self.max_velocity / 2
            right_velocity = -self.max_velocity / 2
        
        # In a real system, these velocities would be sent to the rover's motor controllers
        return left_velocity, right_velocity
    
    def visualize_camera_view(self, frame, objects):
        """
        Visualize camera feed with detected objects.
        
        Args:
            frame: Current camera frame
            objects: Detected and classified objects
            
        Returns:
            Visualization image
        """
        # Make a copy to avoid modifying the original
        vis_frame = frame.copy()
        
        # Draw detected objects
        for obj in objects:
            x1, y1, x2, y2 = map(int, obj['bbox'])
            
            # Different colors for static vs dynamic objects
            if obj['type'] == 'static':
                color = (0, 255, 0)  # Green for static
            else:
                color = (0, 0, 255)  # Red for dynamic
            
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Display object class and confidence
            label = f"{obj['class']} ({obj['confidence']:.2f})"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y1 = max(y1, label_size[1])
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1 + baseline - 5), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add tracking information for dynamic objects
        for obj_id, tracked in self.tracked_obstacles.items():
            bbox = tracked['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Draw velocity vector
            vx, vy = tracked['velocity']
            end_x = int(center_x + vx / 5)
            end_y = int(center_y + vy / 5)
            
            cv2.arrowedLine(vis_frame, (center_x, center_y), (end_x, end_y), (255, 0, 255), 2)
            cv2.putText(vis_frame, f"ID:{obj_id}", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Add overlay with performance stats
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        avg_detection_time = sum(self.detection_times) / len(self.detection_times) if self.detection_times else 0
        avg_planning_time = sum(self.planning_times) / len(self.planning_times) if self.planning_times else 0
        
        stats_text = [
            f"FPS: {fps:.1f}",
            f"Detection: {avg_detection_time*1000:.1f}ms",
            f"Planning: {avg_planning_time*1000:.1f}ms",
            f"Replanning: {self.replanning_count}"
        ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(vis_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
        
        return vis_frame
    
    def visualize_occupancy_grid(self):
        """
        Visualize the occupancy grid with color coding.
        Returns:
            Visualization image
        """
        # Create RGB visualization
        grid_vis = np.zeros((self.grid_size[1], self.grid_size[0], 3), dtype=np.uint8)
        
        # Visualize occupancy with gradient based on confidence
        for y in range(self.grid_size[1]):
            for x in range(self.grid_size[0]):
                confidence = self.confidence_grid[y, x]
                if confidence > 0.2:
                    # Red intensity based on confidence
                    red_intensity = int(min(confidence * 255, 255))
                    grid_vis[y, x] = [0, 0, red_intensity]
        
        # Draw path with gradient colors
        if self.path:
            for i, (x, y) in enumerate(self.path):
                if 0 <= y < self.grid_size[1] and 0 <= x < self.grid_size[0]:
                    # Color gradient from green (start) to blue (end)
                    progress = i / len(self.path)
                    color = [
                        int(255 * (1 - progress)),  # B decreases
                        int(255 * (0.5 + 0.5 * progress)), # G increases and stays high
                        int(255 * progress)  # R increases
                    ]
                    grid_vis[y, x] = color
        
        # Draw current position
        cy, cx = self.current_position[1], self.current_position[0]
        if 0 <= cy < self.grid_size[1] and 0 <= cx < self.grid_size[0]:
            # Draw rover as arrow indicating orientation
            angle_rad = np.radians(self.current_orientation)
            end_x = int(cx + 5 * np.cos(angle_rad))
            end_y = int(cy + 5 * np.sin(angle_rad))
            
            # Make sure endpoint is within bounds
            end_x = max(0, min(end_x, self.grid_size[0] - 1))
            end_y = max(0, min(end_y, self.grid_size[1] - 1))
            
            # Draw rover position and orientation
            cv2.circle(grid_vis, (cx, cy), 3, (0, 255, 255), -1)  # Yellow circle
            cv2.line(grid_vis, (cx, cy), (end_x, end_y), (0, 255, 255), 2)  # Direction line
        
        # Draw goal position with a diamond marker to distinguish it
        gy, gx = self.goal_coordinates[1], self.goal_coordinates[0]
        if 0 <= gy < self.grid_size[1] and 0 <= gx < self.grid_size[0]:
            # Draw diamond
            diamond_size = 5
            points = np.array([
                [gx, gy - diamond_size],  # top
                [gx + diamond_size, gy],  # right
                [gx, gy + diamond_size],  # bottom
                [gx - diamond_size, gy]   # left
            ])
            cv2.fillConvexPoly(grid_vis, points, (255, 0, 255))  # Purple diamond
            
            # Add text showing world goal coordinates
            world_x, world_y = self.world_goal_position
            cv2.putText(grid_vis, f"Goal: {world_x:.1f}m, {world_y:.1f}m", 
                    (10, self.grid_size[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add grid lines every 10 cells
        for i in range(0, self.grid_size[0], 10):
            cv2.line(grid_vis, (i, 0), (i, self.grid_size[1]-1), (30, 30, 30), 1)
        
        for i in range(0, self.grid_size[1], 10):
            cv2.line(grid_vis, (0, i), (self.grid_size[0]-1, i), (30, 30, 30), 1)
        
        return grid_vis


    def visualize_traversability(self):
        """
        Visualize the traversability grid.
        
        Returns:
            Visualization image
        """
        # Create a visualization using color gradients
        traversability_vis = np.zeros((self.grid_size[1], self.grid_size[0], 3), dtype=np.uint8)
        
        # Apply color gradient (green = easy, yellow = medium, red = hard)
        for y in range(self.grid_size[1]):
            for x in range(self.grid_size[0]):
                value = self.traversability_grid[y, x]
                
                if value > 0.66:  # Easy terrain
                    traversability_vis[y, x] = [0, int(255 * value), 0]
                elif value > 0.33:  # Medium terrain
                    traversability_vis[y, x] = [0, int(255 * value), int(255 * (1 - value))]
                else:  # Difficult terrain
                    traversability_vis[y, x] = [0, 0, int(255 * (1 - value))]
        
        # Draw current position and goal
        cy, cx = self.current_position[1], self.current_position[0]
        if 0 <= cy < self.grid_size[1] and 0 <= cx < self.grid_size[0]:
            cv2.circle(traversability_vis, (cx, cy), 3, (0, 255, 255), -1)
        
        gy, gx = self.goal_coordinates[1], self.goal_coordinates[0]
        if 0 <= gy < self.grid_size[1] and 0 <= gx < self.grid_size[0]:
            cv2.circle(traversability_vis, (gx, gy), 5, (255, 0, 255), -1)
        
        # Draw path if available
        if self.path:
            for i in range(len(self.path)-1):
                pt1 = (self.path[i][0], self.path[i][1])
                pt2 = (self.path[i+1][0], self.path[i+1][1])
                cv2.line(traversability_vis, pt1, pt2, (255, 255, 255), 1)
        
        return traversability_vis
    
    def visualize(self, frame, objects):
        """
        Create comprehensive visualization with multiple views.
        
        Args:
            frame: Current camera frame
            objects: Detected and classified objects
            
        Returns:
            Combined visualization image
        """
        # Generate individual visualizations
        viz_frame = self.visualize_camera_view(frame, objects)
        viz_occupancy = self.visualize_occupancy_grid()
        viz_traversability = self.visualize_traversability()
        
        # Resize for consistent display
        h_frame, w_frame = viz_frame.shape[:2]
        
        grid_size = 320
        viz_occupancy_resized = cv2.resize(viz_occupancy, (grid_size, grid_size), interpolation=cv2.INTER_NEAREST)
        viz_traversability_resized = cv2.resize(viz_traversability, (grid_size, grid_size), interpolation=cv2.INTER_NEAREST)
        
        # Create layout
        # Top row: Camera view | Occupancy grid
        # Bottom row: Traversability | Command info
        
        # Calculate dimensions
        top_row_height = max(h_frame, grid_size)
        bottom_row_height = grid_size
        
        # Create combined visualization
        width = max(w_frame + grid_size, grid_size * 2)
        height = top_row_height + bottom_row_height
        
        combined = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Place camera view
        combined[:h_frame, :w_frame] = viz_frame
        
        # Place occupancy grid
        combined[:grid_size, w_frame:w_frame+grid_size] = viz_occupancy_resized
        
        # Place traversability grid
        combined[top_row_height:top_row_height+grid_size, :grid_size] = viz_traversability_resized
        
        # Create command info panel
        command_panel = np.ones((grid_size, grid_size, 3), dtype=np.uint8) * 30  # Dark background
        
        # Show current command
        cv2.putText(command_panel, f"Command: {self.current_command}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show wheel velocities
        left_vel, right_vel = self.execute_command(self.current_command)
        cv2.putText(command_panel, f"Left wheels: {left_vel:.2f} m/s", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)
        cv2.putText(command_panel, f"Right wheels: {right_vel:.2f} m/s", (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)
        
        # Show goal information
        cv2.putText(command_panel, f"Goal: ({self.goal_coordinates[0]}, {self.goal_coordinates[1]})", (20, 170), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
        
        # Show pathfinding method
        method = "Potential Fields" if self.use_potential_fields else "A* Search"
        cv2.putText(command_panel, f"Method: {method}", (20, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
        
        # Show orientation
        cv2.putText(command_panel, f"Orientation: {self.current_orientation:.1f}°", (20, 230), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Place command panel
        combined[top_row_height:top_row_height+grid_size, grid_size:grid_size*2] = command_panel
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Camera Feed & Object Detection", (10, 20), font, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, "Occupancy Grid & Path", (w_frame + 10, 20), font, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, "Traversability Map", (10, top_row_height + 20), font, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, "Command & Status", (grid_size + 10, top_row_height + 20), font, 0.6, (255, 255, 255), 2)
        
        return combined
    
    def run(self):
        """Main loop for autonomous navigation"""
        try:
            while True:
                # 1. Capture frame
                frame = self.capture_frame()
                if frame is None:
                    print("Failed to capture frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                # 2. Detect objects
                detected_objects = self.detect_objects(frame)
                
                # 3. Classify objects
                classified_objects = self.classify_objects(detected_objects)
                
                # 4. Update occupancy grid
                self.update_occupancy_grid(classified_objects, frame)
                
                # 5. Get movement command
                command = self.get_movement_command()
                
                # 6. Execute command (in real system, this would send to motor controllers)
                left_vel, right_vel = self.execute_command(command)
                print(f"Command: {command}, Left: {left_vel:.2f} m/s, Right: {right_vel:.2f} m/s")
                
                # 7. Visualize
                visualization = self.visualize(frame, classified_objects)
                cv2.imshow("Autonomous Rover Navigation", visualization)
                
                # Check if we reached the goal
                if command == self.cmd_stop and self.current_position == self.goal_coordinates:
                    print("Goal reached!")
                    # Display success message on visualization
                    cv2.putText(visualization, "GOAL REACHED!", (visualization.shape[1]//2 - 100, visualization.shape[0]//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    cv2.imshow("Autonomous Rover Navigation", visualization)
                    cv2.waitKey(3000)  # Wait 3 seconds to show success
                    break
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Maintain loop rate
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("Navigation stopped by user")
        except Exception as e:
            print(f"Error in navigation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up
            self.camera.release()
            cv2.destroyAllWindows()
import cv2
import numpy as np
import time
from collections import deque
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import math

class MockDetector:    
    def __init__(self):
        self.classes = ["person", "bicycle", "car", "motorbike", "bus", "truck", 
                        "wall", "rock", "tree", "fence", "pole", "building"]
    
    def __call__(self, frame):
        class MockResult:
            def __init__(self, detections):
                self.detections = detections
                
            def pandas(self):
                class MockPandas:
                    def __init__(self, detections):
                        self.xyxy = [detections]
                        
                return MockPandas(self.detections)
        
        # Generate random detections
        height, width = frame.shape[:2]
        detections = []
        
        # Add a few static obstacles
        for _ in range(np.random.randint(1, 4)):
            obj_class = np.random.choice(["wall", "rock", "tree", "building", "fence"])
            x1 = np.random.randint(0, width - 100)
            y1 = np.random.randint(height // 2, height - 50)
            w = np.random.randint(50, 150)
            h = np.random.randint(30, 100)
            
            detections.append({
                'xmin': x1,
                'ymin': y1,
                'xmax': x1 + w,
                'ymax': y1 + h,
                'confidence': np.random.uniform(0.7, 0.95),
                'name': obj_class
            })
        
        # Add a few dynamic obstacles
        for _ in range(np.random.randint(0, 3)):
            obj_class = np.random.choice(["person", "bicycle", "car"])
            x1 = np.random.randint(0, width - 80)
            y1 = np.random.randint(height // 2, height - 100)
            w = np.random.randint(30, 80)
            h = np.random.randint(50, 120)
            
            detections.append({
                'xmin': x1,
                'ymin': y1,
                'xmax': x1 + w,
                'ymax': y1 + h,
                'confidence': np.random.uniform(0.6, 0.9),
                'name': obj_class
            })
            
        return MockResult(detections)


class RoverNavigation:
    def __init__(self, camera_id=0, resolution=(640, 480), goal_coordinates=(100, 100)):
        # Camera setup
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        # Navigation parameters
        self.resolution = resolution
        self.goal_coordinates = goal_coordinates
        self.grid_size = (120, 120)  # Increased grid resolution
        self.occupancy_grid = np.zeros(self.grid_size)
        self.confidence_grid = np.zeros(self.grid_size)  # Confidence in obstacle detection
        self.traversability_grid = np.ones(self.grid_size)  # Ease of traversal (1 = easy, 0 = impossible)
        
        # Movement states
        self.current_position = (self.grid_size[0]//2, self.grid_size[1]-10)  # Start at bottom center
        self.current_orientation = 0  # Degrees, 0 = forward
        self.previous_positions = deque(maxlen=20)  # Store previous positions to detect getting stuck
        
        # Object detection model (YOLO)
        self.detection_model = self.load_detection_model()
        
        # Pathfinding parameters
        self.path = []
        self.current_path_index = 0
        self.path_history = []  # Store previous paths for visualization
        self.replanning_count = 0
        
        # Classification parameters with confidence thresholds
        self.static_objects = {
            "wall": 0.6, "rock": 0.65, "tree": 0.7, "building": 0.75, 
            "fence": 0.6, "pole": 0.6, "bench": 0.6
        }
        self.dynamic_objects = {
            "person": 0.7, "animal": 0.65, "car": 0.75, "bicycle": 0.7,
            "motorcycle": 0.7, "truck": 0.75, "dog": 0.65, "cat": 0.65
        }
        
        # Movement commands
        self.cmd_forward = "FORWARD"
        self.cmd_reverse = "REVERSE"
        self.cmd_left = "LEFT"
        self.cmd_right = "RIGHT"
        self.cmd_stop = "STOP"
        self.current_command = self.cmd_stop
        
        # Rover parameters (4-wheel chained with 2 DOF)
        self.wheel_radius = 0.1  # meters
        self.wheel_base = 0.3  # meters
        self.max_velocity = 0.5  # m/s
        self.max_angular_velocity = 0.8  # rad/s
        
        # Visualization parameters
        self.viz_frame = None
        self.viz_occupancy = None
        self.viz_traversability = None
        self.viz_path = None
        self.viz_velocity = None
        self.viz_potential_field = None
        
        # Statistics for visualization
        self.start_time = time.time()
        self.frame_count = 0
        self.detection_times = deque(maxlen=30)
        self.planning_times = deque(maxlen=30)
        
        # Dynamic obstacle tracking
        self.tracked_obstacles = {}  # Dictionary to track dynamic obstacles over time
        self.next_obstacle_id = 0
        
        # Potential field parameters
        self.use_potential_fields = True
        self.attractive_gain = 0.5
        self.repulsive_gain = 20.0
        self.potential_field = np.zeros(self.grid_size)

    def load_detection_model(self):
        """Load and initialize object detection model (YOLO)"""
        try:
            # Try to load YOLOv5 model using PyTorch
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading YOLOv5: {e}")
            try:
                # Fallback to YOLOv8
                model = torch.hub.load('ultralytics/yolov8', 'yolov8n', pretrained=True, trust_repo=True)
                model.eval()
                return model
            except Exception as e2:
                print(f"Error loading YOLOv8: {e2}")
                # Create a simple mock detector for testing when YOLO fails to load
                print("Creating mock detector for testing")
                return MockDetector()
    
    def capture_frame(self):
        """Capture a frame from the camera"""
        ret, frame = self.camera.read()
        if not ret:
            print("Failed to capture frame")
            return None
        return frame
    
    def detect_objects(self, frame):
        """
        Detect objects in the current frame using the detection model.
        
        Returns:
            List of detected objects with class, confidence, and bounding box
        """
        detection_start = time.time()
        
        results = self.detection_model(frame)
        detections = results.pandas().xyxy[0]  # Convert to pandas DataFrame
        
        objects = []
        for _, detection in detections.iterrows():
            obj = {
                'class': detection['name'],
                'confidence': detection['confidence'],
                'bbox': (detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax'])
            }
            objects.append(obj)
        
        self.detection_times.append(time.time() - detection_start)
        return objects
    
    def classify_objects(self, detected_objects):
        """
        Classify detected objects as static or dynamic based on class and confidence.
        
        Args:
            detected_objects: List of detected objects
            
        Returns:
            List of classified objects with type information
        """
        classified_objects = []
        
        for obj in detected_objects:
            obj_class = obj['class']
            confidence = obj['confidence']
            
            # Check if this class is in our static objects dictionary with sufficient confidence
            if obj_class in self.static_objects and confidence >= self.static_objects[obj_class]:
                obj_type = "static"
                classified_objects.append({**obj, 'type': obj_type})
            
            # Check if this class is in our dynamic objects dictionary with sufficient confidence
            elif obj_class in self.dynamic_objects and confidence >= self.dynamic_objects[obj_class]:
                obj_type = "dynamic"
                classified_objects.append({**obj, 'type': obj_type})
                
                # Track this dynamic object for velocity estimation
                self.track_dynamic_object(obj)
        
        return classified_objects
    
    def track_dynamic_object(self, obj):
        """
        Track dynamic objects across frames to estimate movement.
        
        Args:
            obj: Object dictionary with bbox and class
        """
        obj_class = obj['class']
        bbox = obj['bbox']
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        obj_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        # Check if this object matches any tracked object
        matched = False
        for obj_id, tracked in list(self.tracked_obstacles.items()):
            if tracked['class'] == obj_class:
                # Check if centers are close enough
                tracked_center_x = (tracked['bbox'][0] + tracked['bbox'][2]) / 2
                tracked_center_y = (tracked['bbox'][1] + tracked['bbox'][3]) / 2
                tracked_size = (tracked['bbox'][2] - tracked['bbox'][0]) * (tracked['bbox'][3] - tracked['bbox'][1])
                
                # Calculate distance between centers
                distance = np.sqrt((center_x - tracked_center_x)**2 + (center_y - tracked_center_y)**2)
                size_ratio = max(obj_size / tracked_size, tracked_size / obj_size)
                
                # If centers are close and size is similar, it's the same object
                if distance < 50 and size_ratio < 2.0:
                    # Update velocity based on movement
                    dx = center_x - tracked_center_x
                    dy = center_y - tracked_center_y
                    timestamp = time.time()
                    dt = timestamp - tracked['timestamp']
                    
                    if dt > 0:
                        velocity_x = dx / dt
                        velocity_y = dy / dt
                        
                        # Smooth velocity with previous estimate
                        if 'velocity' in tracked:
                            alpha = 0.7  # Smoothing factor
                            velocity_x = alpha * velocity_x + (1 - alpha) * tracked['velocity'][0]
                            velocity_y = alpha * velocity_y + (1 - alpha) * tracked['velocity'][1]
                        
                        self.tracked_obstacles[obj_id] = {
                            'class': obj_class,
                            'bbox': bbox,
                            'timestamp': timestamp,
                            'velocity': (velocity_x, velocity_y),
                            'updates': tracked['updates'] + 1
                        }
                    matched = True
                    break
        
        # If no match, create a new tracked object
        if not matched:
            self.tracked_obstacles[self.next_obstacle_id] = {
                'class': obj_class,
                'bbox': bbox,
                'timestamp': time.time(),
                'velocity': (0, 0),
                'updates': 1
            }
            self.next_obstacle_id += 1
        
        # Remove stale tracks
        current_time = time.time()
        for obj_id in list(self.tracked_obstacles.keys()):
            if current_time - self.tracked_obstacles[obj_id]['timestamp'] > 2.0:
                del self.tracked_obstacles[obj_id]
    
    def update_occupancy_grid(self, classified_objects, frame):
        """
        Update the 2D occupancy grid based on detected objects.
        
        Args:
            classified_objects: List of classified objects
            frame: Current camera frame
        """
        # Decay confidence in previous detections
        self.confidence_grid *= 0.9
        
        # Calculate perspective transform to map camera view to 2D grid
        h, w = frame.shape[:2]
        
        # Improved perspective transform - assumes camera mounted at a height and looking forward
        # These points need calibration for specific rover and camera setup
        src_points = np.float32([
            [w*0.1, h*0.8],    # Bottom left
            [w*0.9, h*0.8],    # Bottom right
            [w*0.3, h*0.55],   # Middle left
            [w*0.7, h*0.55],   # Middle right
        ])
        
        dst_width = self.grid_size[0]
        dst_depth = self.grid_size[1] // 2  # Visible area is half the grid
        
        dst_points = np.float32([
            [dst_width*0.1, dst_depth*0.9],  # Bottom left
            [dst_width*0.9, dst_depth*0.9],  # Bottom right
            [dst_width*0.2, dst_depth*0.1],  # Top left
            [dst_width*0.8, dst_depth*0.1],  # Top right
        ])
        
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Create a new confidence mask for this frame
        new_confidence = np.zeros_like(self.confidence_grid)
        
        # Map each object to grid space
        for obj in classified_objects:
            bbox = obj['bbox']
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            confidence = obj['confidence']
            
            # Get bottom center of object (assuming it's on the ground)
            # For more accuracy, use multiple points along the bottom of the bounding box
            base_points = []
            for i in range(5):
                x_point = x1 + (x2 - x1) * i / 4
                base_points.append([x_point, y2])
            
            object_points = np.array([base_points], dtype=np.float32)
            
            # Transform to grid coordinates
            try:
                grid_points = cv2.perspectiveTransform(object_points, transform_matrix)
                
                # Calculate inflation size based on object class and size
                if obj['type'] == 'static':
                    inflation_base = max(int((x2 - x1 + y2 - y1) / 15), 3)
                else:
                    inflation_base = max(int((x2 - x1 + y2 - y1) / 12), 4)
                    
                    # Use velocity information if available
                    for tracked_id, tracked_obj in self.tracked_obstacles.items():
                        if tracked_obj['class'] == obj['class']:
                            # Check if bounding boxes overlap
                            if (x1 < tracked_obj['bbox'][2] and x2 > tracked_obj['bbox'][0] and
                                y1 < tracked_obj['bbox'][3] and y2 > tracked_obj['bbox'][1]):
                                # Calculate predicted position based on velocity
                                vx, vy = tracked_obj['velocity']
                                if abs(vx) > 5 or abs(vy) > 5:  # Only if moving significantly
                                    # Predict position in next few seconds
                                    for t in range(1, 4):
                                        # Transform predicted screen position to grid
                                        future_x = x1 + vx * t
                                        future_y = y2 + vy * t
                                        future_point = np.array([[[future_x, future_y]]], dtype=np.float32)
                                        try:
                                            future_grid = cv2.perspectiveTransform(future_point, transform_matrix)
                                            grid_x, grid_y = int(future_grid[0][0][0]), int(future_grid[0][0][1])
                                            
                                            # Add to confidence grid with decay over time
                                            if (0 <= grid_x < self.grid_size[0] and 
                                                0 <= grid_y < self.grid_size[1]):
                                                decay = 0.7 ** t  # Confidence decreases with time
                                                inflation = max(inflation_base - t, 2)
                                                for dx in range(-inflation, inflation + 1):
                                                    for dy in range(-inflation, inflation + 1):
                                                        nx, ny = grid_x + dx, grid_y + dy
                                                        if (0 <= nx < self.grid_size[0] and 
                                                            0 <= ny < self.grid_size[1]):
                                                            dist = np.sqrt(dx**2 + dy**2)
                                                            if dist <= inflation:
                                                                weight = (1 - dist/inflation) * confidence * decay
                                                                new_confidence[ny, nx] = max(
                                                                    new_confidence[ny, nx], weight)
                                        except:
                                            pass  # Skip if transformation fails
                
                # Add each transformed point to the grid
                for point in grid_points[0]:
                    grid_x, grid_y = int(point[0]), int(point[1])
                    
                    if 0 <= grid_x < self.grid_size[0] and 0 <= grid_y < self.grid_size[1]:
                        # Add inflation radius based on object type and size
                        inflation = inflation_base
                        
                        for dx in range(-inflation, inflation + 1):
                            for dy in range(-inflation, inflation + 1):
                                nx, ny = grid_x + dx, grid_y + dy
                                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                                    dist = np.sqrt(dx**2 + dy**2)
                                    if dist <= inflation:
                                        # Weight by distance from center and confidence
                                        weight = (1 - dist/inflation) * confidence
                                        new_confidence[ny, nx] = max(new_confidence[ny, nx], weight)
            except:
                pass  # Skip if transformation fails
        
        # Update main confidence grid with new detections
        self.confidence_grid = np.maximum(self.confidence_grid, new_confidence)
        
        # Convert confidence to occupancy (1 = occupied, 0 = free)
        self.occupancy_grid = (self.confidence_grid > 0.6).astype(np.float32)
        
        # Update traversability grid based on occupancy and confidence
        self.traversability_grid = 1.0 - self.confidence_grid
        
        # Ensure completely occupied cells are not traversable
        self.traversability_grid[self.occupancy_grid > 0.9] = 0.0
        
        # Add extra safety margin around occupied cells with distance transform
        dist_transform = cv2.distanceTransform((1 - self.occupancy_grid).astype(np.uint8), 
                                              cv2.DIST_L2, 3)
        
        # Normalize distance transform to [0, 1] and use it to modify traversability
        if dist_transform.max() > 0:
            normalized_dist = dist_transform / dist_transform.max()
            safety_factor = np.clip(normalized_dist * 5, 0, 1)  # Cells within 1/5 of max distance get penalty
            self.traversability_grid *= safety_factor
            
        # Update potential field if enabled
        if self.use_potential_fields:
            self.update_potential_field()
    
    def update_potential_field(self):
        """
        Update the potential field for navigation using attractive and repulsive forces.
        Attractive force pulls the rover toward the goal.
        Repulsive force pushes the rover away from obstacles.
        """
        # Reset potential field
        self.potential_field = np.zeros(self.grid_size, dtype=np.float32)
        
        # Add attractive potential (goal)
        for y in range(self.grid_size[1]):
            for x in range(self.grid_size[0]):
                # Distance to goal
                goal_dist = math.sqrt((x - self.goal_coordinates[0])**2 + (y - self.goal_coordinates[1])**2)
                # Quadratic attractive potential
                self.potential_field[y, x] += self.attractive_gain * goal_dist**2
        
        # Add repulsive potential (obstacles)
        obstacle_mask = self.occupancy_grid > 0.7
        if np.any(obstacle_mask):
            # Distance transform gives distance to nearest obstacle
            dist_transform = cv2.distanceTransform(
                (1 - obstacle_mask).astype(np.uint8), 
                cv2.DIST_L2, 5
            ).astype(np.float32)  # Ensure float32 type for compatibility
            
            # Add repulsive potential only for cells close to obstacles
            influence_radius = 15  # Cells
            
            # Calculate repulsive potential
            repulsive = np.zeros_like(self.potential_field)
            close_to_obstacle = dist_transform < influence_radius
            
            if np.any(close_to_obstacle):
                # Repulsive potential increases as distance decreases
                # Handle potential division by zero with np.maximum
                safe_dist = np.maximum(dist_transform, 0.1)  # Avoid division by zero
                repulsive[close_to_obstacle] = self.repulsive_gain * (
                    (1.0 / safe_dist[close_to_obstacle]) - 
                    (1.0 / influence_radius)
                )**2
            
            # Add repulsive potential to total potential field
            self.potential_field += repulsive
        
        # Normalize potential field for visualization
        pot_min = self.potential_field.min()
        pot_max = self.potential_field.max()
        if pot_max > pot_min:
            self.potential_field = (self.potential_field - pot_min) / (pot_max - pot_min)
    
    def gradient_descent_pathfinding(self):
        """
        Use gradient descent on the potential field to find a path.
        
        Returns:
            List of grid coordinates representing the path to the goal
        """
        # Start at current position
        path = [self.current_position]
        current = self.current_position
        max_steps = 200  # Prevent infinite loops
        
        # Parameters
        step_size = 1.0
        convergence_distance = 2.0  # Distance to goal to consider path complete
        
        # Calculate gradient manually instead of using Sobel
        # This avoids OpenCV compatibility issues
        potential_padded = np.pad(self.potential_field, 1, mode='edge')
        
        for _ in range(max_steps):
            # Calculate distance to goal
            goal_distance = np.sqrt((current[0] - self.goal_coordinates[0])**2 + 
                                    (current[1] - self.goal_coordinates[1])**2)
            
            # If close enough to goal, end path
            if goal_distance < convergence_distance:
                path.append(self.goal_coordinates)
                break
                
            # Get current position as integers for array indexing
            cx, cy = int(current[0]), int(current[1])
            
            # Check boundaries
            if cx < 0 or cx >= self.grid_size[0] or cy < 0 or cy >= self.grid_size[1]:
                break
                
            # Calculate gradient manually
            # Use padded array to safely handle edges
            px, py = cx + 1, cy + 1  # Convert to padded coordinates
            
            # Calculate x gradient using central difference
            grad_x = (potential_padded[py, px+1] - potential_padded[py, px-1]) / 2.0
            
            # Calculate y gradient using central difference
            grad_y = (potential_padded[py+1, px] - potential_padded[py-1, px]) / 2.0
            
            # Normalize gradient
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            if gradient_magnitude < 1e-6:  # Prevent division by zero
                break
                
            grad_x /= gradient_magnitude
            grad_y /= gradient_magnitude
            
            # Move in the direction of negative gradient (downhill)
            next_x = current[0] - step_size * grad_x
            next_y = current[1] - step_size * grad_y
            
            # Ensure we stay within grid bounds
            next_x = np.clip(next_x, 0, self.grid_size[0] - 1)
            next_y = np.clip(next_y, 0, self.grid_size[1] - 1)
            
            next_position = (int(next_x), int(next_y))
            
            # If we hit an obstacle or are stuck, end path
            if (self.occupancy_grid[int(next_y), int(next_x)] > 0.8 or
                next_position == current):
                break
                
            # Add to path and continue
            if next_position != path[-1]:  # Avoid duplicates
                path.append(next_position)
            
            current = (next_x, next_y)
        
        # Simplify path by removing redundant waypoints
        simplified_path = self.simplify_path(path)
        
        return simplified_path
    
    def simplify_path(self, path):
        """
        Simplify a path by removing unnecessary waypoints.
        Use line-of-sight optimization.
        
        Args:
            path: List of waypoints
            
        Returns:
            Simplified path
        """
        if len(path) < 3:
            return path
            
        simplified = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            # Try to connect current point with furthest visible point
            for i in range(len(path) - 1, current_idx, -1):
                if self.is_line_free(simplified[-1], path[i]):
                    simplified.append(path[i])
                    current_idx = i
                    break
            else:
                # If no point is visible, add the next point
                current_idx += 1
                if current_idx < len(path):
                    simplified.append(path[current_idx])
        
        return simplified
    
    def is_line_free(self, start, end):
        """
        Check if a straight line between two points is free of obstacles.
        
        Args:
            start: Starting point (x, y)
            end: Ending point (x, y)
            
        Returns:
            True if the line is free, False otherwise
        """
        # Bresenham's line algorithm
        x0, y0 = int(start[0]), int(start[1])
        x1, y1 = int(end[0]), int(end[1])
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while x0 != x1 or y0 != y1:
            if 0 <= x0 < self.grid_size[0] and 0 <= y0 < self.grid_size[1]:
                # Check if point is in an obstacle
                if self.occupancy_grid[y0, x0] > 0.6:
                    return False
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        return True
    
    def a_star_pathfinding(self):
        """
        A* algorithm for pathfinding with improved heuristic.
        
        Returns:
            List of grid coordinates representing the path to the goal
        """
        planning_start = time.time()
        
        # Define heuristic function (Euclidean distance with traversability weight)
        def heuristic(a, b):
            # Base Euclidean distance
            base_dist = np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
            return base_dist
        
        # Initialize open and closed sets
        start = self.current_position
        goal = self.goal_coordinates
        
        # If start or goal are in obstacles, find closest free cell
        if (self.occupancy_grid[start[1], start[0]] > 0.8 or 
            self.traversability_grid[start[1], start[0]] < 0.2):
            # Find closest free cell to start
            free_cells = np.where(self.traversability_grid > 0.5)
            if len(free_cells[0]) > 0:
                distances = np.sqrt((free_cells[1] - start[0])**2 + (free_cells[0] - start[1])**2)
                closest_idx = np.argmin(distances)
                start = (free_cells[1][closest_idx], free_cells[0][closest_idx])
        
        if (self.occupancy_grid[goal[1], goal[0]] > 0.8 or 
            self.traversability_grid[goal[1], goal[0]] < 0.2):
            # Find closest free cell to goal
            free_cells = np.where(self.traversability_grid > 0.5)
            if len(free_cells[0]) > 0:
                distances = np.sqrt((free_cells[1] - goal[0])**2 + (free_cells[0] - goal[1])**2)
                closest_idx = np.argmin(distances)
                goal = (free_cells[1][closest_idx], free_cells[0][closest_idx])
        
        open_set = {start}
        closed_set = set()
        
        # Dictionary to store g and f scores
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        # Dictionary to store path
        came_from = {}
        
        # A* search with timeout
        timeout = 0.5  # seconds
        start_time = time.time()
        
        while open_set and time.time() - start_time < timeout:
            # Find node with lowest f_score
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            
            # Check if we reached the goal
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                
                self.planning_times.append(time.time() - planning_start)
                return path[::-1]  # Reverse to get path from start to goal
            
            # Move current node from open to closed set
            open_set.remove(current)
            closed_set.add(current)
            
            # Check neighbors (8-connected grid)
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Skip if out of bounds or in closed set
                if (neighbor[0] < 0 or neighbor[0] >= self.grid_size[0] or
                    neighbor[1] < 0 or neighbor[1] >= self.grid_size[1] or
                    neighbor in closed_set):
                    continue
                
                # Get traversability cost
                neighbor_traversability = self.traversability_grid[neighbor[1], neighbor[0]]
                
                # Skip if obstacle
                if neighbor_traversability < 0.1:
                    continue
                
                # Calculate movement cost (diagonal movement costs more)
                movement_cost = 1.4 if dx != 0 and dy != 0 else 1.0
                
                # Higher cost for difficult terrain
                terrain_cost = movement_cost / max(neighbor_traversability, 0.1)
                
                # Calculate g_score for this neighbor
                tentative_g_score = g_score[current] + terrain_cost
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue
                
                # This path is the best so far
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
        
        # Timeout or no path found - try a different approach
        if time.time() - start_time >= timeout:
            print("A* search timed out, falling back to gradient descent")
            if self.use_potential_fields:
                return self.gradient_descent_pathfinding()
        
        # No path found with A* or timeout - fall back to straight line with obstacle avoidance
        print("No path found, falling back to direct path")
        self.planning_times.append(time.time() - planning_start)
        return self.fallback_direct_path(start, goal)
    
    def fallback_direct_path(self, start, goal):
        """
        Create a simple direct path when A* fails.
        Uses simple obstacle avoidance.
        
        Args:
            start: Starting position
            goal: Goal position
            
        Returns:
            Path as list of waypoints
        """
        # Get intermediate points with obstacle checking
        points = [start]
        current = start
        
        # Maximum number of iterations to prevent infinite loops
        max_iterations = 100
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            # Calculate direction to goal
            dx = goal[0] - current[0]
            dy = goal[1] - current[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            # If we're close to the goal, add it and finish
            if distance < 3:
                points.append(goal)
                break
            
            # Normalize direction
            if distance > 0:
                dx /= distance
                dy /= distance
            
            # Step size (smaller steps in complex environments)
            step_size = 3
            
            # Check for obstacles in the path
            next_x = int(current[0] + dx * step_size)
            next_y = int(current[1] + dy * step_size)
            next_pos = (next_x, next_y)
            
            # Make sure we're in bounds
            next_x = max(0, min(next_x, self.grid_size[0] - 1))
            next_y = max(0, min(next_y, self.grid_size[1] - 1))
            next_pos = (next_x, next_y)
            
            # If the direct path is blocked, try to go around the obstacle
            if self.traversability_grid[next_y, next_x] < 0.3:
                # Try different angles to find a clear path
                found_clear_path = False
                for angle_offset in [30, -30, 60, -60, 90, -90]:
                    # Convert direction angle to radians
                    base_angle = np.arctan2(dy, dx)
                    test_angle = base_angle + np.radians(angle_offset)
                    
                    # Calculate new direction
                    test_dx = np.cos(test_angle)
                    test_dy = np.sin(test_angle)
                    
                    # Check position
                    test_x = int(current[0] + test_dx * step_size)
                    test_y = int(current[1] + test_dy * step_size)
                    
                    # Make sure we're in bounds
                    test_x = max(0, min(test_x, self.grid_size[0] - 1))
                    test_y = max(0, min(test_y, self.grid_size[1] - 1))
                    
                    # If this direction is clear, use it
                    if self.traversability_grid[test_y, test_x] > 0.5:
                        next_x, next_y = test_x, test_y
                        next_pos = (next_x, next_y)
                        found_clear_path = True
                        break
                
                # If no clear path found, make a small random move
                if not found_clear_path:
                    # Try to move toward areas with higher traversability
                    best_traversability = 0
                    best_pos = current
                    
                    for nx in range(max(0, current[0]-3), min(self.grid_size[0], current[0]+4)):
                        for ny in range(max(0, current[1]-3), min(self.grid_size[1], current[1]+4)):
                            if (nx, ny) != current:
                                traversability = self.traversability_grid[ny, nx]
                                if traversability > best_traversability:
                                    best_traversability = traversability
                                    best_pos = (nx, ny)
                    
                    next_pos = best_pos
            
            # Add next position to path if it's different from current
            if next_pos != current:
                current = next_pos
                points.append(current)
            else:
                # We're stuck, break the loop
                break
        
        return points
    
    def get_movement_command(self):
        """
        Determine the next movement command based on the current path.
        
        Returns:
            Movement command string
        """
        # If we reached the goal, stop
        if self.current_position == self.goal_coordinates:
            return self.cmd_stop
        
        # Store current position for stuck detection
        self.previous_positions.append(self.current_position)
        
        # Check if we're stuck (same position for several iterations)
        if len(self.previous_positions) >= 10:
            recent_positions = list(self.previous_positions)[-10:]
            unique_positions = set(recent_positions)
            if len(unique_positions) <= 2:  # We're oscillating or stuck
                print("Detected stuck condition, forcing replanning")
                self.path = []  # Force replanning
                self.replanning_count += 1
        
        # If no path or we reached the end of current path, recalculate
        if not self.path or self.current_path_index >= len(self.path):
            prev_path = self.path[:] if self.path else []
            
            # Choose pathfinding method
            if self.use_potential_fields and self.replanning_count % 3 == 0:
                self.path = self.gradient_descent_pathfinding()
            else:
                self.path = self.a_star_pathfinding()
                
            # Store path history for visualization
            if prev_path:
                self.path_history.append(prev_path)
                if len(self.path_history) > 5:
                    self.path_history.pop(0)
            
            self.current_path_index = 0
            
            # If still no path, try to move away from obstacles
            if not self.path:
                # Check surroundings for obstacles
                obstacles = []
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        nx, ny = self.current_position[0] + dx, self.current_position[1] + dy
                        if (0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1] and 
                            self.occupancy_grid[ny, nx] > 0.7):
                            obstacles.append((dx, dy))
                
                # Move away from obstacles if possible
                if obstacles:
                    avg_dx = -sum(dx for dx, _ in obstacles) / len(obstacles)
                    avg_dy = -sum(dy for _, dy in obstacles) / len(obstacles)
                    
                    # Determine direction
                    if abs(avg_dx) > abs(avg_dy):
                        return self.cmd_right if avg_dx > 0 else self.cmd_left
                    else:
                        return self.cmd_forward if avg_dy < 0 else self.cmd_reverse
                
                # If stuck, try to turn around
                return self.cmd_right
        
        # Get next waypoint
        next_waypoint = self.path[self.current_path_index]
        dx = next_waypoint[0] - self.current_position[0]
        dy = next_waypoint[1] - self.current_position[1]
        
        # If we're already at this waypoint, advance to next
        if dx == 0 and dy == 0:
            self.current_path_index += 1
            if self.current_path_index < len(self.path):
                next_waypoint = self.path[self.current_path_index]
                dx = next_waypoint[0] - self.current_position[0]
                dy = next_waypoint[1] - self.current_position[1]
            else:
                # End of path reached
                return self.cmd_stop
        
        # Determine direction to move based on orientation
        target_orientation = np.degrees(np.arctan2(dy, dx)) % 360
        orientation_diff = (target_orientation - self.current_orientation) % 360
        
        # If we're roughly pointing in the right direction, move forward
        if orientation_diff < 15 or orientation_diff > 345:
            self.current_position = next_waypoint
            self.current_path_index += 1
            self.current_command = self.cmd_forward
            return self.cmd_forward
        # Turn left
        elif 15 <= orientation_diff < 180:
            self.current_orientation = (self.current_orientation + 15) % 360
            self.current_command = self.cmd_left
            return self.cmd_left
        # Turn right
        else:
            self.current_orientation = (self.current_orientation - 15) % 360
            self.current_command = self.cmd_right
            return self.cmd_right
    
    def execute_command(self, command):
        """
        Execute movement command by calculating wheel velocities for the rover.
        
        Args:
            command: Movement command string
        
        Returns:
            Tuple of (left_velocity, right_velocity) for differential drive
        """
        # For 4-wheel chained rover with 2 DOF (differential drive)
        left_velocity = 0
        right_velocity = 0
        
        if command == self.cmd_forward:
            left_velocity = self.max_velocity
            right_velocity = self.max_velocity
        elif command == self.cmd_reverse:
            left_velocity = -self.max_velocity
            right_velocity = -self.max_velocity
        elif command == self.cmd_left:
            left_velocity = -self.max_velocity / 2
            right_velocity = self.max_velocity / 2
        elif command == self.cmd_right:
            left_velocity = self.max_velocity / 2
            right_velocity = -self.max_velocity / 2
        
        # In a real system, these velocities would be sent to the rover's motor controllers
        return left_velocity, right_velocity
    
    def visualize_camera_view(self, frame, objects):
        """
        Visualize camera feed with detected objects.
        
        Args:
            frame: Current camera frame
            objects: Detected and classified objects
            
        Returns:
            Visualization image
        """
        # Make a copy to avoid modifying the original
        vis_frame = frame.copy()
        
        # Draw detected objects
        for obj in objects:
            x1, y1, x2, y2 = map(int, obj['bbox'])
            
            # Different colors for static vs dynamic objects
            if obj['type'] == 'static':
                color = (0, 255, 0)  # Green for static
            else:
                color = (0, 0, 255)  # Red for dynamic
            
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Display object class and confidence
            label = f"{obj['class']} ({obj['confidence']:.2f})"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y1 = max(y1, label_size[1])
            cv2.rectangle(vis_frame, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1 + baseline - 5), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add tracking information for dynamic objects
        for obj_id, tracked in self.tracked_obstacles.items():
            bbox = tracked['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Draw velocity vector
            vx, vy = tracked['velocity']
            end_x = int(center_x + vx / 5)
            end_y = int(center_y + vy / 5)
            
            cv2.arrowedLine(vis_frame, (center_x, center_y), (end_x, end_y), (255, 0, 255), 2)
            cv2.putText(vis_frame, f"ID:{obj_id}", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Add overlay with performance stats
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        avg_detection_time = sum(self.detection_times) / len(self.detection_times) if self.detection_times else 0
        avg_planning_time = sum(self.planning_times) / len(self.planning_times) if self.planning_times else 0
        
        stats_text = [
            f"FPS: {fps:.1f}",
            f"Detection: {avg_detection_time*1000:.1f}ms",
            f"Planning: {avg_planning_time*1000:.1f}ms",
            f"Replanning: {self.replanning_count}"
        ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(vis_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
        
        return vis_frame
    
    def visualize_occupancy_grid(self):
        """
        Visualize the occupancy grid with color coding.
        
        Returns:
            Visualization image
        """
        # Create RGB visualization
        grid_vis = np.zeros((self.grid_size[1], self.grid_size[0], 3), dtype=np.uint8)
        
        # Visualize occupancy with gradient based on confidence
        for y in range(self.grid_size[1]):
            for x in range(self.grid_size[0]):
                confidence = self.confidence_grid[y, x]
                if confidence > 0.2:
                    # Red intensity based on confidence
                    red_intensity = int(min(confidence * 255, 255))
                    grid_vis[y, x] = [0, 0, red_intensity]
        
        # Draw path with gradient colors to show direction
        if self.path:
            for i, (x, y) in enumerate(self.path):
                if 0 <= y < self.grid_size[1] and 0 <= x < self.grid_size[0]:
                    # Color gradient from green (start) to blue (end)
                    progress = i / len(self.path)
                    color = [
                        int(255 * (1 - progress)),  # B decreases
                        int(255 * (0.5 + 0.5 * progress)), # G increases and stays high
                        int(255 * progress)  # R increases
                    ]
                    grid_vis[y, x] = color
        
        # Draw path history with fading intensity
        for path_idx, old_path in enumerate(self.path_history):
            intensity = 100 + (155 * path_idx / len(self.path_history))
            for x, y in old_path:
                if 0 <= y < self.grid_size[1] and 0 <= x < self.grid_size[0]:
                    # Don't overwrite current path
                    if grid_vis[y, x].any() == 0:
                        grid_vis[y, x] = [intensity, intensity, intensity]  # Gray with increasing intensity
        
        # Draw current position
        cy, cx = self.current_position[1], self.current_position[0]
        if 0 <= cy < self.grid_size[1] and 0 <= cx < self.grid_size[0]:
            # Draw rover as arrow indicating orientation
            angle_rad = np.radians(self.current_orientation)
            end_x = int(cx + 5 * np.cos(angle_rad))
            end_y = int(cy + 5 * np.sin(angle_rad))
            
            # Make sure endpoint is within bounds
            end_x = max(0, min(end_x, self.grid_size[0] - 1))
            end_y = max(0, min(end_y, self.grid_size[1] - 1))
            
            # Draw rover position and orientation
            cv2.circle(grid_vis, (cx, cy), 3, (0, 255, 255), -1)  # Yellow circle
            cv2.line(grid_vis, (cx, cy), (end_x, end_y), (0, 255, 255), 2)  # Direction line
        
        # Draw goal position
        gy, gx = self.goal_coordinates[1], self.goal_coordinates[0]
        if 0 <= gy < self.grid_size[1] and 0 <= gx < self.grid_size[0]:
            cv2.circle(grid_vis, (gx, gy), 5, (255, 0, 255), -1)  # Purple circle
            cv2.circle(grid_vis, (gx, gy), 7, (255, 0, 255), 1)   # Purple outline
        
        # Add grid lines every 10 cells
        for i in range(0, self.grid_size[0], 10):
            cv2.line(grid_vis, (i, 0), (i, self.grid_size[1]-1), (30, 30, 30), 1)
        
        for i in range(0, self.grid_size[1], 10):
            cv2.line(grid_vis, (0, i), (self.grid_size[0]-1, i), (30, 30, 30), 1)
        
        return grid_vis
    
    def visualize_traversability(self):
        """
        Visualize the traversability grid.
        
        Returns:
            Visualization image
        """
        # Create colormap for traversability (green = easy, red = hard)
        cm = LinearSegmentedColormap.from_list('traversability', ['red', 'yellow', 'green'])
        
        # Apply colormap to traversability grid
        traversability_colored = cm(self.traversability_grid)
        traversability_vis = (traversability_colored[:, :, :3] * 255).astype(np.uint8)
        
        # Draw current position and goal
        cy, cx = self.current_position[1], self.current_position[0]
        if 0 <= cy < self.grid_size[1] and 0 <= cx < self.grid_size[0]:
            cv2.circle(traversability_vis, (cx, cy), 3, (0, 255, 255), -1)
        
        gy, gx = self.goal_coordinates[1], self.goal_coordinates[0]
        if 0 <= gy < self.grid_size[1] and 0 <= gx < self.grid_size[0]:
            cv2.circle(traversability_vis, (gx, gy), 5, (255, 0, 255), -1)
        
        # Draw path if available
        if self.path:
            for i in range(len(self.path)-1):
                pt1 = (self.path[i][0], self.path[i][1])
                pt2 = (self.path[i+1][0], self.path[i+1][1])
                cv2.line(traversability_vis, pt1, pt2, (255, 255, 255), 1)
        
        return traversability_vis
    
    def visualize_potential_field(self):
        """
        Visualize the potential field.
        
        Returns:
            Visualization image
        """
        if not self.use_potential_fields:
            return np.zeros((self.grid_size[1], self.grid_size[0], 3), dtype=np.uint8)
        
        # Create colormap for potential field (blue = low potential, red = high potential)
        cm = LinearSegmentedColormap.from_list('potential', ['blue', 'cyan', 'green', 'yellow', 'red'])
        
        # Apply colormap to potential field
        potential_colored = cm(self.potential_field)
        potential_vis = (potential_colored[:, :, :3] * 255).astype(np.uint8)
        
        # Draw goal position
        gy, gx = self.goal_coordinates[1], self.goal_coordinates[0]
        if 0 <= gy < self.grid_size[1] and 0 <= gx < self.grid_size[0]:
            cv2.circle(potential_vis, (gx, gy), 5, (255, 255, 255), -1)
        
        # Draw current position
        cy, cx = self.current_position[1], self.current_position[0]
        if 0 <= cy < self.grid_size[1] and 0 <= cx < self.grid_size[0]:
            cv2.circle(potential_vis, (cx, cy), 3, (255, 255, 255), -1)
        
        # Calculate gradient manually for visualization
        if hasattr(self, 'potential_field') and self.potential_field.size > 0:
            # Pad potential field for safe gradient calculation at edges
            potential_padded = np.pad(self.potential_field, 1, mode='edge')
            
            # Draw arrows for gradient direction (every 10 cells)
            arrow_spacing = 10
            for y in range(arrow_spacing, self.grid_size[1], arrow_spacing):
                for x in range(arrow_spacing, self.grid_size[0], arrow_spacing):
                    # Convert to padded coordinates
                    px, py = x + 1, y + 1
                    
                    # Calculate gradient using central differences
                    gx = -(potential_padded[py, px+1] - potential_padded[py, px-1]) / 2.0
                    gy = -(potential_padded[py+1, px] - potential_padded[py-1, px]) / 2.0
                    
                    # Calculate magnitude
                    magnitude = np.sqrt(gx*gx + gy*gy)
                    
                    # Skip if magnitude is too small
                    if magnitude < 0.01:
                        continue
                    
                    # Normalize and scale
                    gx = gx / magnitude * 5
                    gy = gy / magnitude * 5
                    
                    # Draw arrow
                    cv2.arrowedLine(potential_vis, (x, y), (int(x + gx), int(y + gy)), (255, 255, 255), 1)
        
        return potential_vis
    
    def visualize_velocity(self, left_vel, right_vel):
        """
        Visualize rover velocity as a speedometer.
        
        Args:
            left_vel: Left-side velocity
            right_vel: Right-side velocity
            
        Returns:
            Visualization image
        """
        # Create blank canvas
        width, height = 240, 240
        vis = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw rover outline
        rover_color = (100, 100, 100)
        rover_width = 60
        rover_length = 100
        rover_x = (width - rover_width) // 2
        rover_y = (height - rover_length) // 2
        
        # Draw rover body
        cv2.rectangle(vis, (rover_x, rover_y), (rover_x + rover_width, rover_y + rover_length), rover_color, 2)
        
        # Draw wheels
        wheel_width = 20
        wheel_length = 30
        wheel_color = (50, 50, 50)
        
        # Left front wheel
        lf_x = rover_x - wheel_width
        lf_y = rover_y + 10
        cv2.rectangle(vis, (lf_x, lf_y), (lf_x + wheel_width, lf_y + wheel_length), wheel_color, -1)
        
        # Right front wheel
        rf_x = rover_x + rover_width
        rf_y = rover_y + 10
        cv2.rectangle(vis, (rf_x, rf_y), (rf_x + wheel_width, rf_y + wheel_length), wheel_color, -1)
        
        # Left rear wheel
        lr_x = rover_x - wheel_width
        lr_y = rover_y + rover_length - wheel_length - 10
        cv2.rectangle(vis, (lr_x, lr_y), (lr_x + wheel_width, lr_y + wheel_length), wheel_color, -1)
        
        # Right rear wheel
        rr_x = rover_x + rover_width
        rr_y = rover_y + rover_length - wheel_length - 10
        cv2.rectangle(vis, (rr_x, rr_y), (rr_x + wheel_width, rr_y + wheel_length), wheel_color, -1)
        
        # Draw velocity indicators for each wheel
        def draw_velocity_indicator(x, y, velocity, max_velocity):
            # Normalize velocity
            norm_vel = velocity / max_velocity
            
            # Determine color based on velocity (green for forward, red for reverse)
            if norm_vel >= 0:
                color = (0, 255, 0)  # Green
                arrow_start = (x + wheel_width // 2, y + wheel_length // 2)
                arrow_end = (x + wheel_width // 2, y - int(15 * norm_vel))
            else:
                color = (0, 0, 255)  # Red
                arrow_start = (x + wheel_width // 2, y + wheel_length // 2)
                arrow_end = (x + wheel_width // 2, y + wheel_length + int(15 * abs(norm_vel)))
            
            # Draw arrow if velocity is not zero
            if abs(norm_vel) > 0.05:
                cv2.arrowedLine(vis, arrow_start, arrow_end, color, 2)
            
            # Display velocity value
            cv2.putText(vis, f"{velocity:.1f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Draw velocity indicators for left wheels
        draw_velocity_indicator(lf_x, lf_y, left_vel, self.max_velocity)
        draw_velocity_indicator(lr_x, lr_y, left_vel, self.max_velocity)
        
        # Draw velocity indicators for right wheels
        draw_velocity_indicator(rf_x, rf_y, right_vel, self.max_velocity)
        draw_velocity_indicator(rr_x, rr_y, right_vel, self.max_velocity)
        
        # Display command
        cv2.putText(vis, f"Command: {self.current_command}", (width//2 - 60, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return vis
    
    def visualize(self, frame, objects, left_vel, right_vel):
        """
        Create comprehensive visualization with multiple views.
        
        Args:
            frame: Current camera frame
            objects: Detected and classified objects
            left_vel: Left-side velocity
            right_vel: Right-side velocity
            
        Returns:
            Combined visualization image
        """
        # Generate individual visualizations
        self.viz_frame = self.visualize_camera_view(frame, objects)
        self.viz_occupancy = self.visualize_occupancy_grid()
        self.viz_traversability = self.visualize_traversability()
        self.viz_potential_field = self.visualize_potential_field()
        self.viz_velocity = self.visualize_velocity(left_vel, right_vel)
        
        # Resize for consistent display
        h_frame, w_frame = self.viz_frame.shape[:2]
        
        grid_size = 320
        viz_occupancy_resized = cv2.resize(self.viz_occupancy, (grid_size, grid_size), interpolation=cv2.INTER_NEAREST)
        viz_traversability_resized = cv2.resize(self.viz_traversability, (grid_size, grid_size), interpolation=cv2.INTER_NEAREST)
        viz_potential_resized = cv2.resize(self.viz_potential_field, (grid_size, grid_size), interpolation=cv2.INTER_NEAREST)
        
        # Create layout
        # Top row: Camera view | Occupancy grid
        # Bottom row: Traversability | Potential field | Velocity
        
        # Calculate dimensions
        top_row_height = max(h_frame, grid_size)
        bottom_row_height = grid_size
        
        # Create combined visualization
        width = max(w_frame + grid_size, grid_size + grid_size + self.viz_velocity.shape[1])
        height = top_row_height + bottom_row_height
        
        combined = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Place camera view
        combined[:h_frame, :w_frame] = self.viz_frame
        
        # Place occupancy grid
        combined[:grid_size, w_frame:w_frame+grid_size] = viz_occupancy_resized
        
        # Place traversability grid
        combined[top_row_height:top_row_height+grid_size, :grid_size] = viz_traversability_resized
        
        # Place potential field
        combined[top_row_height:top_row_height+grid_size, grid_size:grid_size*2] = viz_potential_resized
        
        # Place velocity visualization
        vel_viz = self.viz_velocity
        vh, vw = vel_viz.shape[:2]
        combined[top_row_height:top_row_height+vh, grid_size*2:grid_size*2+vw] = vel_viz
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Camera Feed & Object Detection", (10, 20), font, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, "Occupancy Grid", (w_frame + 10, 20), font, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, "Traversability Map", (10, top_row_height + 20), font, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, "Potential Field", (grid_size + 10, top_row_height + 20), font, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, "Wheel Velocities", (grid_size*2 + 10, top_row_height + 20), font, 0.6, (255, 255, 255), 2)
        
        return combined
    
    def run(self):
        """Main loop for autonomous navigation"""
        try:
            while True:
                # 1. Capture frame
                frame = self.capture_frame()
                if frame is None:
                    print("Failed to capture frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                # 2. Detect objects
                detected_objects = self.detect_objects(frame)
                
                # 3. Classify objects
                classified_objects = self.classify_objects(detected_objects)
                
                # 4. Update occupancy grid
                self.update_occupancy_grid(classified_objects, frame)
                
                # 5. Get movement command
                command = self.get_movement_command()
                
                # 6. Execute command
                left_vel, right_vel = self.execute_command(command)
                print(f"Command: {command}, Left: {left_vel:.2f} m/s, Right: {right_vel:.2f} m/s")
                
                # 7. Visualize
                visualization = self.visualize(frame, classified_objects, left_vel, right_vel)
                cv2.imshow("Autonomous Rover Navigation", visualization)
                
                # Save image periodically for debugging
                if self.frame_count % 30 == 0:  # Save every 30 frames
                    cv2.imwrite(f"rover_nav_{self.frame_count}.jpg", visualization)
                
                # Check if we reached the goal
                if command == self.cmd_stop and self.current_position == self.goal_coordinates:
                    print("Goal reached!")
                    # Display success message on visualization
                    cv2.putText(visualization, "GOAL REACHED!", (visualization.shape[1]//2 - 100, visualization.shape[0]//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    cv2.imshow("Autonomous Rover Navigation", visualization)
                    cv2.waitKey(3000)  # Wait 3 seconds to show success
                    break
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Maintain loop rate
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("Navigation stopped by user")
        except Exception as e:
            print(f"Error in navigation: {e}")
        finally:
            # Clean up
            self.camera.release()
            cv2.destroyAllWindows()


def simulate_camera(frame_size=(640, 480), object_configs=None):
    """
    Simulate a camera feed with objects for testing without a real camera.
    
    Args:
        frame_size: Tuple of (width, height) for the frame
        object_configs: List of object configurations to simulate
        
    Returns:
        Generated frame and detected objects
    """
    if object_configs is None:
        object_configs = [
            {"class": "wall", "position": (100, 200), "size": (200, 100), "confidence": 0.85},
            {"class": "rock", "position": (400, 300), "size": (50, 50), "confidence": 0.78},
            {"class": "person", "position": (300, 250), "size": (60, 120), "confidence": 0.92, 
             "velocity": (5, 0)}  # Moving right
        ]
    
    # Create blank frame
    frame = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 240  # Light gray background
    
    # Create a more realistic environment
    # Draw sky gradient
    for y in range(0, frame_size[1]//2):
        blue = 230 - int(y * 120 / (frame_size[1]//2))
        green = 220 - int(y * 80 / (frame_size[1]//2))
        red = 210 - int(y * 50 / (frame_size[1]//2))
        frame[y, :] = (blue, green, red)  # BGR format
    
    # Draw ground with texture
    ground_y = frame_size[1]//2
    ground_color = (90, 140, 90)  # Base ground color
    
    # Fill ground area
    frame[ground_y:, :] = ground_color
    
    # Add ground texture/pattern
    for y in range(ground_y, frame_size[1], 20):
        darkness = np.random.randint(0, 20)
        adjusted_color = tuple(max(0, c - darkness) for c in ground_color)
        cv2.line(frame, (0, y), (frame_size[0], y), adjusted_color, 1)
    
    for x in range(0, frame_size[0], 40):
        darkness = np.random.randint(0, 15)
        adjusted_color = tuple(max(0, c - darkness) for c in ground_color)
        cv2.line(frame, (x, ground_y), (x, frame_size[1]), adjusted_color, 1)
    
    # Draw distant mountains/horizon
    horizon_y = ground_y - 20
    
    # Draw several mountains with different heights
    for i in range(-1, 5):
        mountain_center = frame_size[0] * i // 4
        mountain_width = np.random.randint(frame_size[0]//2, frame_size[0])
        mountain_height = np.random.randint(50, 100)
        
        # Create mountain points
        points = np.array([
            [mountain_center - mountain_width//2, horizon_y],
            [mountain_center, horizon_y - mountain_height],
            [mountain_center + mountain_width//2, horizon_y]
        ])
        
        # Draw mountain
        mountain_color = (90, 90, 110)  # Gray-blue for distant mountains
        cv2.fillPoly(frame, [points], mountain_color)
    
    # Add sun or moon
    if np.random.random() > 0.5:  # 50% chance of sun
        sun_x = np.random.randint(frame_size[0]//4, 3*frame_size[0]//4)
        sun_y = np.random.randint(50, horizon_y - 50)
        sun_radius = np.random.randint(20, 40)
        
        # Sun with glow effect
        for r in range(sun_radius + 20, sun_radius, -4):
            alpha = (sun_radius + 20 - r) / 20  # Fade from 0 to 1
            color = tuple(int(230 * alpha + c * (1-alpha)) for c in (200, 220, 240))
            cv2.circle(frame, (sun_x, sun_y), r, color, -1)
        
        cv2.circle(frame, (sun_x, sun_y), sun_radius, (200, 240, 255), -1)
    
    detected_objects = []
    
    # Draw and track objects
    for config in object_configs:
        obj_class = config["class"]
        x, y = config["position"]
        width, height = config["size"]
        confidence = config["confidence"]
        
        # Update position if object has velocity
        if "velocity" in config:
            vx, vy = config["velocity"]
            config["position"] = (x + vx, y + vy)
            
            # Bounce off edges
            if x + vx < 0 or x + vx + width > frame_size[0]:
                config["velocity"] = (-vx, vy)
            if y + vy < ground_y or y + vy + height > frame_size[1]:
                config["velocity"] = (vx, -vy)
        
        # Draw object based on class
        if obj_class == "wall":
            color = (110, 110, 110)  # Gray
            # Draw brick pattern
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, -1)
            
            # Add brick texture
            brick_height = 15
            for by in range(y, y + height, brick_height):
                cv2.line(frame, (x, by), (x + width, by), (80, 80, 80), 1)
                
                # Stagger bricks in alternate rows
                offset = 0 if (by - y) % (brick_height * 2) == 0 else width // 4
                for bx in range(x + offset, x + width, width // 2):
                    cv2.line(frame, (bx, by), (bx, min(by + brick_height, y + height)), (80, 80, 80), 1)
                    
        elif obj_class == "rock":
            # Draw with irregular polygon instead of rectangle
            center_x = x + width // 2
            center_y = y + height // 2
            
            # Create irregular rock shape
            points = []
            for angle in range(0, 360, 45):
                r = min(width, height) // 2 * (0.8 + 0.4 * np.random.random())
                px = center_x + int(r * np.cos(np.radians(angle)))
                py = center_y + int(r * np.sin(np.radians(angle)))
                points.append([px, py])
            
            points = np.array(points, np.int32)
            points = points.reshape((-1, 1, 2))
            
            # Draw rock with gradient
            cv2.fillPoly(frame, [points], (100, 110, 130))  # Main color
            
            # Add texture with lines
            for i in range(3):
                p1_idx = np.random.randint(0, len(points))
                p2_idx = (p1_idx + len(points)//2) % len(points)
                p1 = (points[p1_idx][0][0], points[p1_idx][0][1])
                p2 = (points[p2_idx][0][0], points[p2_idx][0][1])
                cv2.line(frame, p1, p2, (90, 100, 120), 1)
            
        elif obj_class == "tree":
            # Draw trunk
            trunk_width = width // 3
            trunk_x = x + (width - trunk_width) // 2
            cv2.rectangle(frame, (trunk_x, y + height//3), (trunk_x + trunk_width, y + height), (60, 80, 120), -1)
            
            # Draw foliage as circles
            foliage_color = (40, 120, 80)  # Dark green
            cv2.circle(frame, (x + width//2, y + height//4), height//3, foliage_color, -1)
            cv2.circle(frame, (x + width//4, y + height//3), height//4, foliage_color, -1)
            cv2.circle(frame, (x + 3*width//4, y + height//3), height//4, foliage_color, -1)
            
        elif obj_class == "person":
            body_width = width // 2
            body_x = x + (width - body_width) // 2
            
            # Draw legs
            leg_width = body_width // 3
            cv2.rectangle(frame, (body_x, y + height//2), (body_x + leg_width, y + height), (0, 0, 120), -1)
            cv2.rectangle(frame, (body_x + body_width - leg_width, y + height//2), 
                           (body_x + body_width, y + height), (0, 0, 120), -1)
            
            # Draw body
            cv2.rectangle(frame, (body_x, y + height//5), (body_x + body_width, y + height//2), (0, 0, 200), -1)
            
            # Draw head
            head_size = height // 5
            cv2.circle(frame, (x + width//2, y + head_size), head_size, (200, 150, 150), -1)
            
            # Draw arms
            arm_height = height // 10
            cv2.rectangle(frame, (body_x - width//4, y + height//3), (body_x, y + height//3 + arm_height), (0, 0, 200), -1)
            cv2.rectangle(frame, (body_x + body_width, y + height//3), 
                           (body_x + body_width + width//4, y + height//3 + arm_height), (0, 0, 200), -1)
            
        elif obj_class == "car":
            # Draw car body
            cv2.rectangle(frame, (x, y + height//3), (x + width, y + height), (0, 0, 200), -1)
            
            # Draw car top
            top_points = np.array([
                [x + width//4, y + height//3],
                [x + width//4, y],
                [x + 3*width//4, y],
                [x + 3*width//4, y + height//3]
            ])
            cv2.fillPoly(frame, [top_points], (0, 0, 150))
            
            # Draw wheels
            wheel_size = height // 3
            cv2.circle(frame, (x + wheel_size, y + height - wheel_size//2), wheel_size//2, (50, 50, 50), -1)
            cv2.circle(frame, (x + width - wheel_size, y + height - wheel_size//2), wheel_size//2, (50, 50, 50), -1)
            
            # Windows
            window_height = height//6
            cv2.rectangle(frame, (x + width//4 + 5, y + 5), (x + 3*width//4 - 5, y + window_height - 5), (200, 200, 255), -1)
            
        else:
            # Default drawing for unknown objects
            color = (150, 150, 0)  # Cyan
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, -1)
        
        # Add border
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 0), 1)
        
        # Create detection entry
        detected_objects.append({
            'class': obj_class,
            'confidence': confidence,
            'bbox': (x, y, x + width, y + height)
        })
    
    # Add some random noise/grain to the image
    noise = np.zeros_like(frame, np.int16)
    cv2.randn(noise, 0, 5)
    frame = cv2.add(frame, noise, dtype=cv2.CV_8UC3)
    
    return frame, detected_objects


class SimulatedRoverNavigation(RoverNavigation):
    """Extension of RoverNavigation that uses simulated camera input."""
    
    def __init__(self, resolution=(640, 480), goal_coordinates=(100, 100)):
        super().__init__(camera_id=0, resolution=resolution, goal_coordinates=goal_coordinates)
        
        # Generate a more complex and varied environment based on goal location
        self.simulated_objects = self.generate_environment(goal_coordinates)
        
        # Override camera setup
        if hasattr(self, 'camera') and self.camera is not None:
            self.camera.release()
        self.camera = None
        
        # Set starting position at bottom center of grid
        self.current_position = (self.grid_size[0]//2, self.grid_size[1]-10)
    
    def generate_environment(self, goal):
        """
        Generate a varied environment with obstacles based on goal location.
        Creates a more interesting navigation challenge.
        
        Args:
            goal: Goal coordinates (x, y)
            
        Returns:
            List of simulated objects
        """
        objects = []
        width, height = self.resolution
        
        # Create a partial wall or barrier
        wall_x = width // 3
        wall_width = 30
        wall_y = height // 2
        wall_height = height // 3
        
        objects.append({
            "class": "wall", 
            "position": (wall_x, wall_y), 
            "size": (wall_width, wall_height), 
            "confidence": 0.92
        })
        
        # Add random rocks
        for _ in range(3):
            rock_x = np.random.randint(50, width - 100)
            rock_y = np.random.randint(height // 2, height - 80)
            rock_size = np.random.randint(40, 80)
            objects.append({
                "class": "rock", 
                "position": (rock_x, rock_y), 
                "size": (rock_size, rock_size // 2), 
                "confidence": 0.85
            })
        
        # Add trees
        for _ in range(2):
            tree_x = np.random.randint(width // 2, width - 80)
            tree_y = np.random.randint(height // 2, height - 100)
            objects.append({
                "class": "tree", 
                "position": (tree_x, tree_y), 
                "size": (60, 120), 
                "confidence": 0.88
            })
        
        # Create a second wall that partially blocks direct path to goal
        # Place it intelligently based on goal position
        gx, gy = goal
        screen_goal_x = (gx / self.grid_size[0]) * width
        screen_goal_y = ((self.grid_size[1] - gy) / self.grid_size[1]) * height
        
        # Place wall between start point and goal
        wall2_x = width // 2 + np.random.randint(-50, 50)
        wall2_y = height // 2 + 50
        wall2_width = 150
        wall2_height = 25
        
        objects.append({
            "class": "wall", 
            "position": (wall2_x, wall2_y), 
            "size": (wall2_width, wall2_height), 
            "confidence": 0.90
        })
        
        # Add moving people
        objects.append({
            "class": "person", 
            "position": (width // 4, height // 2 + 100), 
            "size": (50, 100), 
            "confidence": 0.92, 
            "velocity": (2, -1)  # Moving right and up
        })
        
        objects.append({
            "class": "person", 
            "position": (width // 2 + 100, height - 150), 
            "size": (60, 120), 
            "confidence": 0.88, 
            "velocity": (-3, 0)  # Moving left
        })
        
        # Add a moving car
        objects.append({
            "class": "car", 
            "position": (width - 200, height - 120), 
            "size": (120, 60), 
            "confidence": 0.95, 
            "velocity": (-4, 0)  # Moving left fast
        })
        
        return objects
    
    def capture_frame(self):
        """Override to use simulated camera feed"""
        frame, _ = simulate_camera(self.resolution, self.simulated_objects)
        return frame
    
    def detect_objects(self, frame):
        """Override to return simulated detections"""
        _, detections = simulate_camera(self.resolution, self.simulated_objects)
        return detections


def test_with_simulation():
    """Test the rover navigation algorithm with simulated input."""
    # Create simulated rover with goal at (80, 30)
    rover = SimulatedRoverNavigation(goal_coordinates=(80, 30))
    
    # Enable potential fields
    rover.use_potential_fields = True
    
    # Run navigation
    rover.run()


# Visualization-only mode to render pre-recorded data
def visualize_recorded_data(data_file):
    """
    Visualize pre-recorded navigation data.
    
    Args:
        data_file: Path to data file with recorded frames and detections
    """
    import pickle
    
    # Load data
    with open(data_file, 'rb') as f:
        recorded_data = pickle.load(f)
    
    # Create rover
    rover = RoverNavigation()
    
    # Replay data
    for frame_data in recorded_data:
        frame = frame_data['frame']
        detections = frame_data['detections']
        position = frame_data['position']
        rover.current_position = position
        
        # Process detections
        classified = rover.classify_objects(detections)
        rover.update_occupancy_grid(classified, frame)
        
        # Get command
        command = rover.get_movement_command()
        left_vel, right_vel = rover.execute_command(command)
        
        # Visualize
        vis = rover.visualize(frame, classified, left_vel, right_vel)
        cv2.imshow("Recorded Navigation", vis)
        
        # Wait for key or delay
        key = cv2.waitKey(100)
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()


# Usage example
if __name__ == "__main__":
    print("Autonomous Rover Navigation with Camera-based Perception")
    print("--------------------------------------------------------")
    
    # Get user input for goal coordinates
    try:
        x_goal = int(input("Enter goal X coordinate (0-100, default is 80): ") or "80")
        y_goal = int(input("Enter goal Y coordinate (0-100, default is 30): ") or "30")
        
        # Validate input
        x_goal = max(0, min(100, x_goal))
        y_goal = max(0, min(100, y_goal))
        
        print(f"\nSetting destination to ({x_goal}, {y_goal})")
        
        # Ask user about pathfinding method
        method_choice = input("\nSelect pathfinding method (a=A*, p=Potential Fields, default is A*): ").lower() or "a"
        use_potential = method_choice.startswith('p')
        
        if use_potential:
            print("Using Potential Fields method for pathfinding")
        else:
            print("Using A* method for pathfinding")
            
        print("\nStarting simulation...\n")
        
        # Initialize simulated rover with the goal coordinates
        rover = SimulatedRoverNavigation(goal_coordinates=(x_goal, y_goal))
        
        # Set pathfinding method
        rover.use_potential_fields = use_potential
        
        # Run autonomous navigation
        rover.run()
        
    except ValueError:
        print("Invalid input. Using default values.")
        # Run with default values
        rover = SimulatedRoverNavigation(goal_coordinates=(80, 30))
        rover.run()
    except KeyboardInterrupt:
        print("\nNavigation stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

# Main function
if __name__ == "__main__":
    print("Autonomous Rover Navigation with Camera-based Perception")
    print("--------------------------------------------------------")
    
    # Get user input for goal coordinates
    try:
        x_goal = int(input("Enter goal X coordinate (0-100, default is 80): ") or "80")
        y_goal = int(input("Enter goal Y coordinate (0-100, default is 30): ") or "30")
        
        # Validate input
        x_goal = max(0, min(100, x_goal))
        y_goal = max(0, min(100, y_goal))
        
        print(f"\nSetting destination to ({x_goal}, {y_goal})")
        
        # Ask user about pathfinding method
        method_choice = input("\nSelect pathfinding method (a=A*, p=Potential Fields, default is A*): ").lower() or "a"
        use_potential = method_choice.startswith('p')
        
        if use_potential:
            print("Using Potential Fields method for pathfinding")
        else:
            print("Using A* method for pathfinding")
            
        print("\nStarting navigation...\n")
        
        # Initialize rover navigation with specified camera and goal
        camera_id = 0  # Use default camera (usually webcam)
        rover = RoverNavigation(camera_id=camera_id, goal_coordinates=(x_goal, y_goal))
        
        # Set pathfinding method
        rover.use_potential_fields = use_potential
        
        # Run autonomous navigation
        rover.run()
        
    except ValueError:
        print("Invalid input. Using default values.")
        # Run with default values
        rover = RoverNavigation(goal_coordinates=(80, 30))
        rover.run()
    except KeyboardInterrupt:
        print("\nNavigation stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()