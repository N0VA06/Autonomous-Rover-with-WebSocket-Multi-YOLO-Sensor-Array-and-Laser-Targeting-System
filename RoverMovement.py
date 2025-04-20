import cv2
import numpy as np
import time
from collections import deque
import math
from gpiozero import PWMOutputDevice, DigitalOutputDevice

class RoverNavigation:
    def __init__(self, camera_id=0, resolution=(640, 480), goal_coordinates=(100, 100)):
        """
        Initialize the autonomous rover navigation system.
        
        Args:
            camera_id: Camera device ID
            resolution: Camera resolution
            goal_coordinates: Target destination coordinates in 2D grid
        """
        # GPIO setup using gpiozero - using BCM numbering
        self.PWM1_PIN = 17  # GPIO 17 (Pin 11)
        self.DIR1_PIN = 27  # GPIO 27 (Pin 13)
        self.PWM2_PIN = 22  # GPIO 22 (Pin 15)
        self.DIR2_PIN = 23  # GPIO 23 (Pin 16)
        
        print(f"Setting up GPIO pins:")
        print(f"Left Motor: PWM={self.PWM1_PIN}, DIR={self.DIR1_PIN}")
        print(f"Right Motor: PWM={self.PWM2_PIN}, DIR={self.DIR2_PIN}")
        
        # Setup PWM and direction pins
        try:
            self.pwm1 = PWMOutputDevice(self.PWM1_PIN, frequency=500)
            self.dir1 = DigitalOutputDevice(self.DIR1_PIN)
            self.pwm2 = PWMOutputDevice(self.PWM2_PIN, frequency=500)
            self.dir2 = DigitalOutputDevice(self.DIR2_PIN)
            print("GPIO devices initialized successfully")
        except Exception as e:
            print(f"Error initializing GPIO devices: {e}")
            raise
        
        # Initialize motors to stop
        self.pwm1.value = 0
        self.pwm2.value = 0
        
        # Fixed PWM value (50/255 ≈ 0.2)
        self.fixed_speed = 0.2
        print(f"Using fixed speed: {self.fixed_speed}")
        
        # Camera setup
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        # Grid system setup
        self.global_grid_size = (100, 100)  # Global grid size in meters
        self.local_grid_size = (20, 20)     # Local grid size in meters
        self.grid_resolution = 0.1          # Grid cell size in meters
        
        # Initialize grids
        self.global_grid = np.zeros(self.global_grid_size)
        self.local_grid = np.zeros(self.local_grid_size)
        
        # Position tracking
        self.global_position = (self.global_grid_size[0]//2, 0)  # Start at center bottom
        self.local_position = (self.local_grid_size[0]//2, self.local_grid_size[1]//2)
        self.current_orientation = 0  # Degrees, 0 = forward
        
        # Goal tracking
        self.global_goal = goal_coordinates
        self.local_goal = None
        self.goal_reached = False
        
        # Movement parameters
        self.movement_step = 0.1  # Meters per step
        self.rotation_step = 15   # Degrees per rotation
        
        # Navigation parameters
        self.resolution = resolution
        self.grid_size = (120, 120)  # Increased grid resolution
        self.occupancy_grid = np.zeros(self.grid_size)
        self.confidence_grid = np.zeros(self.grid_size)
        self.traversability_grid = np.ones(self.grid_size)
        
        # Movement states
        self.current_position = (self.grid_size[0]//2, self.grid_size[1]-10)
        self.previous_positions = deque(maxlen=20)
        
        # Object detection model
        self.detection_model = None
        try:
            import torch
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
        
        # Movement commands
        self.cmd_forward = "FORWARD"
        self.cmd_reverse = "REVERSE"
        self.cmd_left = "LEFT"
        self.cmd_right = "RIGHT"
        self.cmd_stop = "STOP"
        self.current_command = self.cmd_stop
        
        # Test motors on startup
        self.test_motors()

    def test_motors(self):
        """Test each motor individually to ensure proper operation"""
        print("\nTesting motors...")
        
        # Test left motor forward
        print("Testing left motor forward...")
        self.dir1.on()  # HIGH
        self.pwm1.value = self.fixed_speed
        time.sleep(2)  # Increased test duration
        self.pwm1.value = 0
        time.sleep(1)
        
        # Test right motor forward
        print("Testing right motor forward...")
        self.dir2.on()  # HIGH
        self.pwm2.value = self.fixed_speed
        time.sleep(2)  # Increased test duration
        self.pwm2.value = 0
        time.sleep(1)
        
        # Test left motor backward
        print("Testing left motor backward...")
        self.dir1.off()  # LOW
        self.pwm1.value = self.fixed_speed
        time.sleep(2)  # Increased test duration
        self.pwm1.value = 0
        time.sleep(1)
        
        # Test right motor backward
        print("Testing right motor backward...")
        self.dir2.off()  # LOW
        self.pwm2.value = self.fixed_speed
        time.sleep(2)  # Increased test duration
        self.pwm2.value = 0
        time.sleep(1)
        
        print("Motor testing complete")

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
        Improved detection with better obstacle handling.
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
            # Use YOLO for detection with improved confidence threshold
            results = self.detection_model(frame, conf=0.4)  # Lower confidence threshold for better detection
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
        Determine the next movement command based on the current path and goal.
        Improved obstacle avoidance and goal tracking.
        """
        # If goal is reached, stop
        if self.check_goal_reached():
            return self.cmd_stop
        
        # Check for obstacles in the immediate path
        if self.check_immediate_obstacles():
            return self.cmd_stop
        
        # Update local goal
        if not self.update_local_goal():
            # If local goal is outside local grid, turn towards global goal
            dx = self.global_goal[0] - self.global_position[0]
            dy = self.global_goal[1] - self.global_position[1]
            target_angle = math.degrees(math.atan2(dx, dy))
            
            # Calculate angle difference
            angle_diff = (target_angle - self.current_orientation) % 360
            if angle_diff > 180:
                angle_diff -= 360
            
            # Turn towards goal
            if abs(angle_diff) < 15:
                return self.cmd_forward
            elif angle_diff > 0:
                return self.cmd_left
            else:
                return self.cmd_right
        
        # If we have a valid local goal, use A* pathfinding
        if self.local_goal:
            if not self.path or self.current_path_index >= len(self.path):
                self.path = self.a_star_pathfinding()
                self.current_path_index = 0
            
            if self.path and self.current_path_index < len(self.path):
                next_waypoint = self.path[self.current_path_index]
                dx = next_waypoint[0] - self.local_position[0]
                dy = next_waypoint[1] - self.local_position[1]
                
                # If close to waypoint, move to next
                if math.sqrt(dx*dx + dy*dy) < 0.2:
                    self.current_path_index += 1
                    return self.get_movement_command()
                
                # Calculate angle to waypoint
                target_angle = math.degrees(math.atan2(dx, dy))
                angle_diff = (target_angle - self.current_orientation) % 360
                if angle_diff > 180:
                    angle_diff -= 360
                
                # Determine movement
                if abs(angle_diff) < 15:
                    return self.cmd_forward
                elif angle_diff > 0:
                    return self.cmd_left
                else:
                    return self.cmd_right
        
        # Default to forward if no clear path
        return self.cmd_forward

    def update_position(self, command):
        """
        Update the rover's position based on the executed command.
        """
        if command == self.cmd_forward:
            # Convert orientation to radians
            angle_rad = math.radians(self.current_orientation)
            # Update global position
            self.global_position = (
                self.global_position[0] + self.movement_step * math.sin(angle_rad),
                self.global_position[1] + self.movement_step * math.cos(angle_rad)
            )
            # Update local position
            self.local_position = (
                self.local_position[0] + self.movement_step * math.sin(angle_rad),
                self.local_position[1] + self.movement_step * math.cos(angle_rad)
            )
        elif command == self.cmd_left:
            self.current_orientation = (self.current_orientation + self.rotation_step) % 360
        elif command == self.cmd_right:
            self.current_orientation = (self.current_orientation - self.rotation_step) % 360
        
        # Check if local position is out of bounds
        if (self.local_position[0] < 0 or self.local_position[0] >= self.local_grid_size[0] or
            self.local_position[1] < 0 or self.local_position[1] >= self.local_grid_size[1]):
            # Reset local grid and position
            self.local_grid = np.zeros(self.local_grid_size)
            self.local_position = (self.local_grid_size[0]//2, self.local_grid_size[1]//2)
            # Update local goal relative to new local grid
            self.update_local_goal()

    def update_local_goal(self):
        """
        Update the local goal based on the global goal and current position.
        """
        # Calculate relative position of global goal in local grid
        dx = self.global_goal[0] - self.global_position[0]
        dy = self.global_goal[1] - self.global_position[1]
        
        # Convert to local grid coordinates
        local_x = self.local_position[0] + dx
        local_y = self.local_position[1] + dy
        
        # Set local goal
        self.local_goal = (local_x, local_y)
        
        # Check if goal is within local grid
        if (0 <= local_x < self.local_grid_size[0] and 
            0 <= local_y < self.local_grid_size[1]):
            return True
        return False

    def check_goal_reached(self):
        """
        Check if the rover has reached its goal.
        """
        # Calculate distance to global goal
        dx = self.global_goal[0] - self.global_position[0]
        dy = self.global_goal[1] - self.global_position[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        # If within 0.5 meters of goal, consider it reached
        if distance < 0.5:
            self.goal_reached = True
            return True
        return False

    def execute_command(self, command):
        """
        Execute movement command by controlling MDD10A motor driver via GPIO pins.
        Matches the Arduino motor control logic.
        """
        print(f"\nExecuting command: {command}")
        
        if command == self.cmd_forward:
            print("Moving FORWARD")
            # Forward at fixed speed
            self.dir1.on()  # HIGH
            self.pwm1.value = self.fixed_speed
            self.dir2.on()  # HIGH
            self.pwm2.value = self.fixed_speed
            print(f"Left Motor: DIR={self.dir1.value}, PWM={self.pwm1.value}")
            print(f"Right Motor: DIR={self.dir2.value}, PWM={self.pwm2.value}")
            
        elif command == self.cmd_reverse:
            print("Moving BACKWARD")
            # Backward at fixed speed
            self.dir1.off()  # LOW
            self.pwm1.value = self.fixed_speed
            self.dir2.off()  # LOW
            self.pwm2.value = self.fixed_speed
            print(f"Left Motor: DIR={self.dir1.value}, PWM={self.pwm1.value}")
            print(f"Right Motor: DIR={self.dir2.value}, PWM={self.pwm2.value}")
            
        elif command == self.cmd_left:
            print("Turning LEFT")
            # Turn Left
            self.dir1.off()  # LOW
            self.pwm1.value = self.fixed_speed
            self.dir2.on()  # HIGH
            self.pwm2.value = self.fixed_speed
            print(f"Left Motor: DIR={self.dir1.value}, PWM={self.pwm1.value}")
            print(f"Right Motor: DIR={self.dir2.value}, PWM={self.pwm2.value}")
            
        elif command == self.cmd_right:
            print("Turning RIGHT")
            # Turn Right
            self.dir1.on()  # HIGH
            self.pwm1.value = self.fixed_speed
            self.dir2.off()  # LOW
            self.pwm2.value = self.fixed_speed
            print(f"Left Motor: DIR={self.dir1.value}, PWM={self.pwm1.value}")
            print(f"Right Motor: DIR={self.dir2.value}, PWM={self.pwm2.value}")
            
        elif command == self.cmd_stop:
            print("STOPPING")
            # Stop
            self.pwm1.value = 0
            self.pwm2.value = 0
            print(f"Left Motor: DIR={self.dir1.value}, PWM={self.pwm1.value}")
            print(f"Right Motor: DIR={self.dir2.value}, PWM={self.pwm2.value}")

        return self.fixed_speed, self.fixed_speed, command

    def cleanup(self):
        """Clean up GPIO resources"""
        print("\nCleaning up GPIO resources")
        # Stop motors
        self.pwm1.value = 0
        self.pwm2.value = 0
        
        # Close GPIO devices
        self.pwm1.close()
        self.pwm2.close()
        self.dir1.close()
        self.dir2.close()
        print("GPIO cleanup complete")

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
        
        # Draw path
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
        left_vel, right_vel, _ = self.execute_command(self.current_command)
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
    
    def detect_people(self, frame):
        """
        Detect people in the frame using YOLOv8 and return their positions.
        Improved detection with better zone handling and distance estimation.
        """
        # Run YOLOv8 inference
        results = self.detection_model(frame, conf=0.4)  # Lower confidence threshold for better detection
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Define detection zones (wider zones for better coverage)
        left_zone = width // 4
        right_zone = 3 * width // 4
        
        # Initialize lists to store detected people
        left_people = []
        right_people = []
        center_people = []
        
        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Calculate person's center point
                person_center_x = (x1 + x2) // 2
                person_center_y = (y1 + y2) // 2
                
                # Calculate person's height and width
                person_height = y2 - y1
                person_width = x2 - x1
                
                # Estimate distance based on person's height in frame
                # Assuming average person height is 1.7m
                distance = (1.7 * self.focal_length) / person_height
                
                # Determine which zone the person is in
                if person_center_x < left_zone:
                    left_people.append((person_center_x, person_center_y, distance, person_width))
                elif person_center_x > right_zone:
                    right_people.append((person_center_x, person_center_y, distance, person_width))
                else:
                    center_people.append((person_center_x, person_center_y, distance, person_width))
                
                # Draw detection box and distance
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Dist: {distance:.2f}m", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw zone boundaries
        cv2.line(frame, (left_zone, 0), (left_zone, height), (255, 0, 0), 2)
        cv2.line(frame, (right_zone, 0), (right_zone, height), (255, 0, 0), 2)
        
        return left_people, right_people, center_people, frame

    def avoid_people(self, frame):
        """
        Avoid people by analyzing their positions and determining the best path.
        Improved avoidance logic with better decision making.
        """
        # Detect people in the frame
        left_people, right_people, center_people, frame = self.detect_people(frame)
        
        # Calculate average distances and positions
        left_avg_dist = np.mean([p[2] for p in left_people]) if left_people else float('inf')
        right_avg_dist = np.mean([p[2] for p in right_people]) if right_people else float('inf')
        center_avg_dist = np.mean([p[2] for p in center_people]) if center_people else float('inf')
        
        # Check for immediate obstacles (people too close)
        min_safe_distance = 2.0  # Minimum safe distance in meters
        
        # Check center zone first
        if center_people and any(p[2] < min_safe_distance for p in center_people):
            # If people are too close in center, decide based on side distances
            if left_avg_dist > right_avg_dist:
                return self.cmd_left, frame  # Turn left if left side is clearer
            else:
                return self.cmd_right, frame  # Turn right if right side is clearer
        
        # Check side zones
        if left_people and any(p[2] < min_safe_distance for p in left_people):
            return self.cmd_right, frame  # Turn right if left side is blocked
        
        if right_people and any(p[2] < min_safe_distance for p in right_people):
            return self.cmd_left, frame  # Turn left if right side is blocked
        
        # If no immediate obstacles, continue forward
        return self.cmd_forward, frame
    
    def check_immediate_obstacles(self):
        """
        Check for obstacles in the immediate path of the rover.
        Returns True if obstacles are detected within safe distance.
        """
        # Get current frame
        ret, frame = self.camera.read()
        if not ret:
            return False
        
        # Detect objects in the frame
        objects = self.detect_objects(frame)
        
        # Check for obstacles in the immediate path
        for obj in objects:
            x1, y1, x2, y2 = obj['bbox']
            obj_center_x = (x1 + x2) // 2
            obj_center_y = (y1 + y2) // 2
            
            # Calculate distance to object
            obj_height = y2 - y1
            distance = (1.7 * self.focal_length) / obj_height  # Assuming average person height of 1.7m
            
            # If object is too close (less than 2 meters) and in the center of the frame
            if distance < 2.0 and abs(obj_center_x - frame.shape[1]//2) < frame.shape[1]//4:
                return True
        
        return False

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
                
                # 3. Check for immediate obstacles
                if self.check_immediate_obstacles():
                    print("Obstacle detected! Stopping...")
                    self.execute_command(self.cmd_stop)
                    time.sleep(0.5)  # Wait before checking again
                    continue
                
                # 4. Get movement command
                command = self.get_movement_command()
                
                # 5. Execute command
                left_vel, right_vel, _ = self.execute_command(command)
                print(f"Command: {command}, Left: {left_vel:.2f} m/s, Right: {right_vel:.2f} m/s")
                
                # 6. Update position
                self.update_position(command)
                
                # 7. Visualize
                visualization = self.visualize(frame, detected_objects)
                cv2.imshow("Autonomous Rover Navigation", visualization)
                
                # Check if we reached the goal
                if self.check_goal_reached():
                    print("Goal reached!")
                    cv2.putText(visualization, "GOAL REACHED!", (visualization.shape[1]//2 - 100, visualization.shape[0]//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    cv2.imshow("Autonomous Rover Navigation", visualization)
                    cv2.waitKey(3000)
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
            self.cleanup()


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
        rover = RoverNavigation(camera_id=camera_id, 
                              goal_coordinates=(x_goal, y_goal))
        
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
    finally:
        rover.cleanup()  # Clean up GPIO resources