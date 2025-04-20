import serial
import time
import matplotlib.pyplot as plt
import numpy as np

# Serial Config
PORT = "/dev/cu.usbmodem1201"  # Change this to your port (e.g., "COM3" for Windows)
BAUD = 9600
MAX_DISTANCE = 100  # Max range in cm

# Setup Serial
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)  # Allow Arduino to initialize

# Setup Plot
plt.ion()
fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'projection': 'polar'})
ax.set_ylim(0, MAX_DISTANCE)
ax.set_theta_zero_location("N")  # 0° is straight up
ax.set_theta_direction(-1)       # Clockwise
ax.set_thetamin(-60)             # Show left to right ±60°
ax.set_thetamax(60)

# Initial Plot Dot
angle_deg = 0  # Centered forward
angle_rad = np.radians(angle_deg)
dot, = ax.plot([], [], 'ro', markersize=10)

def update_plot(distance):
    distance = min(distance, MAX_DISTANCE)
    dot.set_data([angle_rad], [distance])
    plt.draw()
    plt.pause(0.01)

try:
    print("Starting front-facing arc sonar visualization...")
    while True:
        line = ser.readline().decode('utf-8').strip()
        if line:
            try:
                distance = float(line)
                update_plot(distance)
            except ValueError:
                pass

except KeyboardInterrupt:
    print("Stopped.")

finally:
    ser.close()
    plt.ioff()
    plt.show()
