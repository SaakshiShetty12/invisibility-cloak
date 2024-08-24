import cv2
import numpy as np

# Initial function for the trackbar callback
def hello(x):
    pass  # Placeholder function

# Initialize the camera
cap = cv2.VideoCapture(0)
bars = cv2.namedWindow("bars")

# Create trackbars for upper and lower HSV values
cv2.createTrackbar("upper_hue", "bars", 80, 180, hello)
cv2.createTrackbar("upper_saturation", "bars", 255, 255, hello)
cv2.createTrackbar("upper_value", "bars", 150, 255, hello)
cv2.createTrackbar("lower_hue", "bars", 40, 180, hello)
cv2.createTrackbar("lower_saturation", "bars", 40, 255, hello)
cv2.createTrackbar("lower_value", "bars", 40, 255, hello)

# Capturing the initial frame for background creation
while True:
    cv2.waitKey(1000)
    ret, init_frame = cap.read()
    if ret:
        break

# Start capturing frames for the magic!
while True:
    ret, frame = cap.read()
    inspect = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Getting the HSV values from trackbars
    upper_hue = cv2.getTrackbarPos("upper_hue", "bars")
    upper_saturation = cv2.getTrackbarPos("upper_saturation", "bars")
    upper_value = cv2.getTrackbarPos("upper_value", "bars")
    lower_value = cv2.getTrackbarPos("lower_value", "bars")
    lower_hue = cv2.getTrackbarPos("lower_hue", "bars")
    lower_saturation = cv2.getTrackbarPos("lower_saturation", "bars")

    # Define the range of dark green color in HSV
    upper_green = np.array([upper_hue, upper_saturation, upper_value])
    lower_green = np.array([lower_hue, lower_saturation, lower_value])

    # Create a mask for the green color and perform morphological operations
    mask = cv2.inRange(inspect, lower_green, upper_green)
    mask = cv2.medianBlur(mask, 3)
    mask_inv = 255 - mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=5)

    # Create the invisible effect by blending frames
    b = frame[:, :, 0]
    g = frame[:, :, 1]
    r = frame[:, :, 2]
    b = cv2.bitwise_and(mask_inv, b)
    g = cv2.bitwise_and(mask_inv, g)
    r = cv2.bitwise_and(mask_inv, r)
    frame_inv = cv2.merge((b, g, r))

    b = init_frame[:, :, 0]
    g = init_frame[:, :, 1]
    r = init_frame[:, :, 2]
    b = cv2.bitwise_and(b, mask)
    g = cv2.bitwise_and(g, mask)
    r = cv2.bitwise_and(r, mask)
    blanket_area = cv2.merge((b, g, r))

    final = cv2.bitwise_or(frame_inv, blanket_area)

    # Display the final result
    cv2.imshow("Invisibility Cloak", final)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
