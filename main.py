import cv2

# Load the video
cap = cv2.VideoCapture("robot_video.mp4")

# Set up background subtraction algorithm
fgbg = cv2.createBackgroundSubtractorMOG2()

# Threshold for maximum allowed movement
movement_threshold = 50

# Initialize variables for previous frame and robot position
prev_frame = None
prev_robot_pos = None

while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction to detect the robot
    fgmask = fgbg.apply(frame)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour (assume it's the robot)
        robot_contour = max(contours, key=cv2.contourArea)
        # Get the center of the robot
        robot_pos = cv2.moments(robot_contour)["m10"] / cv2.moments(robot_contour)["m00"]

        if prev_frame is not None:
            # Calculate the robot's movement
            robot_movement = abs(robot_pos - prev_robot_pos)

            if robot_movement > movement_threshold:
                # Draw a red rectangle around the wrong movement
                cv2.rectangle(frame, (int(robot_pos - movement_threshold), 0), (int(robot_pos + movement_threshold), frame.shape[0]), (0, 0, 255), 2)

        prev_frame = frame
        prev_robot_pos = robot_pos

    # Show the video with wrong movements highlighted
    cv2.imshow("Robot video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
