import imutils
import cv2
import time

# Initialize the video capture with the default camera
cam = cv2.VideoCapture(0)
time.sleep(1)

# Initialize the first frame to None
firstScreen = None

# Set the minimum area for a detected object
area = 500

while True:
    # Capture the current frame
    flag, image = cam.read()
    
    # Check if the frame was successfully captured
    if not flag:
        print("Failed to capture image")
        break
    
    # Resize the frame for consistency
    image = imutils.resize(image, width=500)
    
    # Convert the frame to grayscale
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to the grayscale frame
    gaussianImage = cv2.GaussianBlur(grayImage, (21, 21), 0)

    # Initialize the first frame if it is None
    if firstScreen is None:
        firstScreen = gaussianImage
        continue
    
    # Compute the absolute difference between the first frame and the current frame
    imageDiff = cv2.absdiff(firstScreen, gaussianImage)
    
    # Apply thresholding to get a binary image
    threshImage = cv2.threshold(imageDiff, 25, 255, cv2.THRESH_BINARY)[1]
    
    # Dilate the thresholded image to fill in holes
    threshImage = cv2.dilate(threshImage, None, iterations=2)
    
    # Find contours on the thresholded image
    cnts = cv2.findContours(threshImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Initialize text to "Normal"
    text = "Normal"

    # Loop over the contours
    for c in cnts:
        # If the contour is too small, ignore it
        if cv2.contourArea(c) < area:
            continue
        
        # Compute the bounding box for the contour and draw it on the frame
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Moving Object detected"
        
        # Draw the contour on the image
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    
    # Print the status to the console
    print(text)
    
    # Draw the text and timestamp on the frame
    cv2.putText(image, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(image, time.strftime("%A %d %B %Y %I:%M:%S%p"), (10, image.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # Display the frame
    cv2.imshow("cameraFeed", image)
    
    # If the 'q' key is pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
