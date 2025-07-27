import cv2

# Use your external camera (index 2)
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Failed to open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Get dimensions of the frame
    height, width, _ = frame.shape

    # Define the region to crop - top-right corner
    # You can fine-tune these values later
    crop_width = 300
    crop_height = 100
    x_start = width - crop_width
    y_start = 0
    cropped = frame[y_start:y_start + crop_height, x_start:x_start + crop_width]

    # Show both the full frame and cropped timer area
    cv2.imshow("Full Camera Feed", frame)
    cv2.imshow("Timer Crop", cropped)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()