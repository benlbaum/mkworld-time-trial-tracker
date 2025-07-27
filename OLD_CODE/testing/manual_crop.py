import cv2

# Camera index (2 = your external cam)
cap = cv2.VideoCapture(2)

# Globals to store cropping coordinates
crop_start = None
crop_end = None
cropping = False

def mouse_crop(event, x, y, flags, param):
    global crop_start, crop_end, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        crop_start = (x, y)
        crop_end = None
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        crop_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        crop_end = (x, y)
        cropping = False
        print(f"\nSelected crop box: x={crop_start[0]}:{crop_end[0]}, y={crop_start[1]}:{crop_end[1]}")

cv2.namedWindow("Calibration")
cv2.setMouseCallback("Calibration", mouse_crop)

print("🟡 Draw a box around the timer by clicking and dragging with your mouse.")
print("🔴 Press 'q' to quit after cropping.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()

    if crop_start and crop_end:
        x1, y1 = crop_start
        x2, y2 = crop_end
        cv2.rectangle(display_frame, crop_start, crop_end, (0, 255, 0), 2)
        cropped = frame[y1:y2, x1:x2]
        cv2.imshow("Live Crop Preview", cropped)

    cv2.imshow("Calibration", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()