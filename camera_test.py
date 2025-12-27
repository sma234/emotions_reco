import cv2

print("Starting camera test")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR:Could not open camera.Check permissions.")
    exit()

while True:
    ret,frame=cap.read()
    if not ret:
        print("ERROR:Could not read frame from camera.")
        break

    cv2.imshow("Camera Test - press q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("INFO:'q' pressed, exiting.")
        break

cap.release()
cv2.destroyAllWindows()
print("INFO:camera test finished")
