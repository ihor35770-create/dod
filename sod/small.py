# https://github.com/pysource-com/Free-Code-Samples

import cv2
from engine.object_detection import ObjectDetection


od = ObjectDetection("models/yolo11m.pt")

videofn = "14691540-uhd_3840_2160_24fps.mp4"
# videofn = "demo/crowd_FHD.mp4"

cap = cv2.VideoCapture(videofn)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    bboxes, labels, confs = od.detect(frame)
    for bbox, label, conf in zip(bboxes, labels, confs):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label}: {conf:.2f}"
        #cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
