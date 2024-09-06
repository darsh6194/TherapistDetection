import cv2
from ultralytics import YOLO
import numpy as np

# Class IDs for therapist and child (adjust these as per your model)
THERAPIST_CLASS_ID = 1
CHILD_CLASS_ID = 0

class TherapistTracker:
    def __init__(self):
        self.next_id = 0
        self.therapists = {}  # Store therapist centroids and bounding boxes with IDs

    def update(self, detections):
        new_therapists = []

        # Loop over all detections and filter by therapist class
        for det in detections.boxes:
            class_id = int(det.cls.cpu().numpy())
            if class_id == THERAPIST_CLASS_ID:  # Only consider therapists
                # Get bounding box coordinates
                x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
                # Compute centroid of the bounding box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                new_therapists.append((cx, cy, det))

        updated_therapists = {}

        # Match new therapist detections to existing ones
        for cx, cy, box in new_therapists:
            matched_id = None
            for therapist_id, (ox, oy, _) in self.therapists.items():
                # If the new detection is close to an existing one, consider it the same therapist
                if np.linalg.norm([ox - cx, oy - cy]) < 50:  # 50-pixel distance threshold
                    matched_id = therapist_id
                    break

            if matched_id is None:
                # Assign a new ID to the new therapist
                matched_id = self.next_id
                self.next_id += 1

            # Update the therapist with the new position
            updated_therapists[matched_id] = (cx, cy, box)

        # Replace old therapist dictionary with updated one
        self.therapists = updated_therapists
        return self.therapists


# Initialize the YOLO model and video capture
model = YOLO(r'models\best.pt') # your path
cap = cv2.VideoCapture(r'input\ABA Therapy - Social Engagement.mp4') # your path

# Initialize the therapist tracker
tracker = TherapistTracker()

# Open video output file to save the result
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(r'output\annotated_video.mp4', fourcc, 20.0, (640, 480))

# Frame processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    resized_frame = cv2.resize(frame, (640, 480))

    # Perform YOLO detection on the frame
    results = model(resized_frame)

    # Track only therapists
    tracked_therapists = tracker.update(results[0])

    # Annotate frame with therapist bounding boxes and IDs
    for therapist_id, (_, _, box) in tracked_therapists.items():
        # Draw bounding box for therapists only
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cv2.rectangle(resized_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # Display therapist ID
        cv2.putText(resized_frame, f'Therapist #{therapist_id}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Additionally, draw bounding boxes for children (if detected, without assigning IDs)
    for det in results[0].boxes:
        class_id = int(det.cls.cpu().numpy())
        if class_id == CHILD_CLASS_ID:  # Only consider children
            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
            cv2.rectangle(resized_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(resized_frame, 'Child', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Therapist Detection', resized_frame)
    out.write(resized_frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
