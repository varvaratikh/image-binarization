import cv2
import numpy as np

image_path = 'image.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((3, 3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_area = 100
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

object_data = []

for contour in filtered_contours:
    area = cv2.contourArea(contour)

    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = 0, 0

    object_data.append({'contour': contour, 'area': area, 'center': (cx, cy)})

object_data.sort(key=lambda x: x['area'])
smallest_object = object_data[0] if object_data else None
largest_object = object_data[-1] if object_data else None

for obj in object_data:
    cx, cy = obj['center']
    cv2.drawContours(image, [obj['contour']], -1, (0, 255, 0), 2)
    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, f'Objects: {len(filtered_contours)}', (10, 30), font, 1, (255, 0, 0), 2)
if largest_object:
    cv2.putText(image, f'Largest center: {largest_object["center"]}', (10, 70), font, 0.7, (255, 0, 0), 2)
if smallest_object:
    cv2.putText(image, f'Smallest center: {smallest_object["center"]}', (10, 110), font, 0.7, (255, 0, 0), 2)

result_path = 'result.jpg'
cv2.imwrite(result_path, image)
