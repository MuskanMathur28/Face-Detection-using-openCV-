
import cv2

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read the input image
image = cv2.imread('input_image.jpg')  # Replace with your image path

# Convert the image to grayscale (face detection works better on grayscale images)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Show the image with detected faces
cv2.imshow('Face Detection', image)

# Wait for a key press and close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the output image
cv2.imwrite('output_image.jpg', image)


