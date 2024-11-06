import cv2  # OpenCV library for image and video processing
import pytesseract  # Wrapper for Tesseract OCR to extract text from images
import sqlite3  # SQLite for database operations
import os  # To handle file and path operations
from datetime import datetime  # For working with timestamps

# Ensure that the Tesseract executable path is correctly set (only for Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Database setup
conn = sqlite3.connect('license_plate.db')  # Create/connect to an SQLite database
c = conn.cursor()  # Create a cursor object for executing SQL queries

# Create table if it doesn't exist and also save the timestamp and text in a column format
c.execute('''
    CREATE TABLE IF NOT EXISTS LicensePlates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,  # Auto-incrementing unique ID for each entry
        plate_number TEXT,  # Column to store the detected license plate number
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,  # Column to store the time of detection
        confidence INTEGER  # Column to store the confidence score of OCR detection
    )
''')
conn.commit()  # Commit the changes to the database


# Function to preprocess image (denoising, thresholding, etc.)
def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Denoise the grayscale image to remove noise
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    
    # Apply Gaussian blur to further reduce noise
    blurred = cv2.GaussianBlur(denoised, (5, 5), 0)
    
    # Apply adaptive thresholding to convert image to binary (black and white)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Create a rectangular kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    # Apply morphological opening to remove small noise from the image
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Apply morphological closing to fill small holes in the image
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    
    # Return the preprocessed image
    return closing


# Function to detect and extract the license plate from an image with confidence filtering
def detect_license_plate(image):
    # Preprocess the image using the preprocessing function
    preprocessed_image = preprocess_image(image)

    # Perform OCR on the preprocessed image using Tesseract, returning data with text and confidence scores
    data = pytesseract.image_to_data(preprocessed_image, config='--psm 7 --oem 1', output_type=pytesseract.Output.DICT)
    
    plate_text = ""  # Variable to hold the detected license plate text
    max_confidence = 0  # Variable to track the highest confidence score
    n_boxes = len(data['text'])  # Number of text boxes detected by OCR

    # Loop through each detected text box
    for i in range(n_boxes):
        # Check if the confidence score of the detected text is greater than 60
        if int(data['conf'][i]) > 60:
            # Append the detected text to the plate_text string
            plate_text += data['text'][i]
            # Update the max confidence score if the current confidence is higher
            max_confidence = max(max_confidence, int(data['conf'][i]))

    # Remove any non-alphanumeric characters from the detected text
    plate_text = ''.join(e for e in plate_text if e.isalnum())

    # Return the detected license plate and confidence if the plate length is greater than 5 characters
    if len(plate_text) > 5:
        return plate_text, max_confidence
    # Return None if no valid plate is detected
    return None, None


# Function to store detected plate in the database with timestamp and confidence
def add_plate_to_db(plate_number, confidence):
    # Check if a valid plate number is passed
    if plate_number:
        # Get the current timestamp
        timestamp = datetime.now()
        
        # Insert the detected plate, timestamp, and confidence into the LicensePlates table
        c.execute("INSERT INTO LicensePlates (plate_number, timestamp, confidence) VALUES (?, ?, ?)",
                  (plate_number, timestamp, confidence))
        conn.commit()  # Commit the changes to the database
        
        # Print confirmation message with plate number, timestamp, and confidence score
        print(f"License Plate '{plate_number}' stored in the database with timestamp {timestamp} and confidence {confidence}.")


# Function to process video and detect license plates with frame skipping and change detection
def process_video(video_path):
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' does not exist.")  # Print error if file is not found
        return

    print(f"Processing video: {video_path}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was successfully opened
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    frame_count = 0  # Variable to count the frames processed
    last_detected_plate = ""  # To track the last detected plate to avoid duplicates
    previous_frame = None  # Variable to store the previous frame for change detection

    # Loop through the video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame from the video
        if not ret:
            print("Failed to grab frame or video ended")  # Break the loop if there are no more frames or error occurs
            break

        frame_count += 1  # Increment frame count

        # Skip 9 out of 10 frames to reduce processing load
        if frame_count % 10 != 0:
            continue

        # Check if there is a previous frame for comparison
        if previous_frame is not None:
            # Calculate the difference between the current and previous frames
            frame_diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                     cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY))
            # If the difference between the frames is small, skip processing this frame
            if cv2.countNonZero(frame_diff) < 5000:
                previous_frame = frame  # Update the previous frame
                continue

        previous_frame = frame  # Update the previous frame to the current frame

        print(f"Processing frame {frame_count}...")

        # Detect the license plate in the current frame
        plate_number, confidence = detect_license_plate(frame)

        # If a valid plate is detected and it's different from the last detected plate
        if plate_number and plate_number != last_detected_plate:
            print(f"Detected License Plate: {plate_number}")  # Print the detected plate
            add_plate_to_db(plate_number, confidence)  # Add the detected plate to the database
            last_detected_plate = plate_number  # Update the last detected plate

        # Display the current frame
        cv2.imshow('Video', frame)

        # Check for 'q' key press to exit the video processing loop
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()  # Release the video capture object
    cv2.destroyAllWindows()  # Close any OpenCV windows


# Main function to drive the system
def main():
    while True:
        print("\nLicense Plate Tracking System")
        print("1. Process a video for license plates")
        print("2. Exit")

        choice = input("Enter your choice: ")  # Ask the user for a choice

        if choice == '1':
            # Ask the user for the video path and process the video
            video_path = input("Enter the full path of the video: ").strip()
            process_video(video_path)  # Call the function to process the video
        elif choice == '2':
            print("Exiting the system.")
            break  # Exit the program
        else:
            print("Invalid choice, please try again.")  # Handle invalid input


if __name__ == '__main__':
    main()  # Call the main function when the script is executed

# Close the database connection when the script ends
conn.close()
