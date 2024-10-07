# Attendance-automation-using-opencv

[TO BE UPDATED MORE , NOT FINAL VERSION]

This code implements a facial recognition-based attendance system using Python and OpenCV. Here's a description of its key components and functionality:
1)Face Detection and Recognition:

  Uses Haar Cascade classifier for face detection
  Implements LBPH (Local Binary Patterns Histograms) for face recognition

2)Dataset Management:

  Loads images from a 'dataset' directory
  Each person's images should be in a separate subdirectory named after them
  Includes data augmentation (flipping and rotation) to increase the dataset size

3)Image Preprocessing:

  Resizes faces to 100x100 pixels
  Applies histogram equalization for improved recognition

4)Training:

  Trains the LBPH face recognizer model using the loaded and augmented dataset

5)Real-time Recognition:

  Captures video from a camera (assumed to be camera index 1)
  Detects faces in each frame
  Attempts to recognize detected faces using the trained model
  Draws bounding boxes and labels recognized faces on the video feed

6)Attendance Marking:

  Records attendance in a CSV file named 'attendance.csv'
  Marks a person as present only once per session
  Includes date information for each attendance record

7)Main Execution:

  Runs the recognition system for 30 seconds
  Displays the video feed with recognition results
  Allows early termination by pressing 'q'

This system is designed for scenarios where you need to automatically track attendance using facial recognition, such as in classrooms or workplaces. It provides a basic framework that could be extended with additional features or improved recognition algorithms for more robust performance.
