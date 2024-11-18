import cv2
from tkinter import Tk, Button, Label, Text, Scrollbar, Toplevel
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from ultralytics import YOLO


def process_image():
    # Load the model 
    model = YOLO('best.pt')

    # Ask user to select an image file
    image_path = askopenfilename()

    # Read the image
    frame = cv2.imread(image_path)

    # Check if the image is successfully read
    if frame is None:
        print("Error: Could not read the image.")
        return

    # Get predictions
    results = model(frame)
    boxes = results[0].boxes

    # Create a new window for results
    result_window = Toplevel(window)
    result_window.title("Results")

    # Display predictions in the new window
    result_text = Text(result_window, height=10, width=50)
    result_text.pack()

    # Add a scrollbar to the text area
    scrollbar = Scrollbar(result_window, command=result_text.yview)
    scrollbar.pack(side="right", fill="y")
    result_text.config(yscrollcommand=scrollbar.set)

    # Insert predictions into the text area
    for box in boxes:
        result_text.insert('end', f"{box.xyxy}\n")

    # Display the image with predictions
    cv2.imshow('Image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Create Tkinter window
window = Tk()
window.title("Image Processing with YOLO")

# Create a label
label = Label(window, text="Click 'Process Image' to select an image and see predictions.")
label.pack()

# Create a button to trigger image processing
process_button = Button(window, text="Process Image", command=process_image)
process_button.pack()

# Run the Tkinter event loop
window.mainloop()
