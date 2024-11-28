import os
import sys
import datetime
from ultralytics import YOLO

def save_command_output(command, output):
    # Get the current date and time
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Generate the filename
    filename = f"{timestamp}.txt"
    
    # Write to the file
    with open(filename, 'w') as f:
        f.write("Command:\n")
        f.write(command + "\n\n")
        f.write("Output:\n")
        f.write(output)

def main():
    # Capture the command-line input
    command = ' '.join(sys.argv)
    
    # Load the model and train
    model = YOLO("yolo11s-pose.pt")
    results = model.train(data="datasets/roboflow_datasets/2024-11-27_vertical_aug/1127_vertical_aug/data.yaml", epochs=300, imgsz=640, degrees=0)

    # Convert results to string
    output = str(results)

    # Save the command-line input and results
    save_command_output(command, output)

if __name__ == "__main__":
    main()