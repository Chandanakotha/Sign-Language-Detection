# Sign Language Detection

A real-time American Sign Language (ASL) finger spelling detection system using Convolutional Neural Networks (CNN) and MediaPipe hand tracking.

## Features

- **Real-time Detection**: Recognize ASL hand gestures using your webcam
- **Text Output**: Detected signs appear as text in real-time
- **Text-to-Speech**: Convert recognized text to speech
- **Smart Prediction**: Word suggestions based on context
- **High Accuracy**: 97-99% accuracy with proper lighting
- **Dark Mode Support**: Easy on the eyes during extended use

## Project Structure

```
Sign Language Detection/
├── final_pred.py           # Main application with GUI
├── my_hand_tracker.py      # Hand tracking module (MediaPipe)
├── cnn8grps_rad1_model.h5  # Pre-trained CNN model
├── hand_landmarker.task    # MediaPipe hand landmark model
├── white.jpg               # Background template for skeleton
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── data/                   # Dataset directory (optional)
    └── AtoZ_3.1/           # Training images A-Z
```

## Quick Start

### 1. Install Dependencies

```bash
cd "Sign Language Detection"
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python final_pred.py
```

Or if you have both Python 2 and 3:

```bash
python3 final_pred.py
```

### 3. Using the Application

1. **Position your hand** in front of the webcam
2. **Show a sign** - hold it steady for ~0.5 seconds
3. **Watch the character appear** in the "Sentence" field
4. **Change signs** to type more letters

#### Gesture Controls

| Gesture | Action |
|---------|--------|
| ASL Letters A-Z | Type corresponding letter |
| Thumbs Up | Add space |
| Thumbs Down/Left | Backspace |

#### GUI Buttons

- **Backspace (⌫)**: Delete last character
- **Clear**: Clear entire sentence
- **Speak**: Convert text to speech
- **Save**: Save sentence to file
- **Copy**: Copy to clipboard
- **Dark Mode (🌙/☀️)**: Toggle dark mode

## Requirements

### Hardware
- Webcam

### Software
- Python 3.9+
- macOS / Windows / Linux

### Dependencies
- opencv-python
- numpy
- tensorflow
- keras
- mediapipe
- pyttsx3
- pyenchant
- Pillow
- cvzone
- protobuf

## Architecture

### Data Processing Pipeline
1. **Hand Detection**: MediaPipe detects hand landmarks from webcam
2. **Skeleton Drawing**: Hand landmarks drawn on white background
3. **CNN Classification**: Pre-trained model classifies gesture into 8 groups
4. **Fine Classification**: Secondary logic refines to specific letter

### Group Classification
The model classifies gestures into 8 groups for better accuracy:
- Group 0: A, E, M, N, S, T
- Group 1: B, D, F, I, K, R, U, V, W
- Group 2: C, O
- Group 3: G, H
- Group 4: L
- Group 5: P, Q, Z
- Group 6: X
- Group 7: J, Y

## Troubleshooting

### Camera not detected
- Check webcam permissions
- Ensure no other application is using the camera

### Low accuracy
- Ensure good lighting
- Use a clean, contrasting background
- Hold gestures steady for ~0.5 seconds

### Dependencies installation failed
```bash
# Try upgrading pip first
pip install --upgrade pip

# Then install requirements
pip install -r requirements.txt
```

## License

This project is open source and available under the MIT License.

## Acknowledgments

- MediaPipe for hand tracking
- TensorFlow/Keras for the CNN model
- OpenCV for image processing

