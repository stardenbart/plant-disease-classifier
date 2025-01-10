# Plant Disease Classification Using MobileNetV2

A web-based application for detecting plant diseases from leaf images using the **MobileNetV2** transfer learning model.

## Features
- Upload leaf images directly or via a URL.
- Predict the disease category based on the uploaded image.
- Provide detailed information about the detected disease.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/plant-disease-classification.git
   cd plant-disease-classification
2. Install required repository:
   ```bash
   pip install -r requirements.txt

## Library Used
The following libraries are used in the application:
- `tensorflow` and `keras`: For building and loading the VGG19 models.
- `streamlit`: To create an interactive web interface.
- `numpy`: For numerical computations.
- `Pillow`: For image processing.
- `TensorFlow Hub`: For loading the MobileNetV2 module.

## Project Structure
plant-disease-classification/
- ├── app.py            # Main application script
- ├── utils.py          # Utility functions for preprocessing and model architecture
- ├── model/            # Directory for trained models
- ├── info/             # Disease information files
- └── Data/             # Training data (optional, not included in the repo)

## Usage
1. Run the application:
   ```bash
   streamlit run app.py
2. Open the URL provided by Streamlit in your browser.
3. Upload a plant leaf image or provide an image URL to classify the disease.

## Model Information
The application uses a MobileNetV2 transfer learning model with the following architecture:
- Input size: 128x128x3
- Output: 38 classes

## Author
Abdullah Farauk/Stardenbart
