# Smart Office Object Detection Dashboard

This repository contains a comprehensive solution for detecting objects in a smart office environment using deep learning. We leverage the YOLOv8 model from Ultralytics, trained on a custom dataset, to identify objects such as **Person**, **Chair**, **Monitor**, **Keyboard**, **Laptop**, and **Phone**. The project includes training scripts, inference capabilities, and a Streamlit-based dashboard for real-time visualization of detection results on images, videos, and GIFs.

## Introduction

The Smart Office Object Detection project aims to automate the identification of office-related objects to enhance workspace management, security, and efficiency. Using the YOLOv8 architecture, we conducted hyperparameter sweeps with Weights & Biases (W&B) to optimize model performance, achieving a balance between accuracy and speed. The trained model is integrated into a user-friendly Streamlit dashboard, allowing users to upload media and view annotated results in real-time. This solution is particularly useful for office monitoring systems, inventory tracking, and smart environment analysis.

### Detected Objects
- Person
- Chair
- Monitor
- Keyboard
- Laptop
- Phone

### Models Used
- **YOLOv8 Variants**: The training process explored multiple YOLOv8 models (`yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`) to find the best fit. The final model used is `yolov8l`, selected for its superior performance after hyperparameter tuning, details (`src/model`).

## Setup Instructions

### Prerequisites
- Python 3.11 or higher
- Git (for cloning the repository)
- CUDA-enabled GPU (optional but recommended for faster training and inference)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/office-object-detection-hackathon_challenge.git
   cd office-object-detection-hackathon_challenge
   ```

2. **Create a Virtual Environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   Install the required packages listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Weights & Biases**
   - Sign up or log in at [wandb.ai](https://wandb.ai).
   - Obtain your API key from the W&B dashboard and log in via the command line:
     ```bash
     wandb login
     ```
   - Enter your API key when prompted.

5. **Prepare the Dataset**
   - Place your dataset in the `data` directory with the following structure:
     ```
     data/
     ├── train/
     │   ├── images/
     │   └── labels/
     ├── valid/
     │   ├── images/
     │   └── labels/
     ├── testing/
     │   ├── images/
     │   └── labels/
     └── data.yaml
     ```
   - Update `data/data.yaml` with the paths to your image directories and the list of classes (Person, Chair, Monitor, Keyboard, Laptop, Phone).

6. **Update Model Path**
   - In `app.py`, ensure the `model_path = '../model/best_model.pt'` points to the location of your trained model (e.g., `src/model/best_model.pt` after training).
   - In `training.ipynb` and `inference.py`, update the model path to match your trained model location.

## Running the Project

### 1. Training the Model
The training process uses a Jupyter Notebook with hyperparameter sweeps managed by W&B.

- **Open the Notebook**
  ```bash
  jupyter notebook training.ipynb
  ```

- **Execute the Cells**
  - Run the cells sequentially to set up the environment, configure the sweep, and initiate training.
  - The sweep ID will be generated, and you can monitor progress on the W&B dashboard.
  - The best model weights will be saved in `src/model/best_model.pt` (or similar, depending on the run).

- **Notes**
  - Training requires a GPU for optimal performance. Ensure CUDA is set up if using `device='cuda'`.
  - Adjust `epochs`, `batch_size`, and other parameters in the sweep configuration as needed.

### 2. Running Inference
To test the model on individual images:

- **Update the Script**
  - Edit `inference.py` to point `test_dir` or `image_path` to your test image or directory.

- **Run the Script**
  ```bash
  python inference.py
  ```
  - Annotated images will be saved in the `runs/detect` directory. Check the console for the output location.

### 3. Launching the Streamlit Dashboard
The dashboard allows real-time object detection on uploaded images, videos, and GIFs.

- **Run the App**
  ```bash
  streamlit run app.py
  ```
  - Open the URL provided (e.g., `http://localhost:8501`) in your browser.

- **Usage**
  - Select the input type (Image, Video, or GIF) from the sidebar.
  - Upload a file to see the annotated results in the main area.
  - Detection results (objects, confidence, and bounding boxes) are displayed below the annotated media.

- **Notes**
  - Ensure the `temp` directory has write permissions.
  - The dashboard uses the model specified in `app.py`. Update the path if necessary.
  - For videos, detection is real-time, and playback stops at the end.

## Project Structure
```
smart-office-detection/
├── data/              # Dataset (train, valid, data.yaml)
├── src/
│   ├── dashboard/     # Streamlit app files
|   |   ├── temp/              # Temporary files for uploads
│   │   ├── app.py
│   ├── training.ipynb # Training notebook
│   ├── inference.py   # Inference script
├── model/             # Trained model weights (e.g., best_model.pt)
├── requirements.txt   # Dependencies
└── README.md          # This file
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Special thanks to the hackathon challenge organized by HumbleBeeAI (https://humblebee.ai/), which inspired and supported the development of this project.
- Ultralytics for the YOLOv8 framework (https://www.ultralytics.com/).
- Weights & Biases for hyperparameter tuning and visualization (https://wandb.ai/site/).
- Streamlit for the interactive dashboard (https://streamlit.io/).