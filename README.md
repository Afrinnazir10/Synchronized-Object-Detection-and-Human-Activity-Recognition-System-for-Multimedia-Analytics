

\# Synchronized Object Detection and Human Activity Recognition



<p align="center">

&#x20; <b>Real-Time Object Detection and Human Activity Recognition for Multimedia Analytics</b>

</p>



<p align="center">

&#x20; <img src="https://img.shields.io/badge/Python-3.8-blue?logo=python\&logoColor=white" alt="Python">

&#x20; <img src="https://img.shields.io/badge/YOLOv8-Object%20Detection-purple" alt="YOLOv8">

&#x20; <img src="https://img.shields.io/badge/MediaPipe-Pose%20Estimation-orange" alt="MediaPipe">

&#x20; <img src="https://img.shields.io/badge/TensorFlow-LSTM-orange?logo=tensorflow" alt="TensorFlow">

</p>



\## Overview



This project combines \*\*object detection\*\* and \*\*human activity recognition (HAR)\*\* to analyze video streams.



It integrates YOLOv8 for detecting objects, MediaPipe Pose for extracting human body landmarks, and a trained LSTM model for classifying human activities.



The system processes video frames and displays detected objects alongside the predicted human activity.



\## Key Features



\- Detects objects in video frames using YOLOv8.

\- Extracts 33 human pose landmarks using MediaPipe Pose.

\- Classifies human activities using a trained LSTM model.

\- Combines object detection and activity recognition in one pipeline.

\- Supports video-based testing.

\- Displays detection and activity predictions during processing.



\## System Architecture



```text

&#x20;            Input Video

&#x20;                 |

&#x20;                 v

&#x20;          Video Frame Capture

&#x20;                 |

&#x20;         +-------+-------+

&#x20;         |               |

&#x20;         v               v

&#x20;   YOLOv8 Detection   MediaPipe Pose

&#x20;         |               |

&#x20;         v               v

&#x20;   Detected Objects   33 Pose Landmarks

&#x20;         |               |

&#x20;         |               v

&#x20;         |          LSTM Classifier

&#x20;         |               |

&#x20;         v               v

&#x20;      Object Labels   Activity Label

&#x20;         |               |

&#x20;         +-------+-------+

&#x20;                 |

&#x20;                 v

&#x20;       Combined Video Analysis

```



\## Technologies Used



| Component | Technology | Purpose |

|---|---|---|

| Object Detection | YOLOv8 | Detects objects in video frames |

| Pose Estimation | MediaPipe Pose | Extracts human body landmarks |

| Activity Recognition | LSTM | Classifies human activities from pose sequences |

| Deep Learning | TensorFlow / Keras | Loads and runs the trained HAR model |

| Computer Vision | OpenCV | Reads and processes video frames |

| Programming Language | Python | Integrates the complete pipeline |



\## Recognized Activities



The trained HAR model contains 10 activity classes:



\- Carrying

\- Checking the time on a watch

\- Drinking

\- Eating

\- Reading

\- Standing

\- Talking through phone

\- Using a computer

\- Walking

\- Writing



\## Project Structure



```text

.

├── models/

│   └── HAR3333.h5

├── model\_weights/

│   └── yolov8x.pt

├── src/

│   └── integration.py

├── tests/

│   └── test\_integration.py

├── results/

├── coco.names

├── requirements.txt

├── .gitignore

├── .gitattributes

└── README.md

```



Additional files may be present depending on the development environment. Virtual environments, temporary videos, and local IDE settings should not be committed.



\## Installation



\### 1. Clone the repository



```bash

git clone https://github.com/Afrinnazir/Synchronized-Object-Detection-and-Human-Activity-Recognition-System-for-Multimedia-Analytics.git

cd Synchronized-Object-Detection-and-Human-Activity-Recognition-System-for-Multimedia-Analytics

```



\### 2. Create and activate a virtual environment



Windows PowerShell:



```powershell

python -m venv venv

.\\venv\\Scripts\\Activate.ps1

```



\### 3. Install dependencies



```bash

python -m pip install --upgrade pip

pip install -r requirements.txt

```



\### 4. Retrieve model weights



The YOLOv8 model is tracked using Git LFS.



If the model file is not downloaded when cloning, install Git LFS and pull the model:



```bash

git lfs install

git lfs pull

```



The HAR model is stored at `models/HAR3333.h5`.



\## Run the Project



Run the integrated pipeline from the repository root:



```powershell

python src/integration.py "path\\to\\your\\video.mp4"

```



Example:



```powershell

python src/integration.py "C:\\Users\\YourName\\Videos\\sample.mp4"

```



The application processes the video and displays object detections and activity predictions.



Press \*\*Q\*\* in the video window to stop processing, if supported by the running script.



\## Model Input and Output



| Model | Input | Output |

|---|---|---|

| YOLOv8 | Video frame | Object classes and bounding boxes |

| MediaPipe Pose | Video frame containing a person | 33 pose landmarks |

| LSTM HAR | Sequence of pose landmarks | One of 10 activity classes |



The HAR model expects pose data shaped as `(33, 3)` for an individual frame, with temporal sequences prepared by the integration pipeline.



\## Testing



The project was validated using a video sample containing a drinking activity.



The integrated pipeline successfully loaded the YOLOv8 and HAR models, processed video frames, detected objects including a person and bottle, and produced activity predictions.



\## Applications



\- Multimedia video analysis

\- Human activity recognition

\- Intelligent video monitoring

\- Computer vision research

\- Human-object interaction analysis



\## Future Improvements



\- Add command-line options for input and output paths.

\- Improve activity recognition through temporal smoothing.

\- Add support for batch video processing.

\- Provide quantitative evaluation on a documented test dataset.

\- Develop a graphical interface for easier use.



\## Author



\*\*Afrin Nazir\*\*  

M.Tech Artificial Intelligence and Data Science



\## License



This repository is intended for academic and research purposes. Review the licenses of the included models, datasets, and dependencies before redistribution or commercial use.



