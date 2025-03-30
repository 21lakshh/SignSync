# SignSync - Learning App for Deaf and Mute

SignSync is an AI-powered learning application designed to bridge communication gaps for the Deaf and Mute community. The application utilizes advanced gesture detection and natural language processing techniques to convert American Sign Language (ASL) gestures into human-readable text and vice versa.

## 🚀 Features
- **Gesture Detection Model**: Uses OpenCV and MediaPipe with custom training to detect specific hand gestures.
- **Real-Time Gesture Processing**: Extracts gestures from video input, creating a structured sequence.
- **LLM Integration**: Converts detected gestures into meaningful, grammatically correct human language.
- **ASL Text Transformation**: Takes ASL-based input and refines it into human-like conversational text.
- **Scalability**: Expandable dataset and gesture recognition for future enhancements.

## 📹 Demo Video
Check out our demo video showcasing the features of SignSync:

![SignSync Demo](SignSync.mp4)

## 🏗️ Project Structure
```
SignSync/
│── model/
│   ├── keypoint_classifier/
│   │   ├── keypoint_classifier_label.csv
│   │   ├── keypoint_classifier.hdf5
│   │   ├── keypoint_classifier.keras
│   │   ├── keypoint_classifier.py
│   │   ├── keypoint_classifier.tflite
│   │   ├── keypoint.csv
│   ├── point_history_classifier/
│   │   ├── point_history_classifier_label.csv
│   │   ├── point_history_classifier.hdf5
│   │   ├── point_history_classifier.py
│   │   ├── point_history_classifier.tflite
│   │   ├── point_history.csv
│── templates/
│   ├── index.html
│── app_fastapi.py
│── app.py
│── asltohuman.py
│── cwd.py
│── keypoint_classification_EN.ipynb
│── point_history_classification.ipynb
│── README.md
│── requirements.txt
│── SignSync.mp4
```

## 🛠️ Technologies Used
- **OpenCV & MediaPipe**: For real-time hand gesture detection.
- **Large Language Model (LLM)**: Processes gesture sequences and ASL messages.
- **Python (TensorFlow)**: Machine learning framework for model training.
- **FastAPI**: Backend API implementation.
- **HTML Templates**: For web-based interaction.

## 📂 Dataset
The model is trained on an ASL-based dataset, with potential for expansion to support more gestures and dialects in the future.

## 🚀 Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/21lakshh/SignSync.git   
   cd SignSync
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app_fastapi.py
   ```

## 🔮 Future Enhancements
- **Expand ASL gesture recognition dataset**
- **Improve real-time processing performance**
- **Enhance UI for a seamless user experience**
- **Integrate text-to-sign video generation**
- **Create a website for seamless communication between two users**

## 🤝 Contribution
We welcome contributions! Feel free to open issues and submit pull requests to enhance SignSync.

---
🌟 *SignSync aims to create an inclusive world where communication barriers are minimized.*


