# 🎭 Face Emotion Detector using CNN

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://face-emotion-detector-using-cnn-6dsvkxbugujfqpchdgwwcn.streamlit.app/)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.20](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A real-time facial emotion detection web application built with **Deep Learning** and **Streamlit**. The app uses a Convolutional Neural Network (CNN) trained on the **FER2013 dataset** to classify emotions from facial images into 7 categories.

![App Demo](https://img.shields.io/badge/Status-Live-success)

## 🎯 Features

- **🎥 Real-time Detection**: Upload images or use your camera for instant emotion recognition
- **📊 Confidence Scores**: Visual breakdown of prediction probabilities for all 7 emotions
- **🎨 Clean UI**: Modern, responsive interface with intuitive controls
- **⚡ Fast Inference**: Optimized CNN model for quick predictions
- **� Cloud Deployed**: Accessible anywhere via Streamlit Community Cloud

## 🧠 Emotion Classes

The model can detect 7 different emotions:

1. � **Angry**
2. 🤢 **Disgust**
3. 😨 **Fear**
4. 😊 **Happy**
5. 😢 **Sad**
6. 😲 **Surprise**
7. 😐 **Neutral**

## 🚀 Live Demo

Try the app here: **[Face Emotion Detector](https://face-emotion-detector-using-cnn-6dsvkxbugujfqpchdgwwcn.streamlit.app/)**

## 📋 Prerequisites

- Python 3.11+ (tested on 3.13)
- pip package manager
- (Optional) Virtual environment tool

## 🛠️ Installation & Setup

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/zyna-b/Face-Emotion-Detector-using-CNN.git
   cd Face-Emotion-Detector-using-CNN
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`

## 📁 Project Structure

```
Face-Emotion-Detector-using-CNN/
│
├── app.py                          # Main Streamlit application
├── CNN_model.ipynb                 # Model training notebook
├── fer2013_emotion_cnn.h5          # Trained CNN model (57.75 MB)
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version for deployment
├── packages.txt                    # System dependencies
│
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
│
├── tests/
│   └── test_preprocess.py          # Unit tests for preprocessing
│
├── README.md                       # Project documentation
├── LICENSE                         # MIT License
└── DEPLOYMENT.md                   # Deployment guide
```

## 🧪 Model Architecture

The CNN model consists of:

- **Input Layer**: 48×48×3 RGB images
- **Convolutional Blocks**: 3 blocks with increasing filters (64, 128, 256)
- **Batch Normalization**: After each convolutional layer
- **MaxPooling**: 2×2 pooling to reduce spatial dimensions
- **Dropout Layers**: 25% dropout for regularization
- **Dense Layers**: 256-unit fully connected layer
- **Output Layer**: 7-unit softmax for emotion classification

### Training Details

- **Dataset**: FER2013 (cleaned version)
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Categorical Crossentropy
- **Data Augmentation**: Rotation, shifting, zoom, horizontal flip
- **Class Balancing**: Computed class weights for imbalanced dataset
- **Callbacks**: Early stopping and learning rate reduction

## 📊 Model Performance

- **Test Accuracy**: ~65-70% (on FER2013 validation set)
- **Best Performing Classes**: Happy, Surprise, Neutral
- **Challenging Classes**: Disgust, Fear (due to limited samples)

## 🎮 Usage

### Upload Image
1. Click "Browse files" in the sidebar
2. Select a facial image (JPG, JPEG, PNG)
3. View the detected emotion and confidence scores

### Camera Capture
1. Click "Take a picture" button
2. Allow camera permissions
3. Capture your photo
4. Get instant emotion prediction

### Tips for Best Results
- ✅ Use well-lit, front-facing photos
- ✅ Ensure the face is clearly visible
- ✅ Avoid extreme angles or occlusions
- ✅ Single face per image works best

## 🧪 Running Tests

Execute unit tests to verify preprocessing functionality:

```bash
python -m unittest discover tests
```

Or run specific test:

```bash
python -m unittest tests.test_preprocess
```

## 🚢 Deployment

### Streamlit Community Cloud

1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io/)
3. Connect your repository
4. Set main file to `app.py`
5. Deploy!

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

## 🔧 Configuration

### Requirements
- TensorFlow ≥ 2.20.0
- Streamlit ≥ 1.38
- NumPy ≥ 2.0
- Pillow, Pandas, Matplotlib, h5py

### System Dependencies
- `libgomp1` (for TensorFlow on Linux)

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contribution
- Improve model accuracy with advanced architectures
- Add support for video/real-time webcam processing
- Implement emotion tracking over time
- Add multi-face detection
- Create visualization dashboard for emotion trends

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FER2013 Dataset**: [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Web application framework
- **Community**: All contributors and users

## 📧 Contact

**Zyna** - [@zyna-b](https://github.com/zyna-b)

Project Link: [https://github.com/zyna-b/Face-Emotion-Detector-using-CNN](https://github.com/zyna-b/Face-Emotion-Detector-using-CNN)

---

⭐ **Star this repo** if you find it helpful!

Made with ❤️ using Python, TensorFlow, and Streamlit
