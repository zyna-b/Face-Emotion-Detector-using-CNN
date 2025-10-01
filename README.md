# Face Emotion Detector

A Streamlit front-end for your FER2013-based convolutional neural network. Upload a face photo or take a snapshot, and the app will return the dominant emotion along with a confidence breakdown across all seven classes.

## Highlights

- ✅ Clean, responsive UI with sidebar guidance and subtle styling
- 🎥 Supports both file uploads and real-time camera capture
- 📊 Displays confidence scores and a sortable breakdown for every emotion class
- 🚀 Ready for one-click deployment on Streamlit Community Cloud

## Prerequisites

- Python 3.9 or newer (3.9 is recommended to match the original training environment)
- The trained model file `fer2013_emotion_cnn.h5` placed in the project root (already included)

## Getting started locally

1. (Recommended) Create and activate a virtual environment.
2. Install the dependencies from `requirements.txt`.
3. Launch the Streamlit app.

Once Streamlit finishes starting up, it will open a browser window showing the interface. Upload a cropped face image or use the built-in camera widget to get predictions. For best performance, prefer well-lit, front-facing photos.

## Running the built-in checks

Run the lightweight unit tests to ensure the preprocessing pipeline behaves as expected:

```
python -m unittest
```

## Deployment on Streamlit Community Cloud

1. Push the project (including `fer2013_emotion_cnn.h5`, `app.py`, and `requirements.txt`) to a public GitHub repository.
2. Visit [share.streamlit.io](https://share.streamlit.io/), sign in, and select **New app**.
3. Choose the repository and branch, set the main file to `app.py`, and click **Deploy**.
4. In the **Advanced settings**, add `requirements.txt` so Streamlit installs the right dependencies automatically.

The first deployment may take a few minutes while Streamlit downloads TensorFlow. Subsequent runs are cached and much faster.

## Project layout

```
.
├── app.py                  # Streamlit application entry point
├── fer2013_emotion_cnn.h5  # Trained FER2013 CNN weights
├── requirements.txt        # Runtime dependencies
├── tests
│   └── test_preprocess.py  # Unit tests for the preprocessing pipeline
└── README.md               # This guide
```

## Customisation ideas

- Swap in an updated or better-performing model by replacing `fer2013_emotion_cnn.h5`.
- Extend the UI with webcam-based live inference loops or batch predictions.
- Log predictions to a database or analytics service to monitor real-world usage.

Happy building! 🎉
