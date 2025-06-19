# Pneumonia Detection from Chest X-rays

## Introduction
This project demonstrates how to use deep learning to detect pneumonia in chest X-ray images. It uses a convolutional neural network with transfer learning to classify X-rays as either **PNEUMONIA** or **NORMAL**.

## Dataset
The [Chest X-ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle is used. It contains training, validation, and test images. The `src/data_loader.py` script downloads and prepares the dataset using the Kaggle API.

## Tools & Libraries
- Python 3.10+
- TensorFlow & Keras
- NumPy & Pandas
- Matplotlib & Seaborn
- Scikit-learn
- Streamlit (optional app)

## Model Architecture
A ResNet50 backbone pretrained on ImageNet is fine-tuned with additional layers:
- Global Average Pooling
- Dropout
- Dense layer with sigmoid activation for binary classification

Callbacks such as `EarlyStopping` and `ModelCheckpoint` are used during training.

## Results Summary
After training for several epochs, the model reaches ~90% accuracy on the test set. Confusion matrix and Grad-CAM visualizations demonstrate where the model focuses when predicting pneumonia. Example metrics:

| Metric | Value |
| ------ | ----- |
| Test Accuracy | ~0.90 |
| F1-Score | ~0.91 |

Grad-CAM images highlight regions in the lungs indicative of pneumonia.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Authenticate Kaggle (see [Kaggle API](https://github.com/Kaggle/kaggle-api)).
3. Run training: `python src/train.py`
4. Trained models are saved to `models/` and logs/plots to `data/`.
5. Optionally run `streamlit run app.py` to launch the demo app.

## Training Your Own Model
- Place new images in a directory structure matching `chest_xray/train` and `chest_xray/test`.
- Modify parameters in `train.py` (batch size, epochs, etc.).
- Execute the script to train and evaluate.

## Notebook
A companion notebook in `notebooks/` demonstrates the full pipeline from data loading to Grad-CAM visualization.


## Sample Outputs
Running `train.py` saves evaluation plots and Grad-CAM images to the `data/` directory:

- `loss_curve.png` and `acc_curve.png` for training metrics
- `confusion_matrix.png` summarizing performance on the test set
- `app_gradcam.png` showing an example Grad-CAM heatmap
