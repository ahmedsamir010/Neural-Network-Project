# Gender and Age Prediction CNN Model

This project aims to predict the gender and age of individuals from facial images using Convolutional Neural Networks (CNNs).

## Dataset

The dataset used in this project is the UTKFace dataset, which contains facial images labeled with age, gender, and ethnicity. You can find the dataset [here](https://www.kaggle.com/datasets/jangedoo/utkface-new?rvi=1).

## Getting Started

To use this project, follow these steps:

1. Clone the repository to your local machine:

git clone https://github.com/your_username/your_repository.git


2. Install the required dependencies:


3. Run the Jupyter Notebook or Python script to train the model and make predictions.

## Usage

- `gender_age_prediction.ipynb`: Jupyter Notebook containing the code for training the model, making predictions, and visualizing results.
- `gender_age_prediction.py`: Python script equivalent to the Jupyter Notebook.
- `the_last_trained_model.keras`: Pre-trained model saved in Keras format.
- `requirements.txt`: List of dependencies required to run the project.

## Model Architecture

The CNN model architecture consists of several convolutional layers followed by max-pooling layers, fully connected layers with dropout, and output layers for gender and age prediction.

## Results

The model achieves accuracy metrics for gender and age prediction, as shown in the graphs plotted from training history.

## Testing

To test the model on new images, use the provided functions in the script or notebook. Ensure that the images are preprocessed before feeding them into the model for prediction.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [UTKFace Dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new?rvi=1) - Dataset used for training and testing.
- Libraries used: TensorFlow, Keras, NumPy, pandas, Matplotlib, Seaborn.
