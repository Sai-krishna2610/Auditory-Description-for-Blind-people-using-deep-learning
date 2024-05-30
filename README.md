Project Overview
This project aims to provide auditory descriptions of images to assist blind and visually impaired individuals. By leveraging deep learning techniques, specifically convolutional neural networks (CNNs) and recurrent neural networks (RNNs), we can generate descriptive sentences from images and convert these descriptions into audio. This project uses the Flickr8k and Flickr30k image datasets available on Kaggle.

Datasets
We utilized two prominent datasets:

Flickr8k: [Kaggle link](https://www.kaggle.com/datasets/adityajn105/flickr8k)
Flickr30k: [Kaggle link](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)
Both datasets consist of images with associated captions, which are used to train our models.

Project Steps
Data Preparation:

Downloading and Splitting the Dataset: The dataset was downloaded and split into training and testing sets. This division ensures that the model is evaluated on unseen data to gauge its performance.
Importing Image Data: The images were imported into the project using standard data handling libraries in Python.
Preprocessing:

Text Preprocessing: Each caption associated with the images was preprocessed to remove punctuation marks and standardize the format. Special tokens startseq and endseq were added to the beginning and end of each caption to signify the start and end of a description.
Feature Extraction:

Using InceptionV3: We utilized the InceptionV3 model, pretrained on the ImageNet dataset, to extract features from the images. The InceptionV3 model is a powerful CNN that excels at image recognition tasks.
Feature Extraction Process: The images were passed through the InceptionV3 model to extract high-level features, which serve as input for the next stage.
Description Generation:

LSTM Network: The extracted image features were fed into a Long Short-Term Memory (LSTM) network to generate descriptive sentences. LSTM networks are a type of RNN that are well-suited for sequence prediction tasks like language modeling.
Model Training: The LSTM network was trained on the preprocessed captions and extracted features to learn how to map image features to coherent descriptions.
Audio Conversion:

Text-to-Speech: The generated text descriptions were then converted to audio using text-to-speech (TTS) technology. This step ensures that the descriptions can be heard by blind users.
