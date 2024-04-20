# NFR-Classification-CNN
Legacy code (2018) for non-functional requirement (NFR) sentence classification via a Convolutional Neural Network (CNN), now outdated. Current standards for Natural Language Processing (NLP) favor transformers (e.g., BERT, GPT).

### Multi-class Text Classification CNN Model README

This README document provides an overview of the Multi-class Text Classification Convolutional Neural Network (CNN) model implemented in a Jupyter Notebook.

> [!NOTE]
> This notebook contains test statements, commented-out sections, and lacks polishing. It's a lab version not initially meant for publication, provided here for reference.

#### Overview:

The implemented model is designed to classify text data into multiple classes using a Convolutional Neural Network architecture. It utilizes TensorFlow, a popular deep learning framework, for building and training the model. The model architecture follows the principles of a CNN tailored for text classification tasks.

#### Components:

1. **TextCNN Class:**
    - This class represents the CNN model for text classification.
    - It includes methods for initializing the model, defining placeholders for input data, embedding layer, convolutional and max-pooling layers, dropout, output layer, loss calculation, and accuracy computation.
    - The architecture consists of embedding layer, convolutional layers with max-pooling, dropout layer, and output layer.

2. **Data Preprocessing:**
    - The data preprocessing functions are provided to load and preprocess the input text data.
    - The `load_data_and_labels` function reads the input data from a CSV file, preprocesses the text, and encodes the labels into one-hot vectors.
    - The `clean_str` function is used to clean the text data by removing special characters, punctuation, and non-ASCII characters.

3. **Training Loop:**
    - The `batch_iter` function is used to load the training data in batches, which is essential for training large datasets without loading the entire dataset into memory.
    - It iterates over the dataset in mini-batches, shuffling the data for each epoch.

#### Instructions:

1. **Environment Setup:**
    - Ensure that TensorFlow and other required dependencies are installed in your Python environment.
    - Use a GPU-enabled environment if available for faster training, as indicated by the `tf.device('/device:GPU:0')` statements in the code.

2. **Data Preparation:**
    - Prepare your text data in CSV format with at least two columns: one for text sentences and another for corresponding class labels.
    - Ensure that the class labels are categorical and encoded appropriately.

3. **Model Configuration:**
    - Configure the parameters of the TextCNN model such as sequence length, number of classes, vocabulary size, embedding size, filter sizes, number of filters, and regularization parameters according to your dataset and requirements.

4. **Training:**
    - Instantiate the TextCNN class with the configured parameters.
    - Load the training data using the provided data loading functions.
    - Train the model using the training loop, adjusting hyperparameters as needed.
    - Monitor training progress, loss, and accuracy metrics.

5. **Evaluation:**
    - After training, evaluate the model's performance on a separate validation or test dataset.
    - Calculate classification metrics such as accuracy, precision, recall, and F1-score to assess the model's effectiveness.

6. **Deployment:**
    - Once satisfied with the model's performance, save the trained model for future inference or deployment in other applications.

#### Notes:

- This model is designed for multi-class text classification tasks, where each input text can belong to one of multiple predefined classes.
- Experiment with different hyperparameters, model architectures, and training strategies to optimize performance for your specific task and dataset.

#### References:

- TensorFlow documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- Convolutional Neural Networks for Sentence Classification: [https://arxiv.org/abs/1408.5882](https://arxiv.org/abs/1408.5882)
- Denny Britz's blog post on implementing a CNN for text classification in TensorFlow: [https://dennybritz.com/posts/wildml/implementing-a-cnn-for-text-classification-in-tensorflow/](https://dennybritz.com/posts/wildml/implementing-a-cnn-for-text-classification-in-tensorflow/)

#### Author:

This Multi-class Text Classification CNN model implementation, initially derived from Denny Britz [repository](https://github.com/dennybritz/cnn-text-classification-tf), was adapted by Cody Baker for neural network based NPL research at Towson University.
