# BERT-FineTunned-Model
This repository contains code for BERT model building using Tensorflow library.
Project: BERT-Based Text Classification Model
Overview:
This project implements a text classification model using BERT (Bidirectional Encoder Representations from Transformers) integrated with TensorFlow and Keras. It also includes functionality for saving the trained model, tokenizer, and OneHotEncoder for efficient loading and reuse.

Key Features:
BERT-based Model: The model uses the pre-trained BERT architecture (via Hugging Face Transformers library) for text classification tasks. The TFBertModel is combined with a Keras sequential model for fine-tuning.
Training and Evaluation: The training dataset is tokenized using BERT's tokenizer, and the classification is done using a multi-layer neural network. The project includes model evaluation, confusion matrix generation, and other performance metrics.
Saving and Loading: The model, along with the tokenizer and OneHotEncoder, is saved and loaded for reuse, making it easier to resume training or perform inference without retraining.

Dependencies:
Python 3.x
TensorFlow
Hugging Face Transformers
Scikit-learn (for OneHotEncoder)
Pandas and NumPy (for data manipulation)
Future Enhancements:
Implement cross-validation to improve the robustness of the model.
Add more advanced metrics such as F1-score, precision, and recall.
Allow dynamic length tokenization for better handling of variable-length inputs.

Components:
Model Architecture:

The BERT model is wrapped in a Keras sequential model. It includes layers such as a dense output layer with softmax activation for multi-class classification.
Tokenizer and OneHotEncoder:

The BERT tokenizer is used to process input text data into token IDs and attention masks.
A OneHotEncoder is used to transform categorical labels into one-hot encoded format for training and prediction.
Prediction Function:

A custom Make_Predictions function is designed to handle inference for the model. It efficiently retrieves predicted labels and converts them back into their original format using the OneHotEncoder.
Error Handling:

The code includes error handling for common issues encountered during model training, saving, and loading. For example, ensuring correct input shapes for models, properly saving the model’s custom layers, and handling serialization of the tokenizer.
Files:
BERT_Cased_Classification_Model.h5: This file stores the trained BERT classification model.
BERT_Tokenizer: Directory containing the saved tokenizer files, essential for encoding new text during inference.
OneHotEncoder: The one-hot encoded label mapping is serialized for consistent label transformation during predictions.
