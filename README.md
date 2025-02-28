# Fake News Detection Using Neural Networks

## Introduction
In this project, we develop a fake news detection system using various neural network architectures, including Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), and Bidirectional LSTM. The aim is to effectively distinguish between authentic and fabricated news articles.

## Dataset and Preprocessing
We utilize a dataset presumably titled "WELFake_Dataset.csv" for both training and testing. The preprocessing steps include:

- **Null Value Handling:** Rows with missing values are dropped to ensure data integrity.
- **Text Cleaning and Normalization:** We use regular expressions to retain only alphabetic characters and convert all text to lowercase.
- **Stemming:** The PorterStemmer is employed to reduce words to their root forms.
- **Stopword Removal:** English stopwords are filtered out using NLTK's stopwords library.

## One-Hot Encoding and Padding
- **Vocabulary Size:** Set to 10,000.
- **One-Hot Representation:** Texts are converted into one-hot encoded vectors.
- **Sequence Padding:** Sequences are standardized to a length of 20 using padding.

The final datasets, `X_final` and `y_final`, are prepared for the input to the models, and the data is split into training and testing sets with a 33% test size and a random state of 42 for reproducibility.

## Model Architecture
### RNN Model
- **Embedding Layer:** Maps our one-hot encoded text to a dense embedding space.
- **SimpleRNN Layer:** A simple RNN layer with 100 units.
- **Output Layer:** A dense layer with sigmoid activation for binary classification.

### LSTM Model
- **Embedding Layer:** Similar to the RNN model.
- **Dropout Layer:** Added to prevent overfitting, set at 30%.
- **LSTM Layer:** An LSTM layer with 100 units.
- **Output and Dropout Layers:** Similar to the RNN model, with an additional dropout layer.

### Bidirectional LSTM Model
- **Embedding Layer:** Similar to the other models.
- **Bidirectional LSTM Layer:** A bidirectional wrapper around the LSTM layer to capture dependencies from both past and future contexts.
- **Output and Dropout Layers:** Similar to the LSTM model.

## Training
Each model is trained over 20 epochs with a batch size of 32, using binary cross-entropy as the loss function and accuracy as the metric. Validation data is used to gauge the model's performance during training.

## Evaluation and Performance
After training, models are evaluated on the test set. We use accuracy, confusion matrix, and classification reports as metrics to assess each model's ability to classify news accurately.

- **RNN Model Accuracy:** Approximately 86.99%
- **LSTM Model Accuracy:** Approximately 89.88%
- **Bidirectional LSTM Model Accuracy:** Approximately 89.25%

These metrics indicate how each model performs in terms of precision, recall, and F1-score across the classes.

## Visualization
We visualize the training and validation accuracy and loss for each model to analyze their performance over the epochs. Additionally, histograms of validation accuracies and heatmaps of confusion matrices provide deeper insights into the models' predictive capabilities.

## Conclusion
This project establishes a foundational approach for using neural networks to detect fake news. The preprocessing, model architectures, and evaluations are carefully executed to maximize the detection accuracy. Future enhancements can include exploring more complex model architectures and implementing advanced text representation techniques to further enhance the system's performance.
