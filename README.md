# NLP-classification-scietific-papers
# Introduction
The ML model in this repo is trying to study the Kaggle competition "Science Topic Classification" (https://www.kaggle.com/competitions/science-topic-classification/)[1] to categorize scientific papers into the appropriate categories, including Computer Science, Physics, Mathematics, and Statistics, based on their title and abstract.

# Data Wrangling and Feature Engineering
The provided data is divided into train and test sets. The train set has three columns, TITLE, ABSTRACT and label (the category of the scientific paper in numbers from 0 to 3) while the test set only has the first two. Examples can be found in the table below.   
![Screenshot 2023-01-22 at 17-27-50 SciencePaperCategorization Kaggle](https://user-images.githubusercontent.com/30448897/213952845-53dfe18b-8b4c-4529-87a4-c175bfed4894.png)

The table below confirms that there is no null cell in each column, so no row has been removed.

|#  |Column|   Non-Null Count |Dtype| 
|---|------|------------------|-----|
| 0 |TITLE |    15472 non-null|object|
| 1 |ABSTRACT|  15472 non-null|object|
| 2 |label|     15472 non-null|int64|

Before feeding the train data set into the model, 20% of the rows, i.e. 3094 rows, are randomly selected to form the validation set. As one may be aware of that it's common to have non-alphanumeric characters, such as Greek or Latin characters, in title and abstract in scientifc papers. As the first step for training, those special characters are removed from column TITLE and ABSTRACT, as they are usually defined by human and the meaning could vary from paper to paper. The TITLE and ABSTRACT columns are then concatenated into a new column. The new column and the label column are formated into TensorFlow dataset for training. The same process is applied to validation set to form a TensorFlow dataset. For the train and validation datasets to be trainable by machine learning (ML) model, the strings in the dataset need to be converted into vector of numbers. The step is done by processing the dataset using TensorFlow's TextVectorization layer[2], where the text corpus from train data is "adapted" to generate tokens.
To improve the model, numeric characters, single-digit characters, i.e. a to z, and common stop words[3], imported from nltk library, are also removed from the text corpus of concatenated column of TITLE and ABSTRACT. 

# Machine Learning Model
The ML model used in this study is a neural network composed of (1) Embedding layer[4],(2) Bidirectional LSTM layer[5], and (3) 3 Dense layers, where 20% of coefficients connecting to each Dense layer are set to 0 randomly to reduce overfitting, as shown below.   
![Screenshot 2023-01-19 at 20-34-02 SciencePaperCategorization Kaggle](https://user-images.githubusercontent.com/30448897/213840275-f5038209-caa8-4913-960d-810154f0349e.png)   
The embedding layer encodes each word in the training corpus as a vector of floating point number, and the vectors of words with similar meaning are similar [4]. The LSTM layer provides the capability of long-term memory[4] for words in abstract and title. Adding bidirectional layer on top of LSTM further enables the model to associate a keyword with words before and after it. In Dense layers, ReLU (Rectified Linear Unit)[6] is used as activation function except for the output layer, where Softmax[7] is used as activation function instead because this is a multi-class classification problem.
The hyperparameters used in the model, such as the dimension for embedding layer (global variable embedding_dim), the diimension of output space in LSTM layer (lstm_dim), size of traning batch (batch_size), and learning rate of training (learning_rate), are fine-tuned based on the loss of validation data (process not shown in the result). 

# Result and Discussion
The result of training can be found in the table below. It's clear to see that removing stop words and characters not meaningful does help improve the model, especially in loss.   

|   |Original|No Stop Words|Improvement|
|---|--------|----|---|
|Validation Accuracy*|0.8081|0.8111|0.4%|
|Validation Loss*|0.6108|0.5850|-4.2%|

\*: Arithmetic average of values

During the training, there is a noticeable sign where as more epochs is processed in the training, the loss against the validation set begins to increase and the accuracy begins to plateau while the loss and accuracy of the training dataset keeps improving, as shown in the two sets of figures below.   
- Accuracy and Loss in the Original model

|Accuracy|Loss|
|---|---|
|![Screenshot 2023-01-22 at 19-33-53 SciencePaperCategorization Kaggle](https://user-images.githubusercontent.com/30448897/213975092-0c4e8a7e-19ea-4f82-ad94-00504779bb98.png)|![Screenshot 2023-01-22 at 19-33-41 SciencePaperCategorization Kaggle](https://user-images.githubusercontent.com/30448897/213975111-9b05dcc1-2820-4d8e-b6a5-55449db48fc9.png)|

- Accuracy and Loss in the Model without Stop Words

|Accuracy|Loss|
|---|---|
|![Screenshot 2023-01-22 at 20-34-39 SciencePaperCategorization Kaggle](https://user-images.githubusercontent.com/30448897/213974992-017bc6ff-0e2d-4ab5-88d4-a77de80a7b87.png)|![Screenshot 2023-01-22 at 20-34-28 SciencePaperCategorization Kaggle](https://user-images.githubusercontent.com/30448897/213975010-c940bda7-2ccf-4147-b9d0-85c1f1a9bc40.png)|

Overfitting prevents the model from providing accurate prediction when it is applied to a new data set, such as the test set in the problem. The possible cause could be that (1) the dataset for training is too large and limits the training path, and/or (2) the coefficients in the model are too rigid to change from previous round of optimization or initial state. One possible approach to reduce overfitting is to increase the dropout out ratio in Dropout layer[8]. This proposal could add randomness to the model and could potentially reduce overfitting, but it is pending further test.

## Reference
1. Science Topic Classification, Kaggle (https://www.kaggle.com/competitions/science-topic-classification/)   
2. TextVectorization layer, TensorFlow https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization
3. Removing stop words with NLTK in Python, GeeksforGeeks https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
4. Word Embedding, TensorFlow https://www.tensorflow.org/text/guide/word_embeddings
5. Long Short-term Memory,Hochreiter & Schmidhuber, 1997 https://www.bioinf.jku.at/publications/older/2604.pdf
6. A Gentle Introduction to the Rectified Linear Unit (ReLU), Machine Learning Mastery https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
7. Softmax Activation Function with Python, achine Learning Mastery https://machinelearningmastery.com/softmax-activation-function-with-python/
8. Overfitting with text classification using Transformers, Data Science StackExchange https://datascience.stackexchange.com/questions/72857/overfitting-with-text-classification-using-transformers
