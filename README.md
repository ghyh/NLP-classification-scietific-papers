# NLP-classification-scietific-papers
# Introduction
The ML model in this repo is trying to study the Kaggle competition "Science Topic Classification" (https://www.kaggle.com/competitions/science-topic-classification/)[1] to categorize scientific papers into the appropriate categories, including Computer Science, Physics, Mathematics, and Statistics, based on their title and abstract.

# Data Wrangling and Feature Engineering
The provided data is divided into train and test sets. The train set has three columns, TITLE, ABSTRACT and label (the category of the scientific paper in numbers from 0 to 3) while the test set only has the first two. The table below confirms that there is no null cell in each column, so no row has been removed.

|#  |Column|   Non-Null Count |Dtype| 
|---|------|------------------|-----|
| 0 |TITLE |    15472 non-null|object|
| 1 |ABSTRACT|  15472 non-null|object|
| 2 |label|     15472 non-null|int64|

Before feeding the train data set into the model, 20% of the rows, i.e. 3094 rows, are randomly selected to form the validation set. As one may be aware of that it's common to have non-alphanumeric characters, such as Greek or Latin characters, in title and abstract in scientifc papers. Those special characters are removed from column TITLE and ABSTRACT, as they are usually defined by human and the meaning could vary from paper to paper, as the first step for training. The TITLE and ABSTRACT columns are then concatenated into a new column. The new column and the label column are formated into TensorFlow dataset for training. The same process is applied to validation set to form a TensorFlow dataset. For the train and validation datasets to be trainable by machine learning (ML) model, the strings in the dataset need to be converted into vector of numbers. The step is done by processing the dataset using TensorFlow's TextVectorization layer[2], where the text corpus from train data is "adapted" to generate tokens.
To improve the model, numeric characters, single-digit characters, i.e. a to z, and common stop words, imported from nltk library, are also removed from the text corpus of concatenated column of TITLE and ABSTRACT. 

# Machine Learning Model
The ML model used in 

# Result and Discussion

## Reference
1. Science Topic Classification, Kaggle (https://www.kaggle.com/competitions/science-topic-classification/)   
2. TextVectorization layer, TensorFlow https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization
