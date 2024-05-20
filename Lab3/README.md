# Lab 3
**Author :** ***Ahmed Samady***\
**Supervised by :** ***Pr. Lotfi El Aachak***\
**Course :** ***NLP***\
This directory will contain everything that concerns Lab 3 of the NLP course.
You can find the notebook [here](https://github.com/Samashi47/NLP_Labs/blob/main/Lab3/lab3.ipynb).
## Part 1: Language Modeling - Regression
This part of the lab is focused on a regression task which is short answer grading, the [dataset](https://github.com/dbbrandt/short_answer_granding_capstone_project/blob/master/data/sag/answers.csv) contains answers for [these questions](https://github.com/dbbrandt/short_answer_granding_capstone_project/blob/master/data/sag/questions.csv). We'll be working solely on one question, which is question 11.1 (60 answers) because creating a model that performs regression on answers for different questions wouldn't bring any good results due to different vocabulary used in each question, which means would create confusion for the model.
### NLP Pre-processing Pipeline
Like the other labs we followed a similar pre-processing pipeline, which includes the following:
- Removing punctuations and lowercasing text.
- Tokenization and removing stopwords.
- Stemming and lemmatization
### Word Embeddings
For all the word embeddings we used the lemmatized tokens to generate the vectors.
- **BagOfWords**: We used CountVectorizer module from sci-kit-learn library to generate word embeddings for BagOfWords.
- **TF-IDF**: We used the TfidfVectorizer module from sci-kit-learn library to generate the word embeddings.
- **Word2Vec**: We used the gensim library to train a Word2Vec model, with a window size of 5 and vectors sizes should be 30, each.
### Modeling
#### SVR
To train the SVR model we used the following hyper-parameters:
```python
svr = SVR(kernel='rbf', C=10000, gamma=0.1, tol=0.1)
```
The best kernel after multiple trials is discovered to be the rbf kernel. We obtained the following MSE and RMSE scores:
```txt
SVR MSE Score: 0.3627122499809037
SVR RMSE Score: 0.6022559671608939
```
The MSE and RMSE are reasonable which means thta the SVR correctly predicts the score of the answer with a small error margin.
#### Linear Regression
To train the Linear Regression model we only set the fit_intercept hyper-parameter to True:
```python
lr = LinearRegression(fit_intercept=True)
```
The Linear Regression model doesn't have much parameters to modify, that's why we only modifed this latter. We obtained the following MSE and RMSE scores:
```txt
Linear Regression MSE Score: 0.5040098460750949
Linear Regression RMSE Score: 0.7099365084816353
```
The MSE and RMSE are slightly above those of the SVR model, but they remain reasonable enough for this regression task.
#### Decision Tree Regressor
To train the Decision Tree Regressor model we used a max_depth of 5 and a random_state of 0:
```python
dtr = DecisionTreeRegressor(max_depth=5,random_state=0)
```
We obtained the following MSE and RMSE scores for this model:
```txt
Decision Tree Regressor MSE Score: 0.9120370370370369
Decision Tree Regressor RMSE Score: 0.955006302092838
```
This model is the weakest of the bunch as its RMSE score is 0.95, and an MSE of 0.91.
### Evaluation
The can clearly see that the SVR model is the best regressor in our case, this can be due to the non-linear nature of data, which may explain the low MSE and RMSE. In regards to the Linear Regression model, it seems to be doing okay, in contrast with SVR with an MSE of 0.5 and RMSE of 0.71. However, for Decision Tree Regressor is determined to be the weakest regressorwith an MSE of 0.91 and an RMSE of 0.95, which means that predictions are approximately 0.95 units away from the actual values, which is not bad if the range of the predictions is big, but ours is [0-5] which makes it the worst of the three models.
## Part 2: Language Modeling - Classification
In this section of the lab we'll be focusing on classification, the [dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) we'll be working on contains 74,682 tweets and a validation dataset of 1759 tweets.
### NLP Pre-processing pipeline
The pre-processing done on this dataset is quite heavy due to the nature of the data being pulled out of social media text, which include a lot of punctuations, symbols, emojis, HTML special entities, etc....
The following preprocessing was done on wach tweet:
- Remove user handles tagged in the tweet.
- Remove words that start with th dollar sign.
- Remove hyperlinks.
- Remove hashtags.
- Remove all kinds of punctuations and special characters.
- Remove words with 2 or fewer letters.
- Remove HTML special entities (e.g. &amp;).
- Remove whitespace (including new line characters).
- Remove stopwords.
- Remove single space remaining at the front of the tweet.
- Remove characters beyond Basic Multilingual Plane (BMP) of Unicode and emojis.
- Lowercase
- Remove extra spaces
After this we also dropped duplicates and lines where tweets are missing, and finally label encoded the sentiment column, the following labels are the new labels:
```txt
0 - Irrelevant
1 - Negative
2 - Neutral
3 - Positive
```
Then we pursued the same steps for tokenization, lemmatization and stemming as we did in the previous section
### Word Embeddings
The same process was used in this section as we did in the prevois except Word2Vec was modified accordingly to the new dataset, due to the vocab size getting bigger, and by consquence the vectors should be of higher dimension to capture more information and context.
```python
w2v_model = Word2Vec(tweets_train['lemmatized_tokens'], vector_size=100, window=50, min_count=1, workers=4, epochs=100)
```
### Modeling
Model training in this sections is quiet heavy due to the nature of the dataset, some models take 20min for each trial, th's why finetuning the models is a time consuming and difficult task.
#### SVC
The SVC model is one of the best classifiers, the model's hyper-parameters are left to default, because change it it makes the fiiting take way too much time, and we already have satistfactory results with the default parameters.
The model accuracy is 87% and the classification report is as followed:
```txt
              precision    recall  f1-score   support

           0       0.87      0.84      0.86       172
           1       0.83      0.94      0.88       264
           2       0.93      0.77      0.84       276
           3       0.86      0.91      0.88       275

    accuracy                           0.87       987
   macro avg       0.87      0.86      0.86       987
weighted avg       0.87      0.87      0.87       987
```
#### Logistic Regression
The Logistic Regression model doesn't give any promising results in comparison with SVC, a combination of parameters was tried through experimentation, the best ones are:
```python
logreg = LogisticRegression(penalty='l2',solver='newton-cholesky',multi_class='ovr',tol=1e-6)
```
Even with a lot of hyper-parameter tuning, the hisghest accuracy achieved was 51%, and the classification report is as followed:
```txt
              precision    recall  f1-score   support

           0       0.39      0.16      0.23       172
           1       0.51      0.73      0.60       264
           2       0.52      0.39      0.45       276
           3       0.54      0.66      0.59       275

    accuracy                           0.51       987
   macro avg       0.49      0.48      0.47       987
weighted avg       0.50      0.51      0.49       987
```
#### Naive Bayes
For Naive Bayes, we used the Gaussian Naive Bayes variant, because the other variants can not have vectore with negative values, otherwise we'd have to use a scaler such as MinMaxScaler, to scale the vector's elements to a specific range which can lead to information loss. No hyper-parameters were modified for this model.
This model is no better compared to the previous one, with an accuracy of 47% and the classification report is as followed:
```txt
              precision    recall  f1-score   support

           0       0.32      0.31      0.31       172
           1       0.52      0.61      0.56       264 
           2       0.46      0.38      0.42       276
           3       0.51      0.53      0.52       275

    accuracy                           0.47       987
   macro avg       0.45      0.46      0.45       987
weighted avg       0.47      0.47      0.47       987

```
#### AdaBoost
Since AdaBoost is computationally expensive, the parameters were left untouched.
Even with adaboost, the accuracy of the model and other metrics showcase that the model couldn't generalize the problem. With an accuracy of merely 48% and the classification report is as followed:
```txt
              precision    recall  f1-score   support

           0       0.31      0.20      0.25       172
           1       0.49      0.68      0.57       264
           2       0.49      0.33      0.39       276
           3       0.50      0.59      0.54       275

    accuracy                           0.48       987
   macro avg       0.45      0.45      0.44       987
weighted avg       0.46      0.48      0.46       987
```
### Evaluation
Since the dataset contains a lot of rows, the task became computationally heavy and time consuming, which hindered our ability to tune the hyper-parameters accordingly. However, it seems that SVC remains the top model for sentiment analysis.
with an accuracy of 87%. The other models seemed to underfit. which may be due to the vector's nature, it may be that if we used another method for word embeddings, we would have achieved better results. In this regard, we conclude that embedding have a huge role in the obtained results, which means the quality of the embedding may greatly affect the accuracy of the model.