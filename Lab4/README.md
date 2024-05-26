# Lab 4
**Author :** ***Ahmed Samady***\
**Supervised by :** ***Pr. Lotfi El Aachak***\
**Course :** ***NLP***\
This directory will contain everything that concerns Lab 4 of the NLP course.
You can find the notebook [here](https://github.com/Samashi47/NLP_Labs/blob/main/Lab4/lab4.ipynb).
## Part 1: Classification Regression
### Scraping
We used for scraping the BeautifulSoup library for retrieving the articles from [Wikipedia](https://ar.wikipedia.org/). The topic chosen is `الحرب العالمية الاولى`, we took 500 articles on the subject, some of which have no relation to said topic. The link that was scraped first is [Search results](https://ar.wikipedia.org/w/index.php?limit=500&offset=0&profile=default&search=%D8%A7%D9%84%D8%AD%D8%B1%D8%A8+%D8%A7%D9%84%D8%B9%D8%A7%D9%84%D9%85%D9%8A%D8%A9+%D8%A7%D9%84%D8%A3%D9%88%D9%84%D9%89&title=%D8%AE%D8%A7%D8%B5:%D8%A8%D8%AD%D8%AB&ns0=1) for the topic found using the search bar of Wikipedia.\
The scoring was done manually, to give accurate relevancy scores to each article, this method ensures that the scores are actually reasonable. However, it is quite a tedious task to score these articles, other methods can be used such as word embedding and cosine similarity to automate this task.
### NLP Pipeline
#### Cleaning
During the cleaning process, we removed the Arabic stop words, punctuation, and numbers. We also removed the Arabic diacritics. and then stored the data into a JSON file.
#### Stemming
For the stemming part, we used the `ArabicLightStemmer` from the `tashaphyne` library. We stored the stemmed data into a JSON file.
#### Lemmatization
For the lemmatization part, we used the `lemmatizer` from the `qalsadi` library. We stored the lemmatized data into a JSON file.
### Embeddings
For the embeddings part, we used the `gensim` library to load a Arabic pretrained GloVe model. We then used the embeddings to get the average of the embeddings of each word in the article. We stored the embeddings in a CSV file.
### Language Modeling
For the language mpdeling part we used the `pytorch` library to test out multiple models, such as RNN, Bidirectional RNN, GRU, and LSTM. We used the embeddings to train the model.
The architecture of all models is:
- Input layer of 256 neurones.
- 1 hidden layer of 256 neurones.
- 100 epochs.
- A learning rate of 0.01.
### Evaluation
The RMSE for all models is basically the same, coming down to 2.73 and a loss is 7.49, this shows that the models are overfitting, which is expected since the dataset is quite small. and the embeddings might not accurately represent the articles. We can try to use a larger dataset and a more accurate embedding model to get better results. Unfortunately we couldn't continue working on the embedding and finetuning the model due to time and computational constraints.
## Part 2: Transformer (Text generation)
### Data
The data used for this part is a dataset found in kaggle, The dataset is a dataset for fake news, we only used the column text for this part, and we used the first 1000 rows of the dataset.
### Model
For the model, we used the `transformers` library from huggingface, we used the `GPT2` model to generate text. We used the `GPT2LMHeadModel` model to generate text. We used the `GPT2Tokenizer` to tokenize the text. the model is `gpt2-medium`.
### Fine Tuning
We fine-tuned the model on the dataset, for a 100 epochs, a batch size of 16 and a learning rate of $3*10^{-5}$. We used the `AdamW` optimizer. after fine-tuning the model we saved it to a file. the sum loss is 2992.61, due to computational constraints we couldn't fine-tune the model for more epochs.
### Evaluation
We used the final model to generate text, and the results were reasonable, the model was able to generate text that is coherent and somewhat relevant to the topic. The model can be improved by fine-tuning it for more epochs and using a larger dataset. However, sometimes the text genreated was only pointing out to the same thing over and over again, such as telling to us to go to the article detailes in the followig link, this can be due to the small dataset and the lack of diversity in the data.
**Example of generated text:**
```text
news:Just got off the phone with Rep. John Lewis (D-GA) who was brutally beaten by a white supremacist in Charlottesville, VA. Rep. Lewis is a civil rights icon, and I am heartbroken that he has to endure this. He is a true American hero.  John Lewis (@repjohnlewis) August 14, 2017Featured image via Chip Somodevilla / Getty images<|endoftext|> 

news:You can see the full video below:Featured image via screengrab<|endoftext|> 
```
## Part 3: BERT (Text classification)
### Data
The data used for this part is Amazon reviwes for the `Software` category, we used the first 4000 rows of the dataset. The dataset contains the columns `reviewText` and `overall`.
### Model
We used the `transformers` library from huggingface, we used the `BertForSequenceClassification` model to classify the text. We used the `BertTokenizer` to tokenize the text. the model is `bert-base-uncased`.
### Fine Tuning
The model was fine-tuned on the dataset, for 100 epochs, a batch size of 32 and a learning rate of $5*10^{-5}$. We used the `AdamW` optimizer, the max_len is set to 512 which is the max length for tokens for BERT. The architecture used is as follows:
- A fully connected input layer of 768 neurones.
- A ReLU activation hidden layer.
- Another fully connected hidden layer of 5 neurones that corresponds to the 5 classes.
### Evaluation
The model was able to classify the text with an accuracy of 0.77, an F1 score of 0.82 for class 0, 0.50 for class 1, 0.65 for class 2, 0.61 for class 3, and 0.87 for class 4. The model can be improved by fine-tuning it for more epochs and using a larger dataset. The average train loss is 0.002, and the validation loss is 1.82, the validation accuracy is 0.8078. The model can be improved by fine-tuning it for more epochs and using a larger dataset. However, due to computationa constraints we couldn't fine-tune the model for more epochs.
the full classification report is as follows:
```text
Classification Report for BERT :
               precision    recall  f1-score   support

           1       0.89      0.76      0.82        96
           2       0.51      0.49      0.50        43
           3       0.63      0.66      0.65        86
           4       0.53      0.72      0.61       139
           5       0.91      0.83      0.87       436

    accuracy                           0.77       800
   macro avg       0.70      0.69      0.69       800
weighted avg       0.79      0.77      0.77       800
```