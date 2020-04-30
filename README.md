![](https://i.imgur.com/2rQvXIB.png)

## 1. Introduction

### SMS and spam

Even in these times, when technology is advancing by leaps and bounds, SMS is still a widely-used communication platform. Only the recipient's phone number is needed, and we can send a message that arrives almost instantly, without using data.

*Spam* is defined as unsolicited usually commercial messages (such as e-mails, text messages, or internet postings) sent to a large number of recipients or posted in a large number of places.

**SMS spamming** is an old marketing practice and is still a problem causing headaches and annoyances for consumers everywhere. Spam messages coming from SMS and messaging apps are becoming more widespread. According to the latest industry data, over half of text message users globally now receive at least one weekly spam message. Alarmingly, more than a quarter get spammed every day with unsolicited messages.

Spam can fill your inbox with no requested messages and sometimes is not only ads, sometimes are even a *security threat*.

### Problem description and data available

Keep the inbox without unwanted messages, avoiding unsolicited ads or messages that include threats that can either infect your phone or steal your private data.

An application capable of filtering incoming SMS, discarding those that are spam, would avoid more than one trouble to the users. Using machine learning algorithms and techniques like **_NLP_**, a model can be developed for this purpose. 

A critical aspect is that non-spam SMS reach the inbox always, because a non-spam SMS (ham) tagged as spam, will create suspicion in the process, unlike the occasional SMS spam that eventually reaches the inbox.

Using a [dataset](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/) with 5000+ SMS gathered for mobile phone spam research, we will analyze the data and develop a machine learning model to classify the SMS. The dataset is a CSV file, and it has 5 columns; the first with the flag **_spam/ham_**, which will be the target feature. And the remaining 4 with the text of the SMS.

As explained before, tagging a non-spam SMS as spam, is a big mistake; but also tagging too much spam SMS as a legitimate message, can't be good either. That's why we will measure the effectiveness of the model with the **_Precision, Recall, and F1 scores_**. 

## 2. Data quality check

The dataset has 5571 records (SMS) with no missing or NaN values in the flag or the corpus of the SMS (at least all rows have data on the first column). We found 403 duplicate rows, which we deleted.

To simplify the analysis and modeling process, we've merged the text of the 4 columns with the text of the SMS. Finally, we change the flag **_spam/ham_** flag to **_0/1_** values. 

![](https://i.imgur.com/aDZXe48.png)

## 3. Descriptive statistics

### Target feature distribution

We are dealing with an *imbalanced classification problem*, having ***87.36%*** of the records are labeled as **ham** (*0*), implying that only around 1 of each 7 or 8 messages is tagged as **spam** (*1*).

![](https://i.imgur.com/LS0ppqX.png)

### Basic text features creation

To start the analysis of the SMS content 5 statistical features were created:

* **_smsLen_**: Length of the SMS.
* **_smsWords_**: Number of words in the SMS.
* **_smsUpper_**: Number of words in capital letters in the SMS.
* **_smsSpecChar_**: Number of special characters in the SMS.
* **_smsWordLen_**: Average length of the words of the SMS.

![](https://i.imgur.com/LIlyLxD.png)

### Features statistics

When we calculate the basic statistics of the newly created features, we see that there are several outliers on all of the features, for example, the **_smsLen_** feature has a mean of *79.5* characters long, and the max value has **910**! Here in the feature box plot, you can see the outlier distribution:

![](https://i.imgur.com/78Z97W4.png)

All features have upper range outliers (the other box plots can be seen in the notebook), only the **_smsUpper_** has more than 2% of the records tagged as outliers, even expanding the whisker factor from 1.5 to 2.5 IQR:

![](https://i.imgur.com/xzb6hCl.png)

Even though they are actual records, they influence the group statistics and the analysis we can make of those results. So, for the EDA we'll remove those rows to get a more accurate insight of the data.

## 4. Exploratory data analysis.

### Features statistics per target label

Once we've removed the outliers when we calculate the mean and the STD per label, we see important differences in the data. For example, a non-spam SMS is **_66_** characters long on average, while the mean for spam SMS is more than *twice*, reaching a length mean of **_136_**. 

In the KDE plot, we can see graphically how the density differs. In the notebook, you can see the plot of the other features. 

![](https://i.imgur.com/IRwwQb7.png)

### Frequent words distribution per target label

First, we normalize the SMS corpus applying the following actions:

* Update to lowercase all words.
* Remove all punctuation characters.
* Remove [stopwords](https://en.wikipedia.org/wiki/Stop_words).
* Word [lemmatisation](https://en.wikipedia.org/wiki/Lemmatisation).

Then, we create a BoW ([bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model)) matrices per target label to see which words are the most frequent in each case. In the SMS labeled as spam, we see words we'd expect in commercial or phishing messages like *free*, *reply*, *prize*, and *win*. The most frequent word chart for non-spam SMS can be seen in the notebook.

![](https://i.imgur.com/cPnBwZR.png)

### Correlated features analysis

In general, the correlation between the spam label and the features is weak, being **0.43** the higher coefficient (*smsLen*). Among the other features is relevant and logic the high correlation between *smsWords* and *smsLen*, as more words present in the text, longer the SMS will be. 

![](https://i.imgur.com/LeJMIvM.png)

## 5. Model development

### Train/Test split and baseline

The data is split into a 7:3 train/test ratio. Then, only the basic features created previously for the EDA phase, we create a basic Logistic Regression model and check its performance with a cross-validation score process, just to know where we stand before developing more tunned models. The results were the following:

Metric | Score 
--- | --- 
Accuracy | 0.8927
Precision | 0.6308
Recall | 0.3631
F1 | 0.4608

Accuracy will always be high in an unbalanced dataset, but the results of **_Precision_**, **_Recall_**, and consequently, of **_F1_** scores reflect that the model doesn't handle the minority class (spam) properly. Intending, we need to add more features to feed the model. 

### NLP features creation, model selection and hyperparameter tunning

First, we split the original dataset (only SMS corpus and spam flag) into the same proportions we did before, and create a pipeline with the following steps:
1. Creation of the matrix of features (basic and NLP) with FeatureUnion, the internal steps are:

  * Creation of basic features.
  * Creation of the CountVectorizer matrix, preprocessing the text (lowercase, stopwords, etc.) and limiting the number of features to 2500.
  * Creation of the TF-IDF matrix, preprocessing the text (lowercase, stopwords, etc.) and limiting the number of features to 2500.
  
2. Standardize the data with the StandardScaler method.
3. Tune the model hyperparameters with the GridSearchCV method.

The machine learning algorithms selected are *Multinomial Naïve Bayes*, *Logistic Regression*, and *Linear SVM*. The results of the GridSearchCV process were:

Classifier | Accuracy | Precision | Recall | F1 score | Best parameters 
--- | --- | --- | --- | --- | --- 
Multinomial NB | 0.9547 | 0.7853 | 0.8906 | 0.8323 | alpha: 0.003
**_Logistic Regression_** | **_0.9735_** | **_0.9767_** | **_0.8096_** | **_0.8846_** | **_C : 0.1_**
Linear SVM | 0.9682 | 0.9279 | 0.8118 | 0.8655 | C : 0.05 - gamma : 0.01

**_Logistic Regression_** got the best metrics, with accuracy and precision above 0.97, which means almost all non-spam SMS were tagged correctly., and an F1 score of 0.88, also the highest among the other models.

### Model validation

With the best hyperparameters found for each model, we fitted the models with the train data, and predict with the test data to validate the effectiveness of the models. In the confusion matrices, we can see the results:

![](https://i.imgur.com/tlpTM9V.png)
![](https://i.imgur.com/xbfiQau.png)

The results match with those obtained in the Cross-validation process.**_Logistic Regression_** is again the best model, obtaining a bit higher accuracy score, although the Precision score falls, the F1 score rise, which we interpret the model generalizes better.

We see that the correct classification of the non-spam SMS is a **_99.26_**%! Which is what we wanted to achieve in the first place. Also, only 15.82% of spam SMS was classified erroneously as non-spam. The top-15 most important features of the model are:

![](https://i.imgur.com/AO71cOX.png)

Oddly enough, the most important feature of the model is a basic statistic calculated over the SMS text (*smsWordLen*), and that 3 out of 5 basic statistics features are in the top-15, proving that sometimes the simpler, the better. Note too, that 11 features are TF-IDF weights and only 1 of the Count Vectorizer bag-of-words.

## 6. Conclusion

We have shown that by using NLP techniques and calculating basic text statistics, we can develop a classifier to filter incoming SMS, avoiding ad or threatening messages in the inbox. Of course, the model performance can be improved, especially labeling the spam SMS.

Perhaps using a more advanced NLP technique like [word embedding](https://en.wikipedia.org/wiki/Word_embedding) and developing a deep learning algorithm, the results would improve considerably, bearing in mind the increase in computational cost when training the model, and its complexity when explaining the results to a non-technical audience.
