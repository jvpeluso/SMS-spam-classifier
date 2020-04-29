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

![](https://i.imgur.com/mHm47JA.png)

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
