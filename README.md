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

We will develop a model as accurate as possible, using a dataset with 5000+ SMS gathered for mobile phone spam research. The dataset can be found [here](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/). 

## 2. Data quality check

