## Irony Detection Datasets ##

This repository holds the datasets we use for the project:

* Training datasets 

  * Tweets with binary labels (ironic or non-ironic) **3834 tweets**

    Source: [CodaLab website of the task](https://competitions.codalab.org/competitions/17468)
    1. How data was collected: Searching Twitter for the hashtags #irony, #sarcasm and #not. All tweets were collected between 01/12/2014 and 04/01/2015 and represent 2,676 unique users. The entire corpus was cleaned by removing retweets, duplicates and non-English tweets, and replacement of XML-escaped characters (e.g. &).
    2. Annotation: Three students in linguistics and second-language speakers of English, which each annotated one third of the corpus. Brat (Stenetorp et al., 2012). was used as the annotation tool. To assess the reliability of the annotations, an annotation agreement study was carried out on a subset of the corpus (100 instances).
    3. Train and test corpus: Based on the annotations, 2,396 instances are ironic while 604 are not. To balance class representation in the corpus, 1,792 non-ironic tweets were added from a background corpus. The tweets were manually checked to ascertain that they are non-ironic and are devoid of irony-related hashtags. This brings the total amount of data to 4,792 tweets (2,396 ironic + 2,396 non-ironic).  The corpus is randomly split into a training (80%)and test (20%) set. 

* Test datasets 

  * Tweets with binary labels (ironic or non-ironic) **784 tweets**

    Source: [CodaLab website of the task](https://competitions.codalab.org/competitions/17468)

  * Reddit comment with binary labels (ironic or non-ironic) **1950 comments**

    Source: [Kaggle](https://www.kaggle.com/rtatman/ironic-corpus)




