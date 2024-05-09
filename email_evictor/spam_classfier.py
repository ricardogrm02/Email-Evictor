#!/usr/bin/env python3

#############################

# Spam Classification Program
# Contributors: Ricardo Granado Macias, Destiny Bonillas, Paul Kennedy
# Machine Learning Implemented: Sklearn

#############################

# imports
import time
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import svm

def runModel():
    accuracy_score = None
    training_time = None
    programRuntime = None

# collect time it takes for program to run
    begin = time.time()

# csv file 1
# load data
    dataframe = pd.read_csv("spam_ham_dataset.csv")
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)

# prints data, can comment out later
    print(dataframe.head())

# prints info on our data, can comment out later
# print(dataframe.describe())

# add plot to visualize spam vs not
    label_counts = dataframe['label'].value_counts()
    # plt.figure(figsize=(8, 4))
    # colors = ['blue', 'red']
    # label_counts.plot(kind='bar', color=colors)
    # plt.title('Distribution of Spam vs Not Spam')
    # plt.xlabel('Label')
    # plt.ylabel('Frequency')
    # plt.xticks(rotation=0)
    #plt.show()

# split into training and test data
    x = dataframe["text"]
    y = dataframe["label"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

# extract features
    count_vec = CountVectorizer()
    features = count_vec.fit_transform(x_train)

# build model
    model = svm.SVC()

# model training start time
    train_begin = time.time()

    model.fit(features, y_train)

# model training end time
    train_end = time.time()
    print("Model training time: {:.2f} seconds".format(train_end-train_begin))
    training_time = str("Model training time: {:.2f} seconds".format(train_end-train_begin))

# test accuracy
    features_test = count_vec.transform(x_test)

# print accuracy
    print("Accuracy of Model: ", model.score(features_test, y_test))
    accuracy_score = "Accuracy of Model: " + str(model.score(features_test, y_test))

# determine time @ end of program
    end = time.time()
    print("Overall Program Runtime: {:.2f} seconds".format(end-begin))
    


    return accuracy_score, training_time

def emailPrediction(email_content):
    accuracy_score = None
    training_time = None
    programRuntime = None

# collect time it takes for program to run
    begin = time.time()

# csv file 1
# load data
    dataframe = pd.read_csv("spam_ham_dataset.csv")
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)

# prints data, can comment out later
    print(dataframe.head())

# prints info on our data, can comment out later
# print(dataframe.describe())

# add plot to visualize spam vs not
    label_counts = dataframe['label'].value_counts()
    # plt.figure(figsize=(8, 4))
    # colors = ['blue', 'red']
    # label_counts.plot(kind='bar', color=colors)
    # plt.title('Distribution of Spam vs Not Spam')
    # plt.xlabel('Label')
    # plt.ylabel('Frequency')
    # plt.xticks(rotation=0)
    #plt.show()

# split into training and test data
    x = dataframe["text"]
    y = dataframe["label"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

# extract features
    count_vec = CountVectorizer()
    features = count_vec.fit_transform(x_train)

# build model
    model = svm.SVC()

# model training start time
    train_begin = time.time()

    model.fit(features, y_train)

# model training end time
    train_end = time.time()
    print("Model training time: {:.2f} seconds".format(train_end-train_begin))
    training_time = str("Model training time: {:.2f} seconds".format(train_end-train_begin))


    # Transform the preprocessed email content into features
    email_features = count_vec.transform([email_content])

    # Predict whether the email is spam or not
    prediction = model.predict(email_features)

    prediction_result = ""

    # Print prediction
    print (prediction)
    if prediction == ['spam']:
        prediction_result = "EMAIL IS VERY LIKELY TO BE SPAM"
    else:
        prediction_result = "EMAIL IS VERY LIKELY TO NOT BE SPAM"

    # Calculate accuracy score
    features_test = count_vec.transform(x_test)
    accuracy_score = model.score(features_test, y_test)

    # Print accuracy score
    print("Accuracy of Model: ", accuracy_score)

    # Determine program runtime
    end = time.time()
    program_runtime = end - begin
    print("Overall Program Runtime: {:.2f} seconds".format(program_runtime))

    return accuracy_score, training_time, prediction_result

    


def main():
    output = runModel()
    # emailPrediction()


if __name__ == "__main__":
    main()

###########End of Program############