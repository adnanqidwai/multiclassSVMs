import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

test_file = sys.argv[1]

df=pd.read_csv("penguins_train.csv")
X_generate=pd.read_csv(test_file)

# PREPROCESSING
X_train=df.drop('Species',axis=1)
y_train=df['Species']

## imputing missing values
imputer=SimpleImputer(strategy='mean')
cols= ['Culmen Length (mm)','Culmen Depth (mm)','Flipper Length (mm)','Body Mass (g)','Delta 15 N (o/oo)','Delta 13 C (o/oo)']
X_train[cols]=imputer.fit_transform(X_train[cols])
X_generate[cols]=imputer.fit_transform(X_generate[cols])
imputer= SimpleImputer(strategy='most_frequent')
cols=['Sex']
X_train[cols]=imputer.fit_transform(X_train[cols])
X_generate[cols]=imputer.fit_transform(X_generate[cols])

# one hot encoding
X_train=pd.get_dummies(X_train,drop_first=True)
X_generate = pd.get_dummies(X_generate,drop_first=True)

# drop redundant column
X_train.drop(['Sex_FEMALE'],axis=1,inplace=True)

# scaling
scaler=StandardScaler()
columns=X_train.columns[0:6]
X_train[columns]=scaler.fit_transform(X_train[columns])
X_generate[columns]=scaler.fit_transform(X_generate[columns])

# splitting into test and train
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

def classifiers(X_train, y_train):

    unique_classes = np.unique(y_train)
    binary_classifiers = {}

    for i in range(len(unique_classes)):

        for j in range(i + 1, len(unique_classes)):
            class_1, class_2 = unique_classes[i], unique_classes[j]
            X_train_pair = X_train[(y_train == class_1) | (y_train == class_2)]
            y_train_pair = y_train[(y_train == class_1) | (y_train == class_2)]

            binary_y_train = np.where(y_train_pair == class_1, 1, -1)

            svm_classifier = SVC(kernel='linear', C=1)
            svm_classifier.fit(X_train_pair, binary_y_train)
            binary_classifiers[(class_1, class_2)] = svm_classifier
    return binary_classifiers

def prediction(X_generate, binary_classifiers):

    multi_class_predictions = []
    for i in range(X_generate.shape[0]):
        class_scores = {}
        for (class_1, class_2), classifier in binary_classifiers.items():
            decision_function_score = classifier.decision_function(X_generate[i:i+1])
            if decision_function_score > 0:
                predicted_class = class_1
            else:
                predicted_class = class_2

            class_scores[predicted_class] = class_scores.get(predicted_class, 0) + 1

        # Choose the class with the highest count as the predicted class
        predicted_class = max(class_scores, key=class_scores.get)
        multi_class_predictions.append(predicted_class)
    
    return multi_class_predictions    

multi_class_predictions=prediction(X_generate,classifiers(X_train,y_train))

def accuracy(X_test, y_test, binary_classifiers):
    y_pred = prediction(X_test, binary_classifiers)
    return np.mean(y_pred == y_test)

print(f'accuracy: {accuracy(X_test, y_test, classifiers(X_train,y_train))*100}%')

# get confusion matrix
confusion =np.array(confusion_matrix(y_test, prediction(X_test, classifiers(X_train,y_train))))

# get precision recall and f1 score
print(f'classification report:\n{classification_report(y_test, prediction(X_test, classifiers(X_train,y_train)))}')

with open('ovo.csv', 'w') as f:
    f.write('predicted\n')
    for i in multi_class_predictions:
        f.write(i + '\n')


class_names = ['Adelie Penguin', 'Chinstrap penguin', 'Gentoo penguin']

plt.figure(figsize=(8, 6))
plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix for OvO SVM')
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

for i in range(confusion.shape[0]):
    for j in range(confusion.shape[1]):
        plt.text(j, i, f'{confusion[i, j]}', horizontalalignment="center", color="white" if confusion[i, j] > confusion.max() / 2 else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.tight_layout()
plt.show()