from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from datasets import load_dataset
from data.helpers import augment_dataset
import numpy as np

# Load dataset
datasets = load_dataset('coastalcph/populism-trump-2016')
train_dataset = datasets['train']
train_dataset = augment_dataset(train_dataset, upsample_ratio=5)
train_dataset = train_dataset.shuffle()
test_dataset = datasets['test']

# TRAIN
train_documents = train_dataset['sentence']
train_labels = train_dataset['pop_code']
y_train = []
for doc_label in train_labels:
    if doc_label != 3:
        y_train.append([doc_label])
    else:
        y_train.append([1, 2])

# TEST
test_documents = test_dataset['sentence']
test_labels = test_dataset['pop_code']
y_test = []
for doc_label in test_labels:
    if doc_label != 3:
        y_test.append([doc_label])
    else:
        y_test.append([1, 2])

# Binarize the labels for multi-label classification
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train)
y_test = mlb.transform(y_test)

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 3), use_idf=True, lowercase=True, min_df=20, max_df=0.5, max_features=10000)
vectorizer.fit(train_documents)

# Fit the vectorizer to the training documents and transform both the training and test documents
X_train = vectorizer.fit_transform(train_documents)
X_test = vectorizer.transform(test_documents)

# Train a multi-label SVM classifier using the training data
classifier = MultiOutputClassifier(svm.SVC(kernel='linear'))  # You can change the kernel type as needed
classifier.fit(X_train, y_train)

# Predict the labels of the test documents
y_pred = classifier.predict(X_test)

# Predict the labels of the test documents
y_pred = classifier.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred,
                                                        digits=3, labels=range(3),
                                                        target_names=['None', 'Elite', 'People']))

# Get feature names from TF-IDF
feature_names = np.array(vectorizer.get_feature_names_out())


# For each label, get the top n-grams
def get_top_ngrams_for_label(classifier_idx, top_n=10):
    # classifier_idx corresponds to the position of your label in y_train
    svm_classifier = classifier.estimators_[classifier_idx]
    coef = svm_classifier.coef_.toarray()[0]  # For SVC with linear kernel

    # Get top positive and negative n-grams
    top_pos = coef.argsort()[-top_n:][::-1]
    top_neg = coef.argsort()[:top_n]

    print(f"Top indicative n-grams for class {classifier_idx}:")
    print("Positive:")
    for i in top_pos:
        print(f"{feature_names[i]}: {coef[i]}")

    print("\nNegative:")
    for i in top_neg:
        print(f"{feature_names[i]}: {coef[i]}")
    print("\n" + "=" * 50 + "\n")


for label_idx in range(len(classifier.estimators_)):
    get_top_ngrams_for_label(label_idx, top_n=10)