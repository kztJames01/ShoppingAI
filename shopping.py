import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):

    evidence = []
    labels = []

    month_index = {
        "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4,
        "June": 5, "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11,
    }

    with open(filename, newline='')as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            inner_evidencelist = [
                int(row["Administrative"]),
                float(row["Administrative_Duration"]),
                int(row["Informational"]),
                float(row["Informational_Duration"]),
                int(row["ProductRelated"]),
                float(row["ProductRelated_Duration"]),
                float(row["BounceRates"]),
                float(row["ExitRates"]),
                float(row["PageValues"]),
                float(row["SpecialDay"]),
                month_index[row["Month"]],
                int(row["OperatingSystems"]),
                int(row["Browser"]),
                int(row["Region"]),
                int(row["TrafficType"]),
                1 if row["VisitorType"] == "Returning_Visitor" else 0,
                1 if row["Weekend"] == "TRUE" else 0,
            ]
            evidence.append(inner_evidencelist)


            label = 1 if row["Revenue"] == "TRUE" else 0
            labels.append(label)

    return evidence, labels

def train_model(evidence, labels):

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    total_positives = 0 # truePositives + falseNegatives
    total_negatives = 0 # trueNegatives + falsePositives

    for actual, predicted in zip(labels, predictions):
        if actual == 1:
            total_positives += 1
            if predicted == 1:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            total_negatives += 1
            if predicted == 0:
                true_negatives += 1
            else:
                false_positives += 1

    sensitivity = true_positives / total_positives
    specificity = true_negatives / total_negatives

    return sensitivity, specificity


if __name__ == "__main__":
    main()

