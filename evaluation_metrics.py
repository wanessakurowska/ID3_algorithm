from decision_tree import DecisionTree
import pandas as pd
import numpy as np


def confusion_matrix(root, test_data, class_values):
    # Funkcja obliczająca macierz pomyłek dla drzewa decyzyjnego
    matrix = pd.DataFrame(0, index=class_values, columns=class_values)

    for _, row in test_data.iterrows():  # Iteracja przez wiersze zbioru testowego
        true_value = row["class"]  # Prawdziwa klasa instancji
        predicted_class = root.predict(row)  # Klasa przewidywana przez drzewo decyzyjne
        if predicted_class != None:  # Jeśli przewidziana klasa nie jest wartością None
            # Zwiększenie licznika w macierzy pomyłek dla kombinacji prawdziwej i przewidywanej klasy
            matrix.at[true_value, predicted_class] += 1

    return matrix


def calculate_metrics(conf_matrix, class_values):
    # True Positive = liczba prawidłowych przewidywań
    TP = {value: conf_matrix.at[value, value] for value in class_values}
    # False Positive = suma przypadków, gdy model przewiduje klasę value - liczba prawidłowych przewidywań (TP)
    FP = {value: conf_matrix[value].sum() - conf_matrix.at[value, value] for value in class_values}
    # False Negative = suma wszystkich rzeczywistych przypadków danej klasy - liczba prawidłowych przewidywań (TP)
    FN = {value: conf_matrix.loc[value].sum() - conf_matrix.at[value, value] for value in class_values}
    # True Negative =  łączna liczba wszystkich przypadków - suma True Positive, False Positive i False Negative
    TN = {value: conf_matrix.sum().sum() - (TP[value] + FP[value] + FN[value]) for value in class_values}

    # Sumowanie wartości dla wszystkich klas
    TP_total = sum(TP.values())
    TN_total = sum(TN.values())
    FP_total = sum(FP.values())
    FN_total = sum(FN.values())

    return TP_total, TN_total, FP_total, FN_total


def accuracy(TP, TN, FP, FN):
    # Dokładność - stosunek liczby poprawnych predykcji do liczby instancji
    if (TP + FP + TN + FN) == 0:
        return 0
    return (TP + TN) / (TP + FP + TN + FN)

def precision(TP, FP):
    # Precyzja - jaka część wyników zaklasyfikowanych jako dodatnie jest faktycznie dodatnia.
    if (TP + FP) == 0:
        return 0
    return TP / (TP + FP)

def recall(TP, FN):
    # Czułość - jaka część dodatnich wyników została wykryta
    if (TP + FN) == 0:
        return 0
    return TP / (TP + FN)

def specificity(TN, FP):
    # Swoistość - jaka część ujemnych wyników została wykryta
    if (TN + FP) == 0:
        return 0
    return TN / (TN + FP)

def get_all_evaluation_metrics(dataset, k, max_depth=None, min_samples_split=2):
    # Metoda wyświetlająca obliczone miary jakości

    data = dataset.sample(frac=1, random_state=42)  # Losowe mieszanie danych
                                                    # (random_state w celu zapewnienia powtarzalności)
    subsets = np.array_split(data, k)  # Podział danych na k równych podzbiorów

    # Listy do przechowywania wyników dla testu tożsamościowego
    identity_test_accuracy = []
    identity_train_accuracy = []  # W celu sprawdzania overfittingu
    identity_test_precision = []
    identity_test_recall = []
    identity_test_specificity = []

    # Listy do przechowywania wyników dla testu binarnego
    binary_test_accuracy = []
    binary_train_accuracy = []  # W celu sprawdzania overfittingu
    binary_test_precision = []
    binary_test_recall = []
    binary_test_specificity = []

    # Listy do przechowywania głębokości i liczby liści
    identity_depths = []
    identity_leaves = []

    binary_depths = []
    binary_leaves = []

    # Inicjalizacja sumatorów dla macierzy pomyłek, w celu jej wyświetlenia
    total_TP_identity, total_TN_identity, total_FP_identity, total_FN_identity = 0, 0, 0, 0
    total_TP_binary, total_TN_binary, total_FP_binary, total_FN_binary = 0, 0, 0, 0

    class_values = dataset["class"].unique()  # Klasy w zbiorze danych

    for i in range(k):
        # Podział na dane treningowe i testowe, zgodnie z walidacją krzyżową
        training_data = pd.concat([subsets[j] for j in range(k) if j != i])
        test_data = subsets[i]

        # Test tożsamościowy
        decision_tree_identity = DecisionTree(test_type="identity", max_depth=max_depth, min_samples_split=min_samples_split)
        decision_tree_identity.train(training_data, data.columns.drop("class"))

        # Obliczanie miar jakości dla testu tożsamościowego
        conf_matrix = confusion_matrix(decision_tree_identity.root, test_data, class_values)
        TP, TN, FP, FN = calculate_metrics(conf_matrix, class_values)
        total_TP_identity += TP
        total_TN_identity += TN
        total_FP_identity += FP
        total_FN_identity += FN

        # Wywoływanie funkcji do obliczania miar jakości
        accuracy_identity = accuracy(TP, TN, FP, FN)
        precision_identity = precision(TP, FP)
        recall_identity = recall(TP, FN)
        specificity_identity = specificity(TN, FP)

        # Dodanie do list wyników
        identity_test_accuracy.append(accuracy_identity)
        identity_test_precision.append(precision_identity)
        identity_test_recall.append(recall_identity)
        identity_test_specificity.append(specificity_identity)

        identity_depths.append(decision_tree_identity.root.depth())
        identity_leaves.append(decision_tree_identity.root.num_leaves())

        # Sprawdzenie overfittingu na danych treningowych
        conf_matrix_train = confusion_matrix(decision_tree_identity.root, training_data, class_values)
        TP_train, TN_train, FP_train, FN_train = calculate_metrics(conf_matrix_train, class_values)
        accuracy_identity_train = accuracy(TP_train, TN_train, FP_train, FN_train)
        identity_train_accuracy.append(accuracy_identity_train)

        # Test binarny
        decision_tree_binary = DecisionTree(test_type="binary", max_depth=max_depth, min_samples_split=min_samples_split)
        decision_tree_binary.train(training_data, data.columns.drop("class"))

        # Obliczanie miar jakości dla testu binarnego
        conf_matrix = confusion_matrix(decision_tree_binary.root, test_data, class_values)
        TP, TN, FP, FN = calculate_metrics(conf_matrix, class_values)
        total_TP_binary += TP
        total_TN_binary += TN
        total_FP_binary += FP
        total_FN_binary += FN

        # Wywoływanie funkcji do obliczania miar jakości
        accuracy_binary = accuracy(TP, TN, FP, FN)
        precision_binary = precision(TP, FP)
        recall_binary = recall(TP, FN)
        specificity_binary = specificity(TN, FP)

        # Dodanie do list wyników
        binary_test_accuracy.append(accuracy_binary)
        binary_test_precision.append(precision_binary)
        binary_test_recall.append(recall_binary)
        binary_test_specificity.append(specificity_binary)

        binary_depths.append(decision_tree_binary.root.depth())
        binary_leaves.append(decision_tree_binary.root.num_leaves())

        # Sprawdzenie overfittingu na danych treningowych
        conf_matrix_train = confusion_matrix(decision_tree_binary.root, training_data, class_values)
        TP_train, TN_train, FP_train, FN_train = calculate_metrics(conf_matrix_train, class_values)
        accuracy_binary_train = accuracy(TP_train, TN_train, FP_train, FN_train)
        binary_train_accuracy.append(accuracy_binary_train)

    # Obliczanie średnich miar jakości dla testu tożsamościowego
    mean_accuracy_identity = np.mean(identity_test_accuracy)
    mean_identity_train_accuracy = np.mean(identity_train_accuracy)
    mean_precision_identity = np.mean(identity_test_precision)
    mean_recall_identity = np.mean(identity_test_recall)
    mean_specificity_identity = np.mean(identity_test_specificity)
    mean_depth_identity = np.mean(identity_depths)
    mean_leaves_identity = np.mean(identity_leaves)

    # Dla macierzy pomyłek
    avg_TP_identity = total_TP_identity / k
    avg_TN_identity = total_TN_identity / k
    avg_FP_identity = total_FP_identity / k
    avg_FN_identity = total_FN_identity / k

    # Obliczanie średnich miar jakości dla testu binarnego
    mean_accuracy_binary = np.mean(binary_test_accuracy)
    mean_binary_train_accuracy = np.mean(binary_train_accuracy)  # dodane do sprawdzania overfittingu
    mean_precision_binary = np.mean(binary_test_precision)
    mean_recall_binary = np.mean(binary_test_recall)
    mean_specificity_binary = np.mean(binary_test_specificity)
    mean_depth_binary = np.mean(binary_depths)
    mean_leaves_binary = np.mean(binary_leaves)

    # Dla macierzy pomyłek
    avg_TP_binary = total_TP_binary / k
    avg_TN_binary = total_TN_binary / k
    avg_FP_binary = total_FP_binary / k
    avg_FN_binary = total_FN_binary / k

    # Wyświetlanie wyników
    print("Miary jakości dla testów tożsamościowych:")
    print(f"Dokładność: {mean_accuracy_identity * 100:.2f}%")
    print(f"Precyzja: {mean_precision_identity * 100:.2f}%")
    print(f"Czułość: {mean_recall_identity * 100:.2f}%")
    print(f"Swoistość: {mean_specificity_identity * 100:.2f}%")
    print(f"Głębokość: {mean_depth_identity}")
    print(f"Liczba liści: {mean_leaves_identity} \n")

    print("Macierz pomyłek dla testów tożsamościowych:")
    print(f"TP: {avg_TP_identity}, FP: {avg_FP_identity}")
    print(f"FN: {avg_FN_identity}, TN: {avg_TN_identity}\n")

    print("Miary jakości dla testów binarnych:")
    print(f"Dokładność: {mean_accuracy_binary * 100:.2f}%")
    print(f"Precyzja: {mean_precision_binary * 100:.2f}%")
    print(f"Czułość: {mean_recall_binary * 100:.2f}%")
    print(f"Swoistość: {mean_specificity_binary * 100:.2f}%")
    print(f"Głębokość: {mean_depth_binary}")
    print(f"Liczba liści: {mean_leaves_binary} \n")

    print("Macierz pomyłek dla testów binarnych:")
    print(f"TP: {avg_TP_binary}, FP: {avg_FP_binary}")
    print(f"FN: {avg_FN_binary}, TN: {avg_TN_binary}\n")

    # Wyświetlanie wyników overfittingu
    print("Sprawdzenie overfittingu")
    print(f"Dokładność na danych treningowych dla testów tożsamościowych: {mean_identity_train_accuracy * 100:.2f}%")
    print(f"Dokładność na danych treningowych dla testów binarnych: {mean_binary_train_accuracy * 100:.2f}%")


def test_small_dataset(dataset, k, length):
    # Test na zmniejszonym zbiorze danych
    small_dataset = dataset.sample(n=length, random_state=42)  # Zrandomizowanie i zmniejszenie długości danych

    # Wyświetlanie miar jakości dla malego zbioru
    get_all_evaluation_metrics(small_dataset, k=k, max_depth=3, min_samples_split=2)
