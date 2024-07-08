import math
from node import Node

class DecisionTree:
    def __init__(self, test_type="identity", max_depth=None, min_samples_split=2):
        self.test_type = test_type  # Typ testu: "identity" (tożsamościowy) lub "binary" (binarny)
        self.max_depth = max_depth  # Maksymalna głębokość drzewa w celu uniknięcia overfittingu
        self.min_samples_split = min_samples_split  # Minimalna liczba próbek do podziału w celu uniknięcia overfittingu
        self.root = None

    def train(self, training_data, attributes):
        # Metoda do trenowania drzewa decyzyjnego
        self.root = self.id3_algorithm(training_data, attributes, self.test_type, depth=0)

    def predict(self, instance):
        # Metoda do przewidywania wartości w danych testowych
        return self.root.predict(instance)

    def entropy(self, training_data, attribute):
        # Obliczanie entropii podanego atrybutu w zbiorze danych
        num_of_values = len(training_data) # Liczba elementów w zbiorze
        values = training_data[attribute].unique() # Wartości atryubtu
        entropy = 0
        for val in values:
            attribute_instances = len(training_data[training_data[attribute] == val])  # Liczba instancji atrybutu z określoną wartością
            p = attribute_instances / num_of_values  # p - stosunek ilości instancji do ilości elementów w zbiorze
            entropy -= p * math.log2(p)  # Wzór na entropię: E(S) = -Σ p(x) * log2(p(x)), gdzie suma jest po wszystkich wartościach atrybutu
        return entropy

    def information_gain(self, training_data, attribute):
        # Obliczanie przyrostu informacji dla podanego atrybutu
        total_entropy = self.entropy(training_data, "class")  # Obliczanie entropii dla całego zbioru danych
                                        # W przypadku użycia innego zbioru danych należy dostosować nazwę kolumny klasowej
        values = training_data[attribute].unique()  # Wartości atrybutu
        weighted_entropy = 0
        for val in values:
            subset = training_data[training_data[attribute] == val] # Podzbiór danych dla danej wartości atrybutu
            weighted_entropy += (len(subset) / len(training_data)) * self.entropy(subset, "class")  # Oblicza i sumuje ważoną entropię
        # Przyrost informacji: IG(A) = E(S) - Σ (|Sv| / |S|) * E(Sv)
        info_gain = total_entropy - weighted_entropy
        return info_gain

    def best_attribute_by_information_gain(self, training_data, attributes):
        highest_info_gain = -1 # inicjalizacja zmiennej z wartością -1, aby zawsze przytost informacji atrybutu był większy
        attribute_with_highest_ig = None

        for attribute in attributes: # Iteracja po atrybutach
            info_gain = self.information_gain(training_data, attribute) # Obliczanie przyrostu informacji
            if info_gain > highest_info_gain:  # Jeśli obliczony przyrost informacji jest większy niż poprzedni
                highest_info_gain = info_gain
                attribute_with_highest_ig = attribute

        return attribute_with_highest_ig

    def create_subsets_by_attribute(self, training_data, attribute):
        # Tworzenie podzbiorów danych na podstawie wartości atrybutu
        values = training_data[attribute].unique() # Wartości atrybutu
        subsets = {} # Inicjalizacja słownika na podzbiory
        for val in values:
            data_by_attribute = training_data[training_data[attribute] == val]  # Tworzenie podzbioru
            subsets[val] = data_by_attribute.drop(columns=attribute) # Dodanie do słownika i usunięcie kolumny atrybutu
        return subsets

    def most_frequent_value(self, training_data, attribute):
        # Znalezienie najczęściej występującej wartości atrybutu
        return training_data[attribute].mode()[0]

    def id3_algorithm(self, training_data, attributes, test_type="identity", depth=0):
        # Algorytm ID3 do trenowania drzewa decyzyjnego

        # Przypadek końcowy: wszystkie instancje mają tę samą klasę
        if len(training_data["class"].unique()) == 1 or len(attributes) == 0:
            return Node(is_final=True, class_value=training_data["class"].mode()[0])

        # Przypadek końcowy: osiągnięto maksymalną głębokość
        if self.max_depth != None and depth >= self.max_depth:
            return Node(is_final=True, class_value=training_data["class"].mode()[0])

        # Wybór najlepszego atrybutu do podziału
        best_attribute = self.best_attribute_by_information_gain(training_data, attributes)
        updated_attributes = attributes.drop(best_attribute)  # Aktualizacja listy atrybutów

        root = Node(split_attribute=best_attribute, test_type=test_type)  # Utworzenie korzenia z najlepszym atrybutem

        if test_type == "identity":  # Test tożsamościowy
            subsets = self.create_subsets_by_attribute(training_data, best_attribute)  # Tworzenie podzbiorów
            for attribute_value in subsets:
                subset = subsets[attribute_value]  # Podzbiór danych dla wartości atrybutu
                if len(subset) < self.min_samples_split: # Sprawdzenie minimalnej liczby próbek do podziału
                    # Dodanie liścia do drzewa
                    root.add_child(attribute_value, Node(is_final=True, class_value=training_data["class"].mode()[0]))
                else:
                    # Rekurencyjne wywołanie algorytmu ID3, zwiększenie głębokości
                    child = self.id3_algorithm(subset, updated_attributes, test_type="identity", depth=depth + 1)
                    root.add_child(attribute_value, child) # Dodanie dziecka

        elif test_type == "binary":  # Test binarny
            # Wybór najczęściej występującego atrybutu jako atrybut podziału dla testów binarnych
            split_value = self.most_frequent_value(training_data, best_attribute)
            root.split_value = split_value

            left_subset = training_data[training_data[best_attribute] == split_value]  # Lewy podzbiór danych
            right_subset = training_data[training_data[best_attribute] != split_value]  # Prawy podzbiór danych

            if len(left_subset) < self.min_samples_split:   # Jeśli liczba instancji w lewym podzbiorze
                                                            # jest mniejsza niż minimalna liczba próbek
                # Dodaje nowy liść do lewego dziecka węzła, przypisując mu najczęściej występującą klasę
                root.add_child("left", Node(is_final=True, class_value=training_data["class"].mode()[0]))
            else:
                # Rekurencyjne wywołanie algorytmu ID3, zwiekszenie głębokości
                child = self.id3_algorithm(left_subset, updated_attributes, test_type="binary", depth=depth + 1)
                root.add_child("left", child)  # dodanie lewego dziecka węzła

            if len(right_subset) < self.min_samples_split:  # Jeśli liczba instancji w lewym podzbiorze
                                                            # jest mniejsza niż minimalna liczba próbek
                # Dodaje nowy liść do prawego dziecka węzła, przypisując mu najczęściej występującą klasę
                root.add_child("right", Node(is_final=True, class_value=training_data["class"].mode()[0]))
            else:
                # Rekurencyjne wywołanie algorytmu ID3, zwiekszenie głębokości
                child = self.id3_algorithm(right_subset, updated_attributes, test_type="binary", depth=depth + 1)
                root.add_child("right", child)  # dodanie lewego dziecka węzła

        return root
