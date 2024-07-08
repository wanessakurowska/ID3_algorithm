class Node:
    def __init__(self, split_attribute=None, split_value=None, is_final=False, class_value=None, children=None,
                 test_type="identity"):
        self.split_attribute = split_attribute  # Atrybut używany do podziału na tym węźle
        self.split_value = split_value  # Wartość podziału dla testów binarnych
        self.is_final = is_final  # Czy węzeł jest liściem (węzłem końcowym)
        self.class_value = class_value  # Przewidywana klasa, jeśli węzeł jest liściem
        self.test_type = test_type  # Typ testu: "identity" (tożsamościowy) lub "binary" (binarny)

        # Inicjalizacja dzieci węzła
        if children is not None:
            self.children = children
        else:
            self.children = {}  # Słownik przechowujący dzieci węzła

    def add_child(self, attribute_value, child):
        # Dodaje dziecko do węzła
        self.children[attribute_value] = child

    def predict(self, instance):
        # Przewidywanie klasy dla danej instancji danych
        if self.is_final:
            return self.class_value  # Zwracanie klasy jeśli węzeł jest liściem

        attribute_value = instance[self.split_attribute]  # Pobieranie wartość atrybutu do podziału

        if self.test_type == "identity":  # Test tożsamościowy
            if attribute_value in self.children:  # Jeśli wartość atrybutu jest w dzieciach węzła
                child_node = self.children[attribute_value]  # Pobierz odpowiednie dziecko
                return child_node.predict(instance)   # Rekurencyjne wywołanie metody predict na dziecku
            else:
                return None  # Jeśli wartość atrybutu nie jest w dzieciach, zwróć None

        elif self.test_type == "binary":  # Test binarny
            if attribute_value == self.split_value:  # Jeśli wartość atrybutu jest równa wartości podziału
                return self.children["left"].predict(instance)  # Rekurencyjne wywołanie metody na lewym dziecku
            else:
                return self.children["right"].predict(instance)  # Rekurencyjne wywołanie metody na prawym dziecku

    def depth(self):
        # Obliczanie głębokości drzewa
        if self.is_final:
            return 0  # Jeśli węzeł jest liściem, zwraca 0

        # Obliczenie głebokości dzieci węzła
        max_depth = 0
        for child in self.children.values():
            child_depth = child.depth()
            if child_depth > max_depth:
                max_depth = child_depth
        return 1 + max_depth

    def num_leaves(self):
        if self.is_final:
            return 1  #  Jeśli węzeł jest liściem, zwraca 1

        # Obliczenie sumy liści dzieci węzła
        total_leaves = 0
        for child in self.children.values():
            total_leaves += child.num_leaves()
        return total_leaves

