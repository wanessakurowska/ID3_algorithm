from evaluation_metrics import get_all_evaluation_metrics, test_small_dataset
import pandas as pd

car_data = pd.read_csv('car.data', header=None,
                   names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"])

ttt_data = pd.read_csv('tic-tac-toe.data', header=None,
                   names=["top-left-square", "top-middle-square", "top-right-square", "middle-left-square", "middle-middle-square", "middle-right-square", "bottom-left-square", "bottom-middle-square", "bottom-right-square", "class"])


# Wywołanie metod zwracania miar jakości
# get_all_evaluation_metrics(car_data, 2 , max_depth=6, min_samples_split=10)
# get_all_evaluation_metrics(ttt_data, 20, max_depth=6, min_samples_split=10)

# Wywołanie testu na zmniejszonym zbiorze danych
# test_small_dataset(car_data, 10, 100)
test_small_dataset(ttt_data, 10, 100)


