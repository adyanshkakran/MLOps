from model import load_model
from train import train_model
from data import load_data
from eval import get_all_metrics
from save import save_model

def train_new_model():
    model = load_model()

    separateTest = input("Do you want to separate the test data? (y/n) ")
    x_train, x_test, y_train, y_test = load_data(separateTest == "y")

    model = train_model(model, x_train, y_train)

    metrics = get_all_metrics(model, x_test, y_test)

    for metric in metrics:
        print(f"{metric}: {metrics[metric]}")
    save = input("Do you want to save the new model? (y/n) ")

    if save == "y":
        save_model(model)
        print("Model saved")

def main():
    option = int(
        input(
            """What do you want to do?
1. Train a new model
2. Retrain a model
3. Evaluate a model
4. Save a model
5. Make predictions using existing model
6. Exit
"""
        )
    )
    options = {
        1: train_new_model,
    }
    options[option]()


if __name__ == "__main__":
    main()
