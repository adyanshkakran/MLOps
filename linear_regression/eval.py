from sklearn.metrics import mean_squared_error, mean_absolute_error

# Description: This file contains the functions used to evaluate the model
def get_accuracy(model, x_test, y_test):
    return model.score(x_test, y_test)

def get_all_metrics(model, x_test, y_test):
    predictions = model.predict(x_test)
    return {
        'accuracy': get_accuracy(model, x_test, y_test),
        'mean_squared_error': mean_squared_error(y_test, predictions),
        'mean_absolute_error': mean_absolute_error(y_test, predictions)
    }