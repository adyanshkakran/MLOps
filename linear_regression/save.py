import pickle as pkl

def save_model(model):
    with open('model.pkl', 'wb') as file:
        pkl.dump(model, file)