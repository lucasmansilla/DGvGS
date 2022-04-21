import pickle


def read_dict(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_dict(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
