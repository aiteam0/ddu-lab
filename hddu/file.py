import pickle
import os


def save_documents_to_pkl(documents, filepath):
    abs_path = os.path.abspath(filepath)
    base_path = os.path.splitext(abs_path)[0]
    pkl_path = f"{base_path}.pkl"

    with open(pkl_path, "wb") as f:
        pickle.dump(documents, f)


def load_documents_from_pkl(filepath):

    abs_path = os.path.abspath(filepath)
    base_path = os.path.splitext(abs_path)[0]
    pkl_path = f"{base_path}.pkl"

    with open(pkl_path, "rb") as f:
        documents = pickle.load(f)
    return documents
