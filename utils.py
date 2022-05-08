import pickle


def clf_performance(classifier, model_name):
    print(model_name)
    print('Best Score: ' + str(classifier.best_score_))
    print('Best Parameters: ' + str(classifier.best_params_))


def save_obj(file_name: str = "pickle", **kwargs):
    """This function allows you to save one or more objects and / or variables in a pickle file.
    The elements are put in a list and then saved

    Args:
        file_name (str, optional): Name of the pickle file. Defaults to "pickle".
    """
    pickle_out = open(f"{file_name}", "wb")
    pickle.dump(list(kwargs.values()), pickle_out)
    pickle_out.close()


def load_pickle_obj(file_name: str = "pickle"):
    with open(f"{file_name}", "rb") as pickle_in:
        pickle_obj = pickle.load(pickle_in)
    return pickle_obj
