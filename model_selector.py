
class modelSelector():
    def __init__(self, x_train, y_train, x_test, y_test, models):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model_list = models

    def eval_models(self):
        for model in self.model_list:
            model.fit(self.x_train, self.y_train)
            print(model.score(self.x_test, self.y_test))

