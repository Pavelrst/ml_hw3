
class modelSelector():
    def __init__(self, x_train, y_train, x_test, y_test, models, class_dict):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model_list = models
        self.class_dict = dict
        self.fit_scores = None
        self.winner_acc = None

    def fit(self):
        for model in self.model_list:
            print("training model ", model)
            model.fit(self.x_train, self.y_train)
            #self.fit_scores.append(model.score(self.x_train, self.y_train))

    def score_who_win(self):
        '''
        This function provides a score against validation (test) data
        for each model, about it's prediction who will win the
        elections.
        Notice that the tags ratio should be equal in all sets!
        :return: scores of performance of each model
        '''

        if self.winner_acc is not None:
            return self.winner_acc

        self.winner_acc = []
        for model in self.model_list:
            predictions = model.predict(self.x_test)

            pred_winner = max(set(predictions), key=predictions.tolist().count)
            real_winner = max(set(self.y_test), key=self.y_test.tolist().count)

            if pred_winner != real_winner:
                self.winner_acc.append(0)
            else:
                self.winner_acc.append(100)

        return self.winner_acc




