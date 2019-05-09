import matplotlib.pyplot as plt
import os

PATH_WINNER_PARTY_PLOTS = 'Winner_party_plots'

class modelSelector():
    def __init__(self, x_train, y_train, x_test, y_test, models, model_names, class_dict):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model_list = models
        self.model_names_list = model_names
        self.class_dict = class_dict
        self.num_of_classes = len(class_dict)
        self.fit_scores = None
        self.winner_acc = None

    def fit(self):
        for model, model_name in zip(self.model_list, self.model_names_list):
            print("training model ", model_name)
            model.fit(self.x_train, self.y_train)
            #self.fit_scores.append(model.score(self.x_train, self.y_train))

    def score_who_win(self, graphic = True):
        '''
        This function provides a score against validation (test) data
        for each model, about it's prediction who will win the
        elections.
        Notice that the tags ratio should be equal in all sets!
        :return: scores of performance of each model
        '''

        if self.winner_acc is not None and graphic is False:
            return self.winner_acc
        if graphic:
            if not os.path.exists(PATH_WINNER_PARTY_PLOTS):
                os.mkdir(PATH_WINNER_PARTY_PLOTS)

        self.winner_acc = []
        for model, model_name in zip(self.model_list, self.model_names_list):
            predictions = model.predict(self.x_test)

            if graphic:
                pred_counter = [0] * self.num_of_classes
                real_counter = [0] * self.num_of_classes
                for pred in predictions:
                    pred_counter[pred] += 1
                for vote in self.y_test.tolist():
                    real_counter[vote] += 1
                xa = range(self.num_of_classes)
                plt.hist([predictions,self.y_test.tolist()], bins=13, label=['predictions', 'test data'])
                supttl = 'Winner party predictions'
                plt.suptitle(supttl)
                plt.ylim((0,850))
                ttl = 'model' + model_name
                plt.title(ttl)
                plt.legend()
                fig = plt.gcf()
                path = PATH_WINNER_PARTY_PLOTS + '\\' + supttl + '_' + model_name + '_fig.png'
                fig.savefig(path, bbox_inches='tight')
                plt.show()


            pred_winner = max(set(predictions), key=predictions.tolist().count)
            real_winner = max(set(self.y_test), key=self.y_test.tolist().count)

            if pred_winner != real_winner:
                self.winner_acc.append(0)
            else:
                self.winner_acc.append(100)

        return self.winner_acc




