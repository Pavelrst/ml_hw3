import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

PATH_WINNER_PARTY_PLOTS = 'Winner_party_plots'
PATH_VOTE_PREDICTION_PLOTS = 'Vote_prediction_plots'
PATH_DIVISION_PREDICTION_PLOTS = 'Division_prediction_plots'

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
        self.winner_acc = []
        self.best_models_for_winner_prediction = []  # (model, model_name) automatically selected models
        self.vote_acc = []
        self.best_model_for_vote_prediction = None  # (model, model_name) automatically selected models
        self.division_dist = []
        self.best_model_for_division_prediction = None

    def fit(self):
        for model, model_name in zip(self.model_list, self.model_names_list):
            print("training model ", model_name)
            model.fit(self.x_train, self.y_train)

    def score_who_win(self, graphic = True):
        '''
        This function provides a score against validation (test) data
        for each model, about it's prediction who will win the
        elections.
        Notice that the tags ratio should be equal in all sets!
        best models will be saved in
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
                plt.hist([predictions, self.y_test.tolist()], bins=13, label=['predictions', 'test data'])
                supttl = 'Winner party predictions'
                plt.suptitle(supttl)
                plt.ylim((0, 850))
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
                self.best_models_for_winner_prediction.append((model, model_name))
                self.winner_acc.append(100)

        return self.winner_acc

    def score_vote_prediction(self, graphic = True):
        '''
        This function provides a score against validation (test) data
        for each model, about it's vote prediction.
        Notice that the tags ratio should be equal in all sets!
        best models will be saved in
        :return: scores of performance of each model
        '''
        if self.vote_acc is not None and graphic is False:
            return self.vote_acc
        if graphic:
            if not os.path.exists(PATH_VOTE_PREDICTION_PLOTS):
                os.mkdir(PATH_VOTE_PREDICTION_PLOTS)

        self.vote_acc = []
        best_acc = 0
        for model, model_name in zip(self.model_list, self.model_names_list):
            predictions = model.predict(self.x_test)
            acc = accuracy_score(self.y_test, predictions)
            self.vote_acc.append(acc)
            print("model ", model_name, "reached ", str(np.round(acc*100,3)) , "% accuracy.")
            if acc > best_acc:
                best_acc = acc
                self.best_model_for_vote_prediction = (model, model_name)
        print("best model for vote classification is ", self.best_model_for_vote_prediction[1])
        return self.vote_acc

    def score_division_prediction(self, graphic=True):
        if self.division_dist is not None and graphic is False:
            return self.division_dist
        if graphic:
            if not os.path.exists(PATH_DIVISION_PREDICTION_PLOTS):
                os.mkdir(PATH_DIVISION_PREDICTION_PLOTS)

        self.division_dist = []
        shortest_dist = np.inf
        for model, model_name in zip(self.model_list, self.model_names_list):
            pred_hist = [0]*self.num_of_classes
            true_hist = [0]*self.num_of_classes
            predictions = model.predict(self.x_test)

            for pred in predictions:
                pred_hist[pred] += 1
            for label in self.y_test.tolist():
                true_hist[label] += 1

            dist = np.linalg.norm(np.array(pred_hist) - np.array(true_hist))
            self.division_dist.append(dist)

            if graphic:
                plt.hist([predictions, self.y_test.tolist()], bins=13, label=['predictions', 'test data'])
                supttl = 'Votes division predictions - hist dist = ' + str(np.round(dist))
                plt.suptitle(supttl)
                plt.ylim((0, 850))
                ttl = 'model' + model_name
                plt.title(ttl)
                plt.legend()
                fig = plt.gcf()
                path = PATH_DIVISION_PREDICTION_PLOTS + '\\' + model_name + '_fig.png'
                fig.savefig(path, bbox_inches='tight')
                plt.show()

            if dist < shortest_dist:
                shortest_dist = dist
                self.best_model_for_division_prediction = (model, model_name)
                
        return self.division_dist

