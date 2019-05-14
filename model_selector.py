import matplotlib.pyplot as plt
import os
import numpy as np
import copy
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

PATH_WINNER_PARTY_PLOTS = 'Winner_party_plots'
PATH_VOTE_PREDICTION_PLOTS = 'Vote_prediction_plots'
PATH_DIVISION_PREDICTION_PLOTS = 'Division_prediction_plots'
THRESHOLD_PROBA = 0.2 #20%
EMPTY_DICT = {
    "12": set(),
    "11": set(),
    "10": set(),
    "9": set(),
    "8": set(),
    "7": set(),
    "6": set(),
    "5": set(),
    "4": set(),
    "3": set(),
    "2": set(),
    "1": set(),
    "0": set(),
}

class modelSelector():
    def __init__(self, id_train, x_train, y_train,
                 id_val, x_val, y_val,
                 id_test, x_test, y_test,
                 models, model_names):
        self.id_train = id_train
        self.x_train = x_train
        self.y_train = y_train

        self.id_val = id_val
        self.x_val = x_val
        self.y_val = y_val

        self.id_test = id_test
        self.x_test = x_test
        self.y_test = y_test
        self.model_list = models
        self.model_names_list = model_names
        #self.class_dict = class_dict
        self.num_of_classes = 13
        self.winner_acc = []
        self.best_models_for_winner_prediction = None  # (model, model_name) automatically selected models
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
        === This is first mandatory prediction ===
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
        highest_norm = 0
        for model, model_name in zip(self.model_list, self.model_names_list):
            predictions = model.predict(self.x_val)

            pred_hist = [0] * self.num_of_classes
            for pred in predictions:
                pred_hist[pred] += 1
            curr_norm = np.linalg.norm(pred_hist, ord=np.inf)

            if graphic:
                plt.hist([predictions, self.y_val.tolist()], bins=13, label=['predictions', 'test data'])
                supttl = 'Winner party predictions'
                plt.suptitle(supttl)
                plt.ylim((0, 850))
                ttl = 'model: ' + model_name + "inf norm=" + str(curr_norm)
                plt.title(ttl)
                plt.legend()
                fig = plt.gcf()
                path = PATH_WINNER_PARTY_PLOTS + '\\' + model_name + '_fig.png'
                fig.savefig(path, bbox_inches='tight')
                plt.show()

            pred_winner = max(set(predictions), key=predictions.tolist().count)
            real_winner = max(set(self.y_test), key=self.y_test.tolist().count)

            if pred_winner == real_winner:
                if curr_norm > highest_norm:
                    highest_norm = curr_norm
                    self.best_models_for_winner_prediction=(model, model_name)
        if self.best_models_for_winner_prediction is not None:
            print("best model for winner prediction is ", self.best_models_for_winner_prediction[1])
        else:
            print("All models are terrible for winner prediction task")

    def score_division_prediction(self, graphic=True):
        '''
        === This is second mandatory prediction ===
        :param graphic:
        :return:
        '''
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
            predictions = model.predict(self.x_val)

            for pred in predictions:
                pred_hist[pred] += 1
            for label in self.y_test.tolist():
                true_hist[label] += 1

            dist = np.linalg.norm(np.array(pred_hist) - np.array(true_hist))
            self.division_dist.append(dist)

            if graphic:
                plt.hist([predictions, self.y_val.tolist()], bins=13, label=['predictions', 'test data'])
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

        print("best model for vote division is ", self.best_model_for_division_prediction[1])
        return self.division_dist

    def score_vote_prediction(self, graphic = True):
        '''
        === This is third mandatory prediction ===
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
        true_dict = copy.deepcopy(EMPTY_DICT)
        fill_true_dict(true_dict, self.id_val, self.y_val)
        for model, model_name in zip(self.model_list, self.model_names_list):
            predictions_proba = model.predict_proba(self.x_val)
            pred_dict = copy.deepcopy(EMPTY_DICT)
            fill_pred_dict(pred_dict, self.id_val, predictions_proba)


            print("testing - ", model_name)
            score = 0
            for key in true_dict:
                true_set = true_dict[key]
                pred_set = pred_dict[key]
                intersec = len(true_set.intersection(pred_set))
                forgotten_voters = len(true_set.difference(pred_set))
                false_riders = len(pred_set.difference(true_set))
                print("for party:" + key + ", the intersection size is ", intersec, ", forgotten voters:", forgotten_voters, " anf falce riders=", false_riders)
                score += intersec
                score -= false_riders
                score -= forgotten_voters
            print("score = ", score)

        #print("best model for vote classification is ", self.best_model_for_vote_prediction[1])
        #return self.vote_acc

def fill_true_dict(true_dict, ids, labels):
    for label, id in zip(labels, ids):
        true_dict[str(label)].add(id)
    return true_dict

def fill_pred_dict(pred_dict, ids, probas):
    for proba, id in zip(probas, ids):
        for p,tag in zip(proba, enumerate(proba)):
            if p >= THRESHOLD_PROBA:
                pred_dict[str(tag[0])].add(id)