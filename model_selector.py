import matplotlib.pyplot as plt
import os
import numpy as np
import copy
import pprint as pp
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from hist_plotter import plot_hist

PATH_WINNER_PARTY_PLOTS = 'Winner_party_plots'
PATH_VOTE_PREDICTION_PLOTS = 'Vote_prediction_plots'
PATH_DIVISION_PREDICTION_PLOTS = 'Division_prediction_plots'
PATH_CONFUSION = 'confusion_matrix'
THRESHOLD_PROBA = 0.5 #50%
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
                 models, model_names, names_dict):
        self.party_dict = names_dict
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
        self.best_model_for_winner_prediction = None  # (model, model_name) automatically selected models
        self.vote_acc = []
        self.best_model_for_vote_prediction = None  # (model, model_name) automatically selected models
        self.division_dist = []
        self.best_model_for_division_prediction = None

    def fit(self):
        for model, model_name in zip(self.model_list, self.model_names_list):
            print("training model ", model_name)
            model.fit(self.x_train, self.y_train)

    def get_best_winner_prediction_model(self):
        return self.best_model_for_winner_prediction

    def score_accuracy(self, graphic=True):
        for model, model_name in zip(self.model_list, self.model_names_list):
            acc = model.score(self.x_val, self.y_val)
            print("Model ", model_name, " reached ", np.round(acc*100,2), "% accuracy.")

    def score_who_win(self, graphic=True):
        '''
        === This is first mandatory prediction ===
        This function provides a score against validation (test) data
        for each model, about it's prediction who will win the
        elections.
        Notice that the tags ratio should be equal in all sets!
        best models will be saved in
        :return: scores of performance of each model
        '''

        if graphic:
            if not os.path.exists(PATH_WINNER_PARTY_PLOTS):
                os.mkdir(PATH_WINNER_PARTY_PLOTS)

        highest_norm = 0
        for model, model_name in zip(self.model_list, self.model_names_list):
            predictions = model.predict(self.x_val)

            pred_hist = [0] * self.num_of_classes
            for pred in predictions:
                pred_hist[pred] += 1
            curr_norm = np.linalg.norm(pred_hist, ord=np.inf)

            if graphic:
                supttl = 'Winner party predictions'
                ttl = 'model: ' + model_name + " inf norm=" + str(curr_norm)
                path = PATH_WINNER_PARTY_PLOTS + '\\' + model_name + '_fig.png'

                plot_hist(path, ttl, predictions, self.y_val.tolist(),
                          'predictions', 'true labels', self.party_dict, suptitle=supttl)

            pred_winner = max(set(predictions), key=predictions.tolist().count)
            real_winner = max(set(self.y_test), key=self.y_test.tolist().count)

            if pred_winner == real_winner:
                if curr_norm > highest_norm:
                    highest_norm = curr_norm
                    self.best_model_for_winner_prediction=(model, model_name)
        if self.best_model_for_winner_prediction is not None:
            print("best model for winner prediction is ", self.best_model_for_winner_prediction[1])
        else:
            print("All models are terrible for winner prediction task")

    def score_division_prediction(self, graphic=True):
        '''
        === This is second mandatory prediction ===
        :param graphic:
        :return:
        '''
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
                supttl = 'Votes division predictions - hist dist = ' + str(np.round(dist))
                ttl = 'model' + model_name
                path = PATH_DIVISION_PREDICTION_PLOTS + '\\' + model_name + '_fig.png'
                plot_hist(path, ttl, predictions, self.y_val.tolist(),
                          'predictions', 'true labels', self.party_dict, suptitle=supttl)

            if dist < shortest_dist:
                shortest_dist = dist
                self.best_model_for_division_prediction = (model, model_name)

        print("best model for vote division is ", self.best_model_for_division_prediction[1])
        return self.division_dist

    def score_transportation_prediction(self, graphic = True):
        '''
        === This is third mandatory prediction ===
        This function provides a score against validation (test) data
        for each model, about it's vote prediction.
        Notice that the tags ratio should be equal in all sets!
        best models will be saved in
        :return: scores of performance of each model
        '''
        if graphic:
            if not os.path.exists(PATH_VOTE_PREDICTION_PLOTS):
                os.mkdir(PATH_VOTE_PREDICTION_PLOTS)

        self.vote_acc = []
        true_dict = copy.deepcopy(EMPTY_DICT)
        fill_true_dict(true_dict, self.id_val, self.y_val)
        best_score = 0
        for model, model_name in zip(self.model_list, self.model_names_list):
            predictions_proba = model.predict_proba(self.x_val)
            pred_dict = copy.deepcopy(EMPTY_DICT)
            fill_pred_dict(pred_dict, self.id_val, predictions_proba)


            #print("testing - ", model_name)
            score = 0
            forgotten_voters_list = []
            intersec_list = []
            false_riders_list = []
            for key in true_dict:
                true_set = true_dict[key]
                pred_set = pred_dict[key]
                intersec = len(true_set.intersection(pred_set))
                forgotten_voters = len(true_set.difference(pred_set))
                false_riders = len(pred_set.difference(true_set))
                #print("for party:" + key + ", the intersection size is ", intersec, ", forgotten voters:", forgotten_voters, " and false riders:", false_riders)
                forgotten_voters_list.append(forgotten_voters)
                false_riders_list.append(false_riders)
                intersec_list.append(intersec)
                score += intersec
                score -= false_riders
                score -= forgotten_voters
            #print("score = ", score)
            if score > best_score:
                best_score = score
                self.best_model_for_vote_prediction = (model, model_name)

            if graphic:
                index = np.arange(len(forgotten_voters_list))
                bar_width = 0.2
                forgotten_voters_list = [-1*i for i in forgotten_voters_list]
                false_riders_list = [-1*i for i in false_riders_list]
                plt.bar(index-bar_width, forgotten_voters_list, bar_width, color='grey', label='T-P: Forgotten voters')
                plt.bar(index, intersec_list, bar_width, color='green', label='T AND P: True voters with transportation')
                plt.bar(index+bar_width, false_riders_list, bar_width, color='red', label='P-T: False voters with free ride')
                supttl = 'Transportation predictions - score = ' + str(score)
                keys = list(self.party_dict.keys())
                values = list(self.party_dict.values())
                plt.xticks(keys, values, rotation='vertical')
                plt.suptitle(supttl)
                plt.ylim([-150,450])
                plt.grid()
                ttl = 'model: ' + model_name
                plt.title(ttl)
                plt.legend()
                fig = plt.gcf()
                path = PATH_VOTE_PREDICTION_PLOTS + '\\' + model_name + '_fig.png'
                fig.savefig(path, bbox_inches='tight')
                plt.show()

        print("best model for transportation is ", self.best_model_for_vote_prediction[1])

    def predict_winner(self, x_test):
        model, model_name = self.best_model_for_winner_prediction
        predictions = model.predict(x_test)
        winner = max(set(predictions), key=predictions.tolist().count)
        winner_name = self.party_dict[winner]
        print(model_name," prediction - ", winner_name, " party will win the elections.")
        return winner

    def predict_vote_division(self, x_test):
        model, model_name = self.best_model_for_division_prediction
        predictions = model.predict(x_test)
        pred_hist = [0] * self.num_of_classes
        for pred in predictions:
            pred_hist[pred] += 1
        total = sum(pred_hist)
        print(model_name," prediction - Vote division:")
        for idx, num in enumerate(pred_hist):
            print("Party ", self.party_dict[idx], ":", np.round((num/total)*100, 1), "%")

    def predict_transportation(self, x_test):
        model, model_name = self.best_model_for_vote_prediction
        predictions_proba = model.predict_proba(x_test)
        pred_dict = copy.deepcopy(EMPTY_DICT)
        fill_pred_dict(pred_dict, self.id_val, predictions_proba)
        print(model_name," prediction - Transportation predictions")
        for key in pred_dict:
            print("Party ", self.party_dict[int(key)], ":", pred_dict[key])

    def draw_conf_matrix(self):
        model, model_name = self.best_model_for_vote_prediction
        predictions = model.predict(self.x_test)

        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        class_names = []
        for key in range(self.num_of_classes):
            class_names.append(self.party_dict[key])
        plot_confusion_matrix(self.y_test, predictions, title='Confusion matrix of '+model_name, classes=class_names)
        fig = plt.gcf()
        plt.show()
        path = PATH_CONFUSION + '\\' + 'confusion_fig.png'
        fig.savefig(path, bbox_inches='tight')



def fill_true_dict(true_dict, ids, labels):
    for label, id in zip(labels, ids):
        true_dict[str(label)].add(id)
    return true_dict

def fill_pred_dict(pred_dict, ids, probas):
    for proba, id in zip(probas, ids):
        for p,tag in zip(proba, enumerate(proba)):
            if p >= THRESHOLD_PROBA:
                pred_dict[str(tag[0])].add(id)


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
