import numpy as np
import matplotlib.pyplot as plt
from hist_plotter import plot_hist

PATH_DRAMATIC_FEATURE = 'dramatic_feature'

class featureManipulator():
    def __init__(self, model, x_test, feature_names, party_dict):
        self.model = model[0]
        self.model_name = model[1]
        self.x_test = x_test
        self.party_dict = party_dict
        self.true_winner = 7
        self.feature_names = feature_names
        self.continuous_data = (['Avg_environmental_importance',
                                    'Avg_government_satisfaction',
                                    'Avg_education_importance',
                                    'Number_of_valued_Kneset_members',
                                    'Avg_monthly_expense_on_pets_or_plants',
                                    'Avg_Residancy_Altitude',
                                    'Yearly_ExpensesK',
                                    'Weighted_education_rank'])
        self.one_hot_data = (['Is_Most_Important_Issue__Financial',
                                    'Is_Most_Important_Issue__Healthcare',
                                    'Is_Most_Important_Issue__Education',
                                    'Is_Most_Important_Issue__Environment',
                                    'Is_Most_Important_Issue__Social',
                                    'Is_Most_Important_Issue__Foreign_Affairs',
                                    'Is_Most_Important_Issue__Other',
                                    'Is_Most_Important_Issue__Military'])

    def find_binary_dramatic_feature(self):
        # continuous data
        true_predictions = self.model.predict(self.x_test)
        for col in range(self.x_test.shape[1]):
            if self.feature_names[col] in self.one_hot_data:
                alterated_x = self.set_one_hot(self.x_test, col)
                predictions = self.model.predict(alterated_x)
                winner = max(set(predictions), key=predictions.tolist().count)
                if winner != self.true_winner:
                    title = self.feature_names[col] + ' set to 1 results'
                    path = PATH_DRAMATIC_FEATURE + '\\' + self.feature_names[col] + '_set.png'
                    plot_hist(path, title, predictions, true_predictions,
                              'manipulated', 'original', self.party_dict)

                    winner_name = self.party_dict[winner]
                    print("If ", self.feature_names[col], " will be important to everyone, that will cause ",
                          winner_name,
                          " to win")

    def set_one_hot(self, x_data, col):
        assert self.feature_names[col] in self.one_hot_data
        alterated = np.array(x_data, copy=True)
        for column in range(self.x_test.shape[1]):
            if self.feature_names[column] in self.one_hot_data:
                alterated[:, column] = 0
        alterated[:, col] = 1
        return alterated

    def find_continuous_dramatic_feature(self):
        # continuous data
        true_predictions = self.model.predict(self.x_test)
        for col in range(self.x_test.shape[1]):
            if self.feature_names[col] in self.continuous_data:
                for c in np.arange(0, 10, 0.1):
                    alterated_x = self.alterate_column(self.x_test, col, c)
                    predictions = self.model.predict(alterated_x)
                    winner = max(set(predictions), key=predictions.tolist().count)
                    if winner != self.true_winner:
                        title = self.feature_names[col] + ' grow by ' + str(np.round(c, 2)) + ' results'

                        path = PATH_DRAMATIC_FEATURE + '\\' + self.feature_names[col] + '_increased.png'

                        plot_hist(path, title, predictions, true_predictions,
                                  'manipulated', 'original', self.party_dict)
                        winner_name = self.party_dict[winner]
                        print("If ", self.feature_names[col], " will grow by ", np.round(c, 2), ", that will cause ", winner_name, " to win")
                        break
                for c in np.arange(0, -10, -0.1):
                    alterated_x = self.alterate_column(self.x_test, col, c)
                    predictions = self.model.predict(alterated_x)
                    winner = max(set(predictions), key=predictions.tolist().count)
                    if winner != self.true_winner:
                        plt.hist([predictions, true_predictions], bins=13, label=['manipulated', 'original'])
                        plt.legend()
                        title = self.feature_names[col] + ' decreased by ' + str(np.round(c, 2)) + ' results'
                        plt.ylabel('number of votes')
                        plt.xlabel('party number')
                        plt.title(title)
                        fig = plt.gcf()
                        path = PATH_DRAMATIC_FEATURE + '\\' + self.feature_names[col] + '_decreased.png'
                        fig.savefig(path, bbox_inches='tight')
                        plt.show()
                        winner_name = self.party_dict[winner]
                        print("If ", self.feature_names[col], " will decrease by ", np.round(c, 2), ", that will cause ", winner_name, " to win")
                        break

    def alterate_column(self, x_data, col, c):
        assert self.feature_names[col] in self.continuous_data
        alterated = np.array(x_data, copy=True)
        alterated[:, col] = c*alterated[:, col]
        return alterated

