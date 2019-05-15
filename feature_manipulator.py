
class featureManipulator():
    def __init__(self, model, x_test):
        self.model = model
        self.x_test = x_test

    def find_dramatic_feature(self):
        for col in range(self.x_test.shape[1]):
            alterated_x = self.alterate_column(self.x_test, col, 2)

    def alterate_column(self, x_data, col, c):
        return c*x_data[:, col]

