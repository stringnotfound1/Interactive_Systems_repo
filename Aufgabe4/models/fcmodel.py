
class FCModel:

    @staticmethod
    def load_inputshape():
        return (784,)

    @staticmethod
    def reshape_input_data(x_train, x_test):
        return x_train, x_test

    @staticmethod
    def load_model(classes=10):
        # TODO build your own model here
        return model
