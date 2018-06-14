from keras.models import Sequential
from keras.layers import Dropout, Convolution2D, MaxPooling2D, Flatten, Dense, Activation


class CNNModel:
    img_rows = 28
    img_cols = 28

    @staticmethod
    def load_inputshape():
        return CNNModel.img_rows, CNNModel.img_cols, 1

    @staticmethod
    def reshape_input_data(x_train, x_test):
        x_train = x_train.reshape(x_train.shape[0], CNNModel.img_rows, CNNModel.img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], CNNModel.img_rows, CNNModel.img_cols, 1)
        return x_train, x_test


    @staticmethod
    def load_model(classes=10):
        # TODO build your own model here
        return model
