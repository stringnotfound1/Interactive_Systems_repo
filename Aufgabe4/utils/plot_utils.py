from matplotlib import pyplot as plt
import numpy as np


def plot_model_history(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def plot_result_examples(model, X_test, y_test, img_rows, img_cols):
    """
    The predict_classes function outputs the highest probability class
    according to the trained classifier for each input example.
    :return:
    """
    predicted_classes = model.predict_classes(X_test)

    # Check which items we got right / wrong
    correct_indices = np.nonzero(predicted_classes == y_test)[0]
    incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

    plt.figure()
    for i, correct in enumerate(correct_indices[:9]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_test[correct].reshape(img_rows, img_cols), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))

    plt.figure()
    for i, incorrect in enumerate(incorrect_indices[:9]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_test[incorrect].reshape(img_rows, img_cols), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))

    plt.show()