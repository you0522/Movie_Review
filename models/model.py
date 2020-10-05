import tensorflow as tf
import matplotlib.pyplot as plt


class MODEL:

    def __init__(self, input_size, output_size, layers, loss, optimizer, epochs):
        self.input_size_ = input_size
        self.output_size_ = output_size
        self.layers_ = layers
        self.loss_ = loss
        self.optimizer_ = optimizer
        self.epochs_ = epochs
        self.model_ = self.__build_model__()

    def __build_model__(self):
        model = tf.keras.Sequential()
        for layer in self.layers_:
            model.add(layer)
        model.compile(loss=self.loss_, optimizer=self.optimizer_, metrics=['accuracy'])
        return model

    def __model_summary__(self):
        self.model_.summary()

    def __train__(self, X, y, val_X, val_y):
        self.history_ = self.model_.fit(X, y, epochs=self.epochs_, validation_data=(
            val_X, val_y
        ))
        self.model_.save('models/model.h5')

    def __test__(self, X, y):
        self.test_ = self.model_.evaluate(X, y)
        print("========================= MODEL TEST =========================")
        print("Loss test: {}".format(self.test_[0]))
        print("Accuracy test: {}".format(self.test_[1]))
        print("============================ END ============================")

    def __predict__(self, X):
        self.predict_ = self.model_.predict(X)

    def __plot_graph__(self, string):
        plt.plot(self.history_.history[string])
        plt.plot(self.history_.history['val_' + string])
        plt.xlabel('Epochs')
        plt.ylabel(string)
        plt.legend([string, 'val_' + string])
        plt.show()
