import tensorflow as tf

class Model_loader:
    def __init(self, model_directory):
        self.model = tf.keras.models.load_model(model_directory)
        