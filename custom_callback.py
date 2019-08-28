import keras


class PrintRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("\n ** val/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))
