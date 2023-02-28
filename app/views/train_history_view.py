import matplotlib.pyplot as plt

from common.app_base import AppBase


class TrainHistoryView(AppBase):
    """
    モデルの学習履歴を表示する画面
    """

    def __init__(self):
        super().__init__()

    def show(self, history: dict):
        """
        モデルの学習履歴を表示する
        """

        if "val_accuracy" in history.keys():

            # 訓練データと検証データの学習履歴をプロット

            accuracy = history["accuracy"]
            val_accuracy = history["val_accuracy"]
            loss = history["loss"]
            val_loss = history["val_loss"]
            epochs = range(1, len(accuracy) + 1)

            fig = plt.figure()
            fig.canvas.set_window_title("TrainHistory")

            # 1行、2列レイアウトの1番目(精度の履歴)
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.set_title("Training and validation accuracy")
            ax1.plot(epochs, accuracy, "bo", label="Training acc")
            ax1.plot(epochs, val_accuracy, "b", label="Validation acc")
            ax1.legend()
            ax1.grid()

            # 1行、2列レイアウトの2番目(損失の履歴)
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.set_title("Training and validation loss")
            ax2.plot(epochs, loss, "bo", label="Training acc")
            ax2.plot(epochs, val_loss, "b", label="Validation loss")
            ax2.legend()
            ax2.grid()

            plt.tight_layout()
            plt.show()

        else:

            # 訓練データの学習履歴をプロット

            accuracy = history["accuracy"]
            loss = history["loss"]
            epochs = range(1, len(accuracy) + 1)

            fig = plt.figure()
            fig.canvas.set_window_title("TrainHistory")

            # 1行、2列レイアウトの1番目(精度の履歴)
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.set_title("Training accuracy")
            ax1.plot(epochs, accuracy, "bo", label="Training acc")
            ax1.legend()
            ax1.grid()

            # 1行、2列レイアウトの2番目(損失の履歴)
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.set_title("Training loss")
            ax2.plot(epochs, loss, "bo", label="Training acc")
            ax2.legend()
            ax2.grid()

            plt.tight_layout()
            plt.show()
