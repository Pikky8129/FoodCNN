import matplotlib.pyplot as plt
from keras.models import Sequential

from common.app_base import AppBase


class FilterVisualizationView(AppBase):
    """
    畳み込み層のフィルタを可視化する画面
    """

    def __init__(self):
        super().__init__()

    def show_all_conv_layers(self,  model: Sequential) -> None:
        """
        すべての畳み込み層のフィルタを可視化する

        Parameters
        ----------
        model: Sequential
            学習済みのモデル
        """

        for layer_idx, layer in enumerate(model.layers):

            if "conv" not in layer.name:
                continue

            # 重みを取得する
            filters, biases = layer.get_weights()
            input_channels = filters.shape[2]       # 入力チャネル数
            output_channels = filters.shape[3]      # 出力チャネル数(フィルタ数)

            # 重みを正規化する
            f_min, f_max = filters.min(), filters.max()
            filters = (filters - f_min) / (f_max - f_min)

            max_rows = 6 if output_channels > 6 else output_channels
            max_cols = 8 if input_channels > 8 else input_channels
            fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols)
            fig.canvas.set_window_title("Filter Visualization")
            fig.suptitle("layer_idx={}, shape={}".format(layer_idx+1, filters.shape))

            for i in range(max_rows):
                for j in range(max_cols):
                    axes[i, j].imshow(filters[:, :, j, i], cmap="gray")
                    axes[i, j].set_title("[:, :, {}, {}]".format(j, i))
                    axes[i, j].set_xticks([])
                    axes[i, j].set_yticks([])

            # plt.tight_layout()
            plt.show()

    def show_first_conv_layer(self,  model: Sequential) -> None:
        """
        最初の畳み込み層のフィルタを可視化する

        Parameters
        ----------
        model: Sequential
            学習済みのモデル
        """

        for layer_idx, layer in enumerate(model.layers):

            if "conv" not in layer.name:
                continue

            # 重みを取得する
            filters, biases = layer.get_weights()
            input_channels = filters.shape[2]       # 入力チャネル数
            output_channels = filters.shape[3]      # 出力チャネル数(フィルタ数)

            # カラー画像を入力として受け取る場合
            if input_channels == 3:

                # 重みを正規化する
                f_min, f_max = filters.min(), filters.max()
                filters = (filters - f_min) / (f_max - f_min)

                max_rows = 6
                max_cols = 4
                fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols)
                fig.canvas.set_window_title("Filter Visualization")
                fig.suptitle("layer_idx={}, shape={}".format(layer_idx+1, filters.shape))

                # 各フィルタをチャネルごとにプロットする
                for row_idx in range(6):

                    # Rチャネルを白黒画像として表示
                    axes[row_idx, 0].imshow(filters[:, :, 0, row_idx], cmap='gray')
                    axes[row_idx, 0].set_title("R")
                    axes[row_idx, 0].set_ylabel("[:, :, 0, {}]".format(row_idx), labelpad=30, rotation=360)
                    axes[row_idx, 0].set_xticks([])
                    axes[row_idx, 0].set_yticks([])

                    # Gチャネルを白黒画像として表示
                    axes[row_idx, 1].imshow(filters[:, :, 1, row_idx], cmap='gray')
                    axes[row_idx, 1].set_title("G")
                    axes[row_idx, 1].set_ylabel("[:, :, 1, {}]".format(row_idx), labelpad=30, rotation=360)
                    axes[row_idx, 1].set_xticks([])
                    axes[row_idx, 1].set_yticks([])

                    # Bチャネルを白黒画像として表示
                    axes[row_idx, 2].imshow(filters[:, :, 2, row_idx], cmap='gray')
                    axes[row_idx, 2].set_title("B")
                    axes[row_idx, 2].set_ylabel("[:, :, 2, {}]".format(row_idx), labelpad=30, rotation=360)
                    axes[row_idx, 2].set_xticks([])
                    axes[row_idx, 2].set_yticks([])

                    # RGBチャネルをカラー画像として表示
                    axes[row_idx, 3].imshow(filters[:, :, :, row_idx])
                    axes[row_idx, 3].set_title("RGB")
                    axes[row_idx, 3].set_ylabel("[:, :, :, {}]".format(row_idx), labelpad=30, rotation=360)
                    axes[row_idx, 3].set_xticks([])
                    axes[row_idx, 3].set_yticks([])

                # plt.tight_layout()
                plt.show()
                break

            # 白黒画像を入力として受け取る場合
            else:

                # 重みを正規化する
                f_min, f_max = filters.min(), filters.max()
                filters = (filters - f_min) / (f_max - f_min)

                max_rows = 6
                max_cols = 1
                fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols)
                fig.canvas.set_window_title("filter visualization")
                fig.suptitle("layer_idx={}, shape={}".format(layer_idx+1, filters.shape))

                # 各フィルタをプロットする
                for row_idx in range(6):

                    # RGBチャネルをカラー画像として表示
                    axes[row_idx].imshow(filters[:, :, 0, row_idx], cmap='gray')
                    axes[row_idx].set_title("Gray")
                    axes[row_idx].set_ylabel("[:, :, 0, {}]".format(row_idx), labelpad=30, rotation=360)
                    axes[row_idx].set_xticks([])
                    axes[row_idx].set_yticks([])

                # plt.tight_layout()
                plt.show()
                break
