import math

import matplotlib.pyplot as plt
import numpy as np
from cv2 import Mat
from keras.models import Model, Sequential

from common.app_base import AppBase
from common.image_editor import ImageEditor


class FeatureMapVisualizationView(AppBase):
    """
    畳み込み層の特徴マップを可視化する画面
    """

    def __init__(self):
        super().__init__()

    def show(
        self,
        model: Sequential,
        mat_bgr: Mat
    ) -> None:
        """
        特徴マップを可視化する

        Parameters
        ----------
        model: Sequential
            学習済みのモデル
        mat_bgr : Mat
            Mat形式の画像
        """

        for layer_idx, layer in enumerate(model.layers):

            if "conv" not in layer.name:
                continue

            # 可視化する特徴マップの畳み込み層を出力層として再定義する
            print(model.summary())
            new_model = Model(
                inputs=model.inputs,
                outputs=model.layers[layer_idx].output
            )
            print(new_model.summary())

            # 入力層の形状(画像の高さ/横幅/チャンネル)
            input_image_height = new_model.input_shape[1]
            input_image_width = new_model.input_shape[2]
            input_image_channel = new_model.input_shape[3]

            # 入力画像を入力形状に変換する
            eval_image = None
            if input_image_channel == 1:
                eval_image = ImageEditor.bgr_to_gray(mat_bgr)
            elif input_image_channel == 3:
                eval_image = ImageEditor.bgr_to_rgb(mat_bgr)
            eval_image = ImageEditor.resize(eval_image, input_image_width, input_image_height)
            eval_image = ImageEditor.mat_to_3d_array(eval_image)
            eval_image = ImageEditor.normalize(eval_image)

            # 特徴マップを取得する
            feature_maps = new_model.predict(np.array([eval_image]))

            # 特徴マップのチャネル数
            feature_map_channels = feature_maps.shape[3]

            square = int(math.sqrt(feature_map_channels) + 1)
            square = 12 if square > 12 else square
            fig, axes = plt.subplots(nrows=square, ncols=square)
            axes = axes.ravel()
            fig.canvas.set_window_title("Feature Map Visualization")
            fig.suptitle(
                "layer_idx={}, feature_map_channels={}".format(layer_idx, feature_map_channels)
            )

            for i in range(square*square):
                if i < feature_map_channels:
                    axes[i].imshow(feature_maps[0, :, :, i], cmap="gray")
                    axes[i].set_title("[0, :, :, {}]".format(i))
                    axes[i].set_xticks([])
                    axes[i].set_yticks([])
                else:
                    axes[i].set_visible(False)

        # plt.tight_layout()
        plt.show()
