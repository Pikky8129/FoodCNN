import codecs
import json
import pickle
import random
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from cv2 import Mat
from keras import layers, models, optimizers
from keras.applications import VGG16, MobileNet
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import array_to_img, img_to_array, load_img
from numpy.typing import NDArray
from sklearn import model_selection

from common.app_base import AppBase
from common.directory import Directory
from common.image_editor import ImageEditor


class ImageClassifier(AppBase):
    """
    画像分類器
    モデルの構築・学習、そして推測を実行する。
    """

    def __init__(
        self,
        use_cpu: bool = False,
        gpu_memory_growth: bool = False
    ):
        """
        Parameters
        ----------
        use_cpu : bool, optional
            CPUを利用する場合に指定する
        gpu_memory_growth : bool, optional
            GPUメモリを必要に応じて確保する場合に指定する
        """
        super().__init__()

        if use_cpu:
            # CPUを利用する
            tf.config.set_visible_devices([], 'GPU')
        else:
            if gpu_memory_growth:
                # GPUメモリを必要に応じて確保する
                gpus = tf.config.list_physical_devices('GPU')
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

    def build_model(
        self,
        input_image_width: int,
        input_image_height: int,
        input_image_channel: int,
        output_dimensions: int
    ) -> Sequential:
        """
        モデルを構築する

        Parameters
        ----------
        input_image_width : int
            入力層の形状(画像の横幅)
        input_image_height : int
            入力層の形状(画像の高さ)
        input_image_channel : int
            入力層の形状(画像のチャンネル)
            モノクロで学習する場合「1」、カラーで学習する場合「3」
        output_dimensions : int
            出力層の次元数(画像のカテゴリ数)

        Returns
        -------
        Sequential
            モデル
        """

        # モデルを定義する
        model = Sequential()

        # レイヤーを定義する
        model.add(layers.Conv2D(64, (4, 4), padding='same', activation='relu',
                                input_shape=(input_image_height, input_image_width, input_image_channel)))
        model.add(layers.Conv2D(64, (4, 4), padding='same', activation='relu'))
        model.add(layers.Conv2D(64, (4, 4), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(4, 4)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(output_dimensions, activation='softmax'))

        # モデルをコンパイルする
        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        # モデルのレイヤーを出力する
        self._logger.info(model.summary())

        return model

    def save_model(self, model: Sequential, model_h5_path: str):
        """
        モデルを保存する
        ※モデルには、アーキテクチャ、重み値、およびcompile()情報が含まれる。
        ※モデルには、カテゴリ名が含まれないため、別途保存する必要がある。

        Parameters
        ----------
        model : Sequential
            モデル
        model_h5_path : str
            モデルのファイルパス(Keras H5形式, model.h5)
        """

        # モデルを保存する
        model.save(model_h5_path)

        # モデルのグラフ構造を画像で保存する
        # from keras.utils import plot_model
        # plot_model(model, to_file="./assets/model.png")

    def load_model(self, model_h5_path: str) -> Sequential:
        """
        モデルを読み込む

        Parameters
        ----------
        model_h5_path : str
            モデルのファイルパス(Keras H5形式, model.h5)

        Returns
        -------
        Sequential
            モデル
        """

        # モデルを読み込む
        model = models.load_model(model_h5_path)

        # モデルのレイヤーを出力する
        self._logger.info(model.summary())

        return model

    def save_class_names(self, class_names: List[str], class_names_json_path: str):
        """
        カテゴリ名(class_names.json)を保存する

        Parameters
        ----------
        class_names : List[str]
            カテゴリ名の一覧
        class_names_json_path : str
            カテゴリ名のファイルパス
        """

        # カテゴリ名を保存する
        with codecs.open(class_names_json_path, "w", "utf8") as f:
            json.dump(class_names, f, ensure_ascii=False)

    def load_class_names(self, class_names_json_path: str) -> List[str]:
        """
        カテゴリ名(class_names.json)を読み込む

        Parameters
        ----------
        class_names_json_path : str
            カテゴリ名のファイルパス

        Returns
        -------
        List[str]
            カテゴリ名の一覧
        """
        # カテゴリ名を読み込む
        with codecs.open(class_names_json_path, "r", "utf8") as f:
            return json.load(f)

    def save_weight(self, model: Sequential, weight_h5_path: str):
        """
        モデルの重みを保存する
        ※Keras H5形式のモデルは重みを含むため、通常は利用しなくてよい。

        Parameters
        ----------
        model : Sequential
            モデル
        weight_h5_path : str
            重みのファイルパス
        """
        # モデルの重みを保存する
        model.save_weights(weight_h5_path)

    def load_weight(self, model: Sequential, weight_h5_path: str):
        """
        モデルの重みを読み込む
        ※Keras H5形式のモデルは重みを含むため、通常は利用しなくてよい。

        Parameters
        ----------
        model : Sequential
            モデル
        weight_h5_path : str
            重みのファイルパス
        """
        # モデルの重みを読み込む
        model.load_weights(weight_h5_path)

    def build_dataset(
        self,
        dataset_dir_path: str,
        input_image_width: int,
        input_image_height: int,
        input_image_channel: int,
        max_samples: int = 1000
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        データセットを構築する

        Parameters
        ----------
        dataset_dir_path : str
            データセットのディレクトリパス
        input_image_width : int
            入力層の形状(画像の横幅)
        input_image_height : int
            入力層の形状(画像の高さ)
        input_image_channel : int
            入力層の形状(画像のチャンネル)
            モノクロで学習する場合「1」、カラーで学習する場合「3」
        max_samples : int
            訓練データの最大サンプル数。
            訓練データがメモリに載りきらない場合に最大サンプル数に抑制する。
            利用するサンプルはランダムに選択する。

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray, NDArray]
            データセット[訓練用データ, 検証用データ, 訓練用ラベル, 検証用ラベル]
        """

        class_names = Directory.get_directory_names(dataset_dir_path)

        # データセット(データとラベルのペア)を構築する
        # データ(x)には、0.0-1.0の範囲で正規化した画像データを格納する。
        # ラベル(y)には、OneHot形式[0, 1, 0]で表現したラベルを格納する。

        x = []  # データ
        y = []  # ラベル
        for idx, class_name in enumerate(class_names):
            # ラベル
            label = [0 for i in range(len(class_names))]
            label[idx] = 1
            # label = idx
            # データ
            search_pattern = "{dataset_dir_path}/{class_name}/*.jpg".format(
                dataset_dir_path=dataset_dir_path, class_name=class_name
            )
            file_paths = Directory.get_files(search_pattern)

            # 訓練データが最大サンプル数を超える場合、ランダムに選択する
            if len(file_paths) > max_samples:
                for i in range(len(file_paths)-max_samples):
                    index = random.randint(0, len(file_paths) - 1)
                    file_paths.pop(index)

            for file_path in file_paths:

                # Q. 画像はPillow形式で扱うべきか？OpenCV形式で扱うべきか？
                # A. リサイズ時の補完アルゴリズムに違いがあり、出力結果は視覚的に似ている場合も、
                #    推測結果に大きな違いを引き起こす可能性があるのでPillow推奨らしい。
                #    (つまり異なる画像処理エンジンを混ぜなければいいだけでは？)
                #
                # OpenCV(Mat)形式とPillow(PIL)形式の違い
                # ・KerasはPillow形式を基本としている。
                # ・PillowはRGB、OpenCVはBGR。
                # ・PillowとOpenCVでリサイズアルゴリズムに違いがある。
                # ・Pillowのカラー画像は(h,w,3)の形式、OpenCVは(h,w,3)の形式
                # ・Pillowのモノクロ画像は(h,w,1)の形式、OpenCVは(h,w)の形式

                # Pillow(PIL)で画像を読み込む
                # color_mode = "grayscale" if input_image_channel == 1 else "rgb"
                # image = load_img(
                #     file_path,
                #     color_mode=color_mode,
                #     target_size=(input_image_height, input_image_width)  # height, widthの順で正しい
                # )
                # image = img_to_array(image)
                # image = image.astype('float32') / 255.0
                # x.append(image)
                # y.append(label)

                # OpenCV(MAT)で画像を読み込む
                as_gray = True if input_image_channel == 1 else False
                image = ImageEditor.load_as_mat(file_path, as_gray)
                if input_image_channel == 3:
                    image = ImageEditor.bgr_to_rgb(image)
                image = ImageEditor.resize(image, input_image_width, input_image_height)
                image = ImageEditor.mat_to_3d_array(image)
                image = ImageEditor.normalize(image)
                x.append(image)
                y.append(label)

        x = np.array(x)
        y = np.array(y)

        # x = x.astype('float32') / 255.0
        # from keras.utils import to_categorical
        # y = to_categorical(y, len(class_names))

        # データセットを訓練用と検証用に分ける
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y)

        return (x_train, x_test, y_train, y_test)

    def save_dataset(self, dataset: Tuple[NDArray, NDArray, NDArray, NDArray], dataset_npy_path: str):
        """
        データセット(dataset.npy)を保存する

        Parameters
        ----------
        dataset : Tuple[NDArray, NDArray, NDArray, NDArray]
            データセット[訓練用データ, 検証用データ, 訓練用ラベル, 検証用ラベル]
        dataset_npy_path : str
            データセットのファイルパス
        """
        # データセットを保存する
        # np.save(dataset_npy_path, dataset)
        with open(dataset_npy_path, 'wb') as f:
            pickle.dump(dataset, f)

    def load_dataset(self, dataset_npy_path: str) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        データセット(dataset.npy)を読み込む

        Parameters
        ----------
        dataset_npy_path : str
            データセットのファイルパス

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray, NDArray]
            データセット[訓練用データ, 検証用データ, 訓練用ラベル, 検証用ラベル]
        """
        # データセットを読み込む
        # x_train, x_test, y_train, y_test = np.load(dataset_npy_path)
        # return (x_train, x_test, y_train, y_test)
        with open(dataset_npy_path, 'rb') as f:
            return pickle.load(f)

    def train(
        self,
        model: Sequential,
        dataset: Tuple[NDArray, NDArray, NDArray, NDArray],
        batch_size=10,
        epochs=5
    ) -> Dict[str, list]:
        """
        モデルを学習する

        Parameters
        ----------
        model : Sequential
            モデル
        dataset : Tuple[NDArray, NDArray, NDArray, NDArray]
            データセット[訓練用データ, 検証用データ, 訓練用ラベル, 検証用ラベル]
        batch_size : int, optional
            バッチサイズ
        epochs : int, optional
            エポック

        Returns
        -------
        Dict[str, list]
            モデルの学習履歴
        """

        # データセットを読み込む
        x_train, x_val, y_train, y_val = dataset

        # モデルを学習する
        start_time = time.time()
        history = model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_val, y_val)
        )
        finish_time = time.time()

        # モデルの学習結果を表示する
        self._logger.info("Loss: {} (損失関数値 - 0に近いほど正解に近い)".format(history.history["loss"][-1]))
        self._logger.info("Accuracy: {}% (精度 - 100% に近いほど正解に近い)".format(history.history["accuracy"][-1] * 100))
        self._logger.info("Computation time: {0:.3f} sec (計算時間)".format(finish_time - start_time))

        return history.history

    def train_with_generator(
        self,
        model: Sequential,
        dataset: Tuple[NDArray, NDArray, NDArray, NDArray],
        batch_size=10,
        epochs=5
    ) -> Dict[str, list]:
        """
        モデルを学習する

        Parameters
        ----------
        model : Sequential
            モデル
        dataset : Tuple[NDArray, NDArray, NDArray, NDArray]
            データセット[訓練用データ, 検証用データ, 訓練用ラベル, 検証用ラベル]
        batch_size : int, optional
            バッチサイズ
        epochs : int, optional
            エポック

        Returns
        -------
        Dict[str, list]
            モデルの学習履歴
        """

        # データセットを読み込む
        x_train, x_val, y_train, y_val = dataset

        # 訓練用データのジェネレータ
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest"
        )
        train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)

        # 検証用データのジェネレータ
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow(x_val, y_val, batch_size=batch_size)

        # データ拡張で得られたデータを表示する
        row = 4
        col = 4
        plot_num = 1
        plt.figure("show picture that created by data arguementation.")
        for data_batch, label_batch in train_generator:
            plt.subplot(row, col, plot_num)
            plt.subplots_adjust(wspace=0.4, hspace=0.4)
            plt.tick_params(labelbottom="off")  # x軸の削除
            plt.tick_params(labelleft="off")    # y軸の削除
            # plt.title(self.categories[label_batch.tolist()[0].index(max(label_batch.tolist()[0]))])
            plt.imshow(array_to_img(data_batch[0]))
            if plot_num == row * col:
                break
            plot_num += 1
        plt.show()

        start_time = time.time()

        # モデルを学習する
        history = model.fit(
            train_generator,
            steps_per_epoch=100,
            epochs=epochs,
            validation_data=test_generator,
            validation_steps=50,
            max_queue_size=300
        )

        # モデルの学習結果を表示する
        self._logger.info("Loss: {} (損失関数値 - 0に近いほど正解に近い)".format(history.history["loss"][-1]))
        self._logger.info("Accuracy: {}% (精度 - 100% に近いほど正解に近い)".format(history.history["accuracy"][-1] * 100))
        self._logger.info("Computation time: {0:.3f} sec (計算時間)".format(time.time() - start_time))

        return history.history

    def evaluate(
        self,
        model: Sequential,
        dataset: Tuple[NDArray, NDArray],
    ) -> List[float]:
        """
        モデルを評価する

        Parameters
        ----------
        model : Sequential
            モデル
        dataset : Tuple[NDArray, NDArray]
            データセット[テスト用データ, テスト用ラベル]

        Returns
        -------
        List[float]
            モデルの評価
        """

        x_test,  y_test = dataset

        start_time = time.time()

        # モデルを評価する
        score = model.evaluate(x_test, y_test)

        # モデルの評価を表示する
        self._logger.info("Loss: {} (損失関数値 - 0に近いほど正解に近い)".format(score[0]))
        self._logger.info("Accuracy: {}% (精度 - 100% に近いほど正解に近い)".format(score[1] * 100))
        self._logger.info("Computation time: {0:.3f} sec (計算時間)".format(time.time() - start_time))

        return score

    def predict(
        self,
        model: Sequential,
        class_names: List[str],
        mat_bgr: Mat
    ) -> Tuple[str, int]:
        """
        学習済みのモデルを利用し、画像のカテゴリを推測する。

        Parameters
        ----------
        model: Sequential
            学習済みのモデル
        class_names : List[str]
            カテゴリ名の一覧
        mat_bgr : Mat
            Mat形式の画像

        Returns
        -------
        Tuple[str, int]
            (予測カテゴリ名, 予測度0~100, すべての予測結果)
        """

        # 入力層の形状(画像の高さ/横幅/チャンネル)
        input_image_height = model.input_shape[1]
        input_image_width = model.input_shape[2]
        input_image_channel = model.input_shape[3]

        # 出力層の次元数(画像のカテゴリ数)
        output_dimensions = model.layers[-1].units

        # 入力画像を入力形状に変換する
        eval_image = None
        if input_image_channel == 1:
            eval_image = ImageEditor.bgr_to_gray(mat_bgr)
        elif input_image_channel == 3:
            eval_image = ImageEditor.bgr_to_rgb(mat_bgr)
        eval_image = ImageEditor.resize(eval_image, input_image_width, input_image_height)
        eval_image = ImageEditor.mat_to_3d_array(eval_image)
        eval_image = ImageEditor.normalize(eval_image)

        # 画像のカテゴリを推測する
        start_time = time.time()
        predictions = model.predict(np.array([eval_image]))
        finish_time = time.time()
        prediction = predictions[0]

        # 推測結果を表示する
        self._logger.info("{}: {}%, time: {:.3f}ms".format(
            class_names[prediction.argmax()],
            prediction[prediction.argmax()],
            ((finish_time - start_time) * 1000)
        ))

        return (
            class_names[prediction.argmax()],
            int(prediction[prediction.argmax()] * 100),
            prediction
        )

    def load_and_preprocess_image(self, path: str, width: int, height: int, channel: int) -> NDArray:
        """
        画像を読み込んで前処理を行う。

        Parameters
        ----------
        path : str
            画像ファイルのパス
        width : int
            入力層の形状(画像の横幅)
        height : int
            入力層の形状(画像の高さ)
        channel : int
            入力層の形状(画像のチャンネル)

        Returns
        -------
        NDArray
            前処理済み画像
        """

        # 画像を読み込む
        image = load_img(
            path,
            color_mode="rgb" if channel == 3 else "grayscale",
            target_size=(height, width)
        )
        # 画像を配列に変換する
        image = img_to_array(image)
        # 画像を正規化する
        image = image.astype('float32') / 255.0
        return image
