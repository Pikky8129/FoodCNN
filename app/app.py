from common.directory import Directory
from common.image_editor import ImageEditor
from image_classification.image_classifier import ImageClassifier
from views.feature_map_visualization_view import FeatureMapVisualizationView
from views.filter_visualization_view import FilterVisualizationView
from views.prediction_result_view import PredictionResultView
from views.saliency_map_view import SaliencyMapView
from views.train_history_view import TrainHistoryView


class FoodRecognitionUsecase:

    @staticmethod
    def build_and_train_model():
        """
        料理画像のカテゴリを学習する
        """

        # モデルの配置場所
        dataset_dir_path = "./assets/dataset_food_448"
        model_h5_path = "./assets/keras_cnn_food_224_224_color_model.h5"
        class_names_json_path = "./assets/keras_cnn_food_224_224_color_class_names.json"

        # 入力層の形状(画像の横幅/高さ/チャンネル)
        input_image_width: int = 224        # 画像を指定した形状に変換して学習する
        input_image_height: int = 224       # 画像を指定した形状に変換して学習する
        input_image_channel: int = 3        # モノクロで学習する場合「1」、カラーで学習する場合「3」

        # 画像分類器
        classifier = ImageClassifier()

        # カテゴリ名
        class_names = Directory.get_directory_names(dataset_dir_path)

        # 出力層の次元数(画像のカテゴリ数)
        output_dimensions = len(class_names)

        # モデルを構築する
        model = classifier.build_model(
            input_image_width,
            input_image_height,
            input_image_channel,
            output_dimensions,
        )

        # データセットを構築する
        dataset = classifier.build_dataset(
            dataset_dir_path,
            input_image_width,
            input_image_height,
            input_image_channel,
            max_samples=1500
        )

        # モデルを学習する
        history = classifier.train(model, dataset, 10, 5)

        # モデルを評価する
        # classifier.evaluate(model, (dataset[1], dataset[3]))

        # 学習履歴を表示する
        th_view = TrainHistoryView()
        th_view.show(history)

        # モデルを保存する
        classifier.save_model(model, model_h5_path)

        # カテゴリ名を保存する
        classifier.save_class_names(class_names, class_names_json_path)

    @staticmethod
    def predict_image_category():
        """
        料理画像のカテゴリを推測する
        """

        # モデルの配置場所
        model_h5_path = "./assets/keras_cnn_food_224_224_color_model.h5"
        class_names_json_path = "./assets/keras_cnn_food_224_224_color_class_names.json"

        # 入力画像のファイルパス
        file_paths = Directory.get_files("./assets/input_food/*.png")

        # 画像分類器
        classifier = ImageClassifier()
        model = classifier.load_model(model_h5_path)
        class_names = classifier.load_class_names(class_names_json_path)

        for file_path in file_paths:

            # 入力画像を読み込む
            mat_gbr = ImageEditor.load_as_mat(file_path)

            # カテゴリを推測する
            pre_category, pre_accuracy, prediction = classifier.predict(model, class_names, mat_gbr)

            for idx, accuracy in enumerate(prediction):
                print("{}の確率: {}%".format(class_names[idx], int(accuracy * 100)))
            pr_view = PredictionResultView()
            pr_view.show(mat_gbr, prediction, class_names)

    @staticmethod
    def visualize_filter():
        """
        フィルタを可視化する
        """

        # モデルの配置場所
        model_h5_path = "./assets/keras_cnn_food_224_224_color_model.h5"

        # 画像分類器
        classifier = ImageClassifier()
        model = classifier.load_model(model_h5_path)

        # フィルタを可視化する
        view = FilterVisualizationView()
        view.show_first_conv_layer(model)
        view.show_all_conv_layers(model)

    @staticmethod
    def visualize_feature_map():
        """
        特徴マップを可視化する
        """

        # モデルの配置場所
        model_h5_path = "./assets/keras_cnn_food_224_224_color_model.h5"

        # 画像分類器
        classifier = ImageClassifier()
        model = classifier.load_model(model_h5_path)

        # 入力画像のファイルパス
        file_paths = Directory.get_files("./assets/input_food/*.png")

        for file_path in file_paths:

            # 入力画像を読み込む
            mat_gbr = ImageEditor.load_as_mat(file_path)

            # 特徴マップを可視化する
            view = FeatureMapVisualizationView()
            view.show(model, mat_gbr)

    @staticmethod
    def visualize_saliency_map():
        """
        顕著性マップを可視化する
        """

        # モデルの配置場所
        model_h5_path = "./assets/keras_cnn_food_224_224_color_model.h5"

        # 入力画像のファイルパスとクラス
        image_path = "./assets/input_food/IMG_1597.png"
        class_index = 4

        # 顕著性マップを可視化する
        view = SaliencyMapView()
        view.show(model_h5_path, image_path, class_index, False)


if __name__ == "__main__":

    # 1-1. データセットでモデルを訓練し、画像のカテゴリを覚えさせる
    FoodRecognitionUsecase.build_and_train_model()

    # 1-2. モデルに画像を入力し、画像のカテゴリを推論する。
    FoodRecognitionUsecase.predict_image_category()

    # # フィルタを可視化する
    # FoodRecognitionUsecase.visualize_filter()

    # # 特徴マップを可視化する
    # FoodRecognitionUsecase.visualize_feature_map()

    # # 顕著性マップを可視化する
    # FoodRecognitionUsecase.visualize_saliency_map()