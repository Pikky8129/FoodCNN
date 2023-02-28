import numpy as np
from keras.models import Sequential
from matplotlib import cm
from numpy.typing import NDArray
from tf_keras_vis.gradcam import Gradcam, GradcamPlusPlus
from tf_keras_vis.layercam import Layercam
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.scorecam import ScoreCAM
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

from common.app_base import AppBase


class SaliencyMapGenerator(AppBase):
    """
    顕著性マップを生成するクラス    
    """

    def gen_by_smooth_grad(self, model: Sequential, image: NDArray, class_index: int) -> NDArray:
        """
        SmoothGradで顕著性マップを生成する。

        Parameters
        ----------
        model : Sequential
            モデル
        image : NDArray
            入力画像
        class_index : int
            正解の出力カテゴリ

        Returns
        -------
        NDArray
            顕著性マップ
        """

        # 出力層の正解値を指定する関数
        score_function = CategoricalScore([class_index])

        # 出力層の活性化関数を線形活性化関数に置き換える関数
        replace2linear = ReplaceToLinear()

        # 顕著性マップを生成する
        saliency = Saliency(model, model_modifier=replace2linear, clone=True)
        saliency_map = saliency(score_function, image, smooth_samples=20, smooth_noise=0.20)
        saliency_map = np.uint8(cm.jet(saliency_map[0])[..., :3] * 255)

        return saliency_map

    def gen_by_grad_cam(self, model: Sequential, image: NDArray, class_index: int) -> NDArray:
        """
        GradCAMで顕著性マップを生成する。

        Parameters
        ----------
        model : Sequential
            モデル
        image : NDArray
            入力画像
        class_index : int
            正解の出力カテゴリ

        Returns
        -------
        NDArray
            顕著性マップ
        """

        # 出力層の正解値を指定する関数
        score_function = CategoricalScore([class_index])

        # 出力層の活性化関数を線形活性化関数に置き換える関数
        replace2linear = ReplaceToLinear()

        # 顕著性マップを生成する
        gradcam = Gradcam(model, model_modifier=replace2linear, clone=True)
        saliency_map = gradcam(score_function, image, penultimate_layer=-1)
        saliency_map = np.uint8(cm.jet(saliency_map[0])[..., :3] * 255)

        return saliency_map

    def gen_by_grad_cam_pp(self, model: Sequential, image: NDArray, class_index: int) -> NDArray:
        """
        GradCAM++で顕著性マップを生成する。

        Parameters
        ----------
        model : Sequential
            モデル
        image : NDArray
            入力画像
        class_index : int
            正解の出力カテゴリ

        Returns
        -------
        NDArray
            顕著性マップ
        """

        # 出力層の正解値を指定する関数
        score_function = CategoricalScore([class_index])

        # 出力層の活性化関数を線形活性化関数に置き換える関数
        replace2linear = ReplaceToLinear()

        # 顕著性マップを生成する
        gradcam = GradcamPlusPlus(model, model_modifier=replace2linear, clone=True)
        saliency_map = gradcam(score_function, image, penultimate_layer=-1)
        saliency_map = np.uint8(cm.jet(saliency_map[0])[..., :3] * 255)

        return saliency_map

    def gen_by_score_cam(self, model: Sequential, image: NDArray, class_index: int) -> NDArray:
        """
        ScoreCAMで顕著性マップを生成する。

        Parameters
        ----------
        model : Sequential
            モデル
        image : NDArray
            入力画像
        class_index : int
            正解の出力カテゴリ

        Returns
        -------
        NDArray
            顕著性マップ
        """

        # 出力層の正解値を指定する関数
        score_function = CategoricalScore([class_index])

        # 出力層の活性化関数を線形活性化関数に置き換える関数
        replace2linear = ReplaceToLinear()

        # 顕著性マップを生成する
        scorecam = ScoreCAM(model)
        saliency_map = scorecam(score_function, image, penultimate_layer=-1)
        saliency_map = np.uint8(cm.jet(saliency_map[0])[..., :3] * 255)

        return saliency_map

    def gen_by_layer_cam(self, model: Sequential, image: NDArray, class_index: int) -> NDArray:
        """
        LayerCAMで顕著性マップを生成する。

        Parameters
        ----------
        model : Sequential
            モデル
        image : NDArray
            入力画像
        class_index : int
            正解の出力カテゴリ

        Returns
        -------
        NDArray
            顕著性マップ
        """

        # 出力層の正解値を指定する関数
        score_function = CategoricalScore([class_index])

        # 出力層の活性化関数を線形活性化関数に置き換える関数
        replace2linear = ReplaceToLinear()

        # 顕著性マップを生成する
        layercam = Layercam(model, model_modifier=replace2linear, clone=True)
        saliency_map = layercam(score_function, image, penultimate_layer=-1)
        saliency_map = np.uint8(cm.jet(saliency_map[0])[..., :3] * 255)

        return saliency_map
