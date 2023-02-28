import matplotlib.pyplot as plt

from common.app_base import AppBase
from image_classification.image_classifier import ImageClassifier
from image_classification.saliency_map_generator import SaliencyMapGenerator


class SaliencyMapView(AppBase):
    """
    顕著性マップを表示するクラス    
    """

    def show(self, model_h5_path: str, image_path: str, class_index: int, overlay: bool = True) -> None:
        """
        顕著性マップを表示する

        Parameters
        ----------
        model_h5_path : str
            モデルのファイルパス
        image_path : str
            入力画像のファイルパス
        class_index : int
            正解の出力カテゴリ
        overlay : bool, optional
            顕著性マップを入力画像に重ねて表示する
        """

        # モデルを読み込む
        classifier = ImageClassifier()
        model = classifier.load_model(model_h5_path)

        # 画像を読み込む
        height = model.input_shape[1]
        width = model.input_shape[2]
        channel = model.input_shape[3]
        image = classifier.load_and_preprocess_image(image_path, width, height, channel)

        # 顕著性マップを生成する
        sm_gen = SaliencyMapGenerator()
        sm_smoothgrad = sm_gen.gen_by_smooth_grad(model, image, class_index)
        sm_gradcam = sm_gen.gen_by_grad_cam(model, image, class_index)
        sm_gradcam_pp = sm_gen.gen_by_grad_cam_pp(model, image, class_index)
        sm_scorecam = sm_gen.gen_by_score_cam(model, image, class_index)
        sm_layercam = sm_gen.gen_by_layer_cam(model, image, class_index)

        # 顕著性マップを格納する
        cams = {
            "SmoothGrad": sm_smoothgrad,
            "GradCAM": sm_gradcam,
            "GradCAM++": sm_gradcam_pp,
            "ScoreCAM": sm_scorecam,
            "LayerCAM": sm_layercam
        }

        # 新しいウインドウを作成する
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.canvas.set_window_title("SaliencyMapView")
        axes = axes.ravel()

        # サブプロットに顕著性マップを表示する
        for i, (cam_name, cam) in enumerate(cams.items()):
            axes[i].set_title(cam_name)
            if overlay:
                axes[i].imshow(image)
                axes[i].imshow(cam, alpha=0.5)
            else:
                axes[i].imshow(cam)
            axes[i].axis('off')

        # 画像がないサブプロットを非表示にする
        for ax in axes:
            if len(ax.get_images()) == 0:
                ax.set_visible(False)

        # ウインドウを表示する
        plt.tight_layout()
        plt.show()
