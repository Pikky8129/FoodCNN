# TensorFlowをインストールする

TensorFlow2をインストールします
https://www.tensorflow.org/install?hl=ja

# Python開発環境の構築

```
$ python -V
Python 3.9.6
$ python -m venv venv
$ python -m pip install --upgrade pip
$ python -m pip install "tensorflow<2.11"
```

# インストールの確認

```
$ python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

# 必要ライブラリのインストール

```
python -m pip install -r requirements.txt
```

# 訓練に使用した画像例

<img src="https://user-images.githubusercontent.com/126436972/222972690-9afa207d-f1ef-4b08-818e-4f0344c7c47e.jpg" width="240">
<img src="https://user-images.githubusercontent.com/126436972/222972693-6514ad3c-d04c-45b8-9939-d6cea3dccfc3.jpg" width="240">
<img src="https://user-images.githubusercontent.com/126436972/222972703-8fd4d6e5-b9ee-4508-9b87-98c93d29b793.jpg" width="240">
<img src="https://user-images.githubusercontent.com/126436972/222972706-7076a37e-996b-439d-a88d-1576176a0605.jpg" width="240">
<img src="https://user-images.githubusercontent.com/126436972/222972709-61380754-6580-4472-a8c7-8964dca473c0.jpg" width="240">

# 推論結果
![hiramen](https://user-images.githubusercontent.com/126436972/222972726-26a8128f-1083-4cfa-83e1-a2246389f4ea.png)
![reimen](https://user-images.githubusercontent.com/126436972/222972732-04000005-f8ce-4e3a-9c42-73ec251ad7eb.png)
![soba](https://user-images.githubusercontent.com/126436972/222972737-c6bb9d87-4d00-41b9-8541-5a95dcd316c0.png)
![somen](https://user-images.githubusercontent.com/126436972/222972742-8b1e147a-0ce0-4c5f-a369-d49b7eb8151c.png)
![udon](https://user-images.githubusercontent.com/126436972/222972743-0a9df122-d439-4169-96c2-c7a9490205a8.png)

