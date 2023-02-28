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

