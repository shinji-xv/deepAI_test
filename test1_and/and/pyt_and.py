"""
AND回路の訓練（PyTorch版）
このプログラムは、人工知能（AI）や機械学習の基礎となる「ニューラルネットワーク」を使って、AND回路（論理回路）の動作を学習させるサンプルです。
初心者でも理解しやすいように、各部分に詳しい説明コメントを付けています。
"""
import numpy as np  # 数値計算ライブラリ。行列やベクトルの計算が簡単にできる。
import torch        # PyTorch本体。AIや機械学習のためのライブラリ。
import torch.nn as nn           # ニューラルネットワークの部品が入っている。
import torch.optim as optim     # 学習のための最適化アルゴリズムが入っている。

# ニューラルネットワークのモデル定義
# ここでは「1層のパーセプトロン」を作っています。
class LogicCircuit(nn.Module):
    def __init__(self):
        super().__init__()  # 親クラスの初期化（おまじない）
        # 入力2個、出力2個の全結合層（重みとバイアスを自動で持つ）
        self.fc = nn.Linear(2, 2)
    def forward(self, x):
        # 入力xを全結合層に通し、softmaxで「確率」に変換
        # softmaxは「どちらのクラス（0 or 1）か」を確率で出す関数
        return torch.softmax(self.fc(x), dim=1)

# メイン処理

def main():
    epoch = 1000  # 学習の繰り返し回数（エポック数）。大きいほどたくさん学習する。
    batchsize = 2 # 一度に学習するデータ数（バッチサイズ）。今回は全データを一度に使う。

    # AND回路の入力データ（4パターン）。
    # 例：[0,0]→0, [1,1]→1 など。
    trainx = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
    # AND回路の正解ラベル（[0,0,0,1]）。
    # これは「教師データ」と呼ばれ、正解を教える役割。
    trainy = np.array([0, 0, 0, 1], dtype=np.int64)

    # numpy配列をPyTorchのTensor型に変換
    # TensorはAIでよく使う「多次元配列」
    x = torch.tensor(trainx)
    y = torch.tensor(trainy)

    # モデルのインスタンス化（実体を作る）
    model = LogicCircuit()
    # モデル構造の表示（Kerasのsummaryの代わり）
    print('Model structure:')
    print(model)  # ネットワークの中身（層やパラメータ数）が表示される

    # 損失関数（正解とのズレを測る関数）。分類問題ではCrossEntropyLossが定番。
    criterion = nn.CrossEntropyLoss()
    # 最適化手法（パラメータの更新方法）。Adamはよく使われる手法。
    optimizer = optim.Adam(model.parameters())

    # 学習ループ（ここでAIが「学習」する）
    for ep in range(epoch):
        model.train()  # 学習モードに切り替え（おまじない）
        optimizer.zero_grad()  # 勾配の初期化（前回の情報をリセット）
        outputs = model(x)     # 入力データをモデルに通して予測値を出す
        loss = criterion(outputs, y)  # 予測と正解のズレ（損失）を計算
        loss.backward()  # 誤差逆伝播で勾配計算（AIの「学び」部分）
        optimizer.step() # パラメータ更新（重みを少しずつ修正）
        if (ep+1) % 200 == 0:
            print(f'Epoch {ep+1}, Loss: {loss.item():.4f}')  # 進捗表示

    # 学習済みモデルの保存
    # 学習が終わったら、モデルの「重み」をファイルに保存できる
    torch.save(model.state_dict(), 'and_pytorch.pth')
    print('The model saved.')

    # 学習結果の評価
    print('Evaluation:')
    model.eval()  # 評価モードに切り替え（学習時と区別するため）
    with torch.no_grad():  # 勾配計算を無効化（評価時は不要なので高速化）
        for xx in trainx:
            xx_tensor = torch.tensor(xx, dtype=torch.float32).reshape(1,2)  # 入力データをTensorに
            pred = model(xx_tensor)  # 推論（AIに答えを出させる）
            # argmax()で一番大きい値のインデックス（=予測クラス）を取得
            print(f'input: {xx.tolist()}, prediction: {pred.argmax().item()}')  # 結果表示

if __name__ == '__main__':
    main()


