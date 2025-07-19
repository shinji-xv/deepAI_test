"""
【このプログラムの問題点】
- AND回路のような単純な問題に対して層が10層と深すぎるため、学習が困難になりやすい。
- 各層のノード数が2と少なく、情報伝播が制限される。
- 学習データが4パターンしかなく、深いネットワークでは過学習や未学習になりやすい。
- 活性化関数ReLUは初期値によっては全ての出力が0になりやすい。
- シンプルな構造（1～2層）で十分に学習できる問題。

AND回路の学習には、層を減らし、シンプルなネットワーク構造にすることを推奨します。
"""
"""
AND回路の訓練(10層)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ニューラルネットワークのモデル定義
class LogicCircuit(nn.Module):
    def __init__(self):
        super(LogicCircuit, self).__init__()
        # 各層の定義（10層）
        # 入力は2次元、各中間層も2次元、出力も2次元（クラス数2）
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2, 2)
        self.fc4 = nn.Linear(2, 2)
        self.fc5 = nn.Linear(2, 2)
        self.fc6 = nn.Linear(2, 2)
        self.fc7 = nn.Linear(2, 2)
        self.fc8 = nn.Linear(2, 2)
        self.fc9 = nn.Linear(2, 2)
        self.fc10 = nn.Linear(2, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # 出力層で使用

    def forward(self, x):
        # 入力から出力までの流れ（順伝播）
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.relu(self.fc8(x))
        x = self.relu(self.fc9(x))
        x = self.fc10(x)  # 出力層は活性化なし（損失関数でsoftmaxを使うため）
        return x

def main():
    # ハイパーパラメータ
    epoch = 1000  # 学習の繰り返し回数
    batchsize = 2 # バッチサイズ

    # AND回路の入力データとラベル（教師データ）
    trainx = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
    trainy = np.array([0, 0, 0, 1], dtype=np.int64)  # 0:False, 1:True

    # PyTorchのTensor型に変換
    x_tensor = torch.from_numpy(trainx)
    y_tensor = torch.from_numpy(trainy)

    # データセットとデータローダーの作成
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True)

    # モデルのインスタンス化
    model = LogicCircuit()

    # 損失関数（クロスエントロピー）と最適化手法（Adam）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # モデルの構造を表示
    print(model)

    # データ構造の可視化
    print('--- データ構造の可視化 ---')
    print('trainx:', trainx)
    print('trainx.shape:', trainx.shape)
    print('trainy:', trainy)
    print('trainy.shape:', trainy.shape)
    print('x_tensor:', x_tensor)
    print('x_tensor.shape:', x_tensor.shape)
    print('y_tensor:', y_tensor)
    print('y_tensor.shape:', y_tensor.shape)
    print('--- dataloaderの中身（1バッチ分）---')
    for batch_x, batch_y in dataloader:
        print('batch_x:', batch_x)
        print('batch_y:', batch_y)
        break  # 1バッチだけ表示
    print('-----------------------------')

    # 学習ループ
    for ep in range(epoch):
        for batch_x, batch_y in dataloader:
            # 順伝播
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            # 勾配の初期化
            optimizer.zero_grad()
            # 逆伝播
            loss.backward()
            # パラメータ更新
            optimizer.step()
        # 100エポックごとに損失を表示
        if (ep+1) % 100 == 0:
            print(f'Epoch [{ep+1}/{epoch}], Loss: {loss.item():.4f}')

    # モデルの保存
    torch.save(model.state_dict(), 'and_10.pth')
    print('The model saved.')

    # 評価
    print('Evaluation:')
    model.eval()  # 評価モードに切り替え
    with torch.no_grad():  # 勾配計算を無効化
        for x in trainx:
            xx = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # 1サンプル分の形に
            pred = model(xx)
            pred_label = torch.argmax(pred, dim=1).item()
            print(f'input: {x.tolist()}, prediction: {pred_label}')

if __name__ == '__main__':
    main()