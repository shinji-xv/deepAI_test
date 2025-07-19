"""
このファイルは、他のファイルで生成したPyTorchのチェックポイント（中間保存ファイル）を読み込み、
その続きから学習を再開できるサンプルです。
fit関数のコールバックを使って学習過程のモデルの重みをチェックポイントとして保存
"""

# 必要なライブラリをインポート
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# PyTorchでのモデル定義
class LogicCircuit(nn.Module):
    def __init__(self):
        super(LogicCircuit, self).__init__()
        # 入力2次元、出力2次元の全結合層
        self.fc = nn.Linear(2, 2)
        # He初期化（Kaiming初期化）
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
    def forward(self, x):
        # 出力層（softmaxは使わずlogitsを返す）
        # 理由：PyTorchのnn.CrossEntropyLossは内部でsoftmaxを自動的に適用するため、
        # モデルの出力はsoftmax前（logits）のままにする必要がある。
        x = self.fc(x)
        return x


def main():
    # 学習用パラメータ
    epoch = 200  # 学習回数
    batchsize = 4 # バッチサイズ

    # AND回路の訓練データ作成
    # 入力データ（2入力）
    trainx = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
    # 出力ラベル（ANDの結果）
    trainy = np.array([0, 0, 0, 1], dtype=np.int64)

    # データセットとデータローダーの作成
    # TensorDatasetでデータとラベルをまとめる
    dataset = TensorDataset(torch.from_numpy(trainx), torch.from_numpy(trainy))
    # DataLoaderでバッチ処理を簡単に
    train_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True)


    # モデルのインスタンス化
    model = LogicCircuit()
    # 損失関数（クロスエントロピー）
    criterion = nn.CrossEntropyLoss()
    # 最適化手法（Adam）
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # チェックポイント保存用ディレクトリ
    ckpt_dir = 'log'
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- ここからチェックポイントのロード処理を追加 ---
    # 再開したいチェックポイントファイル名（例：and-0100.pth など）
    resume_ckpt = os.path.join(ckpt_dir, 'and-0100.pth')
    start_epoch = 1
    if os.path.exists(resume_ckpt):
        # チェックポイントが存在すればロード
        model.load_state_dict(torch.load(resume_ckpt))
        print(f'Checkpoint {resume_ckpt} loaded. Continue training from this state.')
        # 例として100エポックごとに保存している場合、次のエポックから再開
        start_epoch = 101
    else:
        print('No checkpoint found. Training from scratch.')
    # --- ここまで ---

    # 学習ループ
    for ep in range(start_epoch, epoch+1):
        model.train()  # 学習モード
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            # 勾配初期化
            optimizer.zero_grad()
            # 順伝播
            outputs = model(batch_x)
            # 損失計算
            loss = criterion(outputs, batch_y)
            # 逆伝播
            loss.backward()
            # パラメータ更新
            optimizer.step()
            running_loss += loss.item()
        # 50エポックごとにモデル保存
        if ep % 50 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'and-{ep:04d}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f'Epoch {ep}: checkpoint saved to {ckpt_path}')
        # 進捗表示
        if ep % 50 == 0 or ep == 1:
            print(f'Epoch {ep}, Loss: {running_loss/len(train_loader):.4f}')

    # 最終モデル保存
    torch.save(model.state_dict(), 'and_3.pth')
    print('The model saved.')

    # 評価
    print('Evaluation:')
    model.eval()  # 評価モード
    with torch.no_grad():
        for x in trainx:
            xx = torch.tensor(x, dtype=torch.float32).reshape(1, 2)
            pred = model(xx)
            pred_label = pred.argmax(dim=1).item()
            print(f'input: {x.tolist()}, prediction: {pred_label}')

# スクリプトが直接実行された場合のみmain()を呼ぶ
if __name__ == '__main__':
    main()