docker compose exec pytorch bash  「起動中の pytorch コンテナの中で bash を起動して、中に入る」
### Windows + Docker Desktop で Docker コンテナから GPU を使う手順（完全版）

## 0. 目的

Windows + Docker Desktop 環境で、Docker コンテナから GPU（例: RTX 3060 Ti）を使えるようにする手順を、コマンド付きでまとめます。

## 1. 前提（事前準備）

- Docker Desktop がインストール済みであること
- WSL2 backend が有効化されていること
- GPU 対応 NVIDIA Driver（WSL 対応版）がインストールされていること

※ 上記は GUI 作業が中心のため、ここではコマンドはありません。

## 2. WSL に Ubuntu を作成する

1. 現在の WSL 状態を確認（Windows PowerShell）:

```powershell
wsl -l -v
```

2. Ubuntu が無ければインストール:

```powershell
wsl --install -d Ubuntu
```

3. GPU 連携のために Windows を再起動（重要）:

```powershell
shutdown /r /t 0
```

4. Ubuntu に入る:

```powershell
wsl -d Ubuntu
```

プロンプト例:

```
shunya@DESKTOP-XXXX:~$
```

## 3. Ubuntu 側の基本セットアップ

Ubuntu 内で以下を実行します:

```bash
sudo apt update
sudo apt install -y curl ca-certificates gnupg lsb-release
```

## 4. NVIDIA Container Toolkit の追加（Ubuntu 上）

1. GPG キーを追加:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
	| sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
```

2. NVIDIA 公式リポジトリを登録:

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
	| sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
	| sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

3. パッケージを更新してインストール:

```bash
sudo apt update
sudo apt install -y nvidia-container-toolkit
```

4. Docker を再起動（Ubuntu 側から）:

```bash
sudo systemctl restart docker
```

※ Docker Engine 自体は Docker Desktop のものです。Ubuntu は設定と操作を行います。

## 5. Docker が GPU を使えるか確認（Windows 側）

Windows の PowerShell で以下を実行して確認します:

```powershell
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

初回はイメージが自動で pull されます。出力例:

```
Unable to find image locally
Pulling from nvidia/cuda
```

成功例（重要）:

```
NVIDIA-SMI 560.35.02
Driver Version: 560.94
CUDA Version: 12.6
GPU: NVIDIA GeForce RTX 3060 Ti
```

これらが確認できれば:

- Docker コンテナから GPU が見える
- CUDA が有効
- WSL + Docker Desktop + GPU 接続成功

## 6. コンテナ内で PyTorch スクリプトを実行する例

コンテナを起動し、内部で Python スクリプトを実行する例:

```powershell
docker compose up -d
docker compose exec pytorch bash
# コンテナ内で
python pytorch_liner1.py
```

以上で手順は完了です。問題が出た場合は、エラーメッセージを共有してください。

---

ファイル: gpu適応docker.md を Markdown 形式で整形しました。