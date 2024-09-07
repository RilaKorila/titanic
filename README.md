# タイタニック分析

## データ
- カテゴリ変数が多いため、ヒストグラムのみ用いる
    - Survived / not Survived で色分けしたヒストグラム

## 機能
- データ一覧表示
- データ可視化
    - ヒストグラム

## 2021年からのupdate
- 散布図表示機能の削除
    - 理由：カテゴリ変数が多く、散布図表示が有効ではないため

## セットアップ

pythonの仮想環境を準備し、必要なライブラリをインストール

```
python3 -m venv titanic
source titanic/bin/activate
pip install -r requirements.txt
```

アプリケーションを起動

```
streamlit run main.py
```