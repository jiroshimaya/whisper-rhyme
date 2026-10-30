# Whisperくん、韻を踏みなさい — 再現実験

Qiita記事「[Whisperくん、韻を踏みなさい](https://qiita.com/shimajiroxyz/items/9653c39599cb2eb59695)」の実験コードです。曖昧な合成音声や短い明瞭語、decoderの直前文脈で、Whisper large-v3の出力の語尾を誘導できるか調べます。今回の実験では、意味が通る押韻フレーズの安定した生成には至りませんでした。

ビートと無意味なCV（子音＋母音）はローカル合成し、「朝・花・山・坂」はVOICEVOX:四国めたん（ノーマル）で合成します。外部の楽曲や既存歌詞は使いません。`transcribe()`の無音判定による区間のスキップを通さず、`model.decode()`の全出力を保存します。

## 記事の再現

必要な環境はPython 3.11、[uv](https://docs.astral.sh/uv/)、FFmpeg、ローカルで起動した[VOICEVOX ENGINE](https://github.com/VOICEVOX/voicevox_engine)です。推論はCPU・float32で行います。依存パッケージとWhisperモデルの初回ダウンロードにはネットワークと空き容量が必要です。

```bash
git clone https://github.com/jiroshimaya/whisper-rhyme.git
cd whisper-rhyme
uv sync --frozen --extra dev --python 3.11

# VOICEVOX ENGINEを127.0.0.1:50021で起動してから、明瞭語を一度だけ合成
uv run --frozen python scripts/synthesize_anchor_words.py \
  --profile rhyme-final --speaker 2 \
  --output-dir generated_audio/article_words \
  --manifest generated_audio/article_words/manifest.json

# 基準 → 一部明瞭語 → 押韻文脈 → 冒頭明瞭語 → 記事用集計
uv run --frozen python scripts/reproduce_article.py \
  --anchor-manifest generated_audio/article_words/manifest.json \
  --output-dir results/qiita
```

既定値は記事と同じ`large-v3`・各20回です。関連する対照条件を含めて240 decodeを実行するため、まず`--model tiny --n-samples 1 --output-dir results/qiita_smoke`で接続・実行を確認できます。tinyの出力は記事の数値と比較できません。

結果は`results/qiita/article_summary.md`と`article_summary.json`に出ます。記事の6条件の集計と共通seed 3585の生出力を確認できます。中断時は同じコマンドに`--resume`を付けます。明瞭語の合成はやり直さず、同じ音声と環境を使ってください。

環境、記事とコードの対応、各段階のコマンド、再現できる範囲は[再現手順](docs/reproduction.md)にまとめています。

## ファイル構成

|場所|内容|
|---|---|
|[docs/reproduction.md](docs/reproduction.md)|記事の6条件、実行順、設定、結果の読み方|
|[docs/experiments.md](docs/experiments.md)|母音・語彙アンカー・モデルサイズなど、その他の実験とCLIの詳細|
|[scripts/](scripts/)|音声合成、実験実行、集計のCLI|
|[src/whisper_rap/](src/whisper_rap/)|合成音声、Whisper decode、測定処理|
|[prompts/](prompts/)|実験用のオリジナル文脈とprefix|
|[reports/](reports/)|保存済みの実験結果の報告|
|[tests/](tests/)|音声の条件間一致、測定、実行順などの検証|

生成音声、全出力JSONL、実行時manifest、モデル重みはGitに含めません。入力・設定・ソース・環境の記録は実行時の出力ディレクトリに保存します。

## 音声のクレジット

明瞭語の合成音声：**VOICEVOX:四国めたん**（ノーマル、speaker ID 2）。音声を配布する場合は、[VOICEVOXの利用規約](https://voicevox.hiroshiba.jp/term/)と[四国めたんの利用規約](https://zunko.jp/con_ongen_kiyaku.html)を確認してください。依存ソフトウェアとモデルの利用条件は各配布元に従ってください。

## 開発時の確認

```bash
uv run --frozen pytest
uv run --frozen ruff check .
```

## ライセンス

Copyright 2026 shimajiroxyz.

本リポジトリのソースコードは[Apache License 2.0](LICENSE)で提供します。合成音声、外部ソフトウェア、モデル重みには各配布元の利用条件が適用されます。
