# Qiita記事の再現手順

対象は「[Whisperくん、韻を踏みなさい](https://qiita.com/shimajiroxyz/items/9653c39599cb2eb59695)」（2026-10-28更新）です。記事の6条件を順に実行し、同じdecoder seedの認識結果を取り出します。過去の母音制御やモデルサイズ比較は[別の実験一覧](experiments.md)に分けています。

## 記事と実験条件の対応

|記事の条件|保存先の段階|`audio_mode`|`context_mode`|入力WAV|
|---|---|---|---|---|
|BGMのみ|`baseline`|`beat`|`none`|`baseline/audio/beat.wav`|
|BGM＋曖昧CV（語尾 `/a a/`）|`baseline`|`cv_final`|`none`|`baseline/audio/cv_final.wav`|
|2・4フレーズ末尾に「花・坂」|`partial`|`partial_final`|`none`|`partial/audio/partial_final.wav`|
|全4フレーズ末尾に「朝・花・山・坂」|`baseline`|`clear_final`|`none`|`baseline/audio/clear_final.wav`|
|全4フレーズ冒頭に「朝・花・山・坂」|`onset`|`clear_onset`|`none`|`onset/audio/clear_onset.wav`|
|2・4フレーズ末尾に明瞭語＋押韻文脈|`context`|`partial_final`|`prompt`|`context/audio/partial_final.wav`|

保存先はすべて`results/qiita/`からの相対パスです。最後の条件の音声PCMは、一部明瞭語条件と同一です。`baseline`の`cv_balanced`や旧文脈・prefix付きの条件は追加の対照であり、記事の「BGM＋曖昧CV」は`cv_final`を指します。

## 環境の準備

リポジトリのルートで実行します。

```bash
uv sync --frozen --extra dev --python 3.11
ffmpeg -version
```

FFmpegは別途インストールしてください。`uv.lock`でPythonパッケージを固定します。`requirements.txt`からのpipインストールはバージョンを固定しないため、記事の再実行には上のコマンドを使ってください。

記事の保存済みmanifestにはPython 3.11.10、openai-whisper 20250625、PyTorch 2.14.0+cu130を記録しています。Whisperの実験はCPU・float32・8スレッドで行いました。CUDA付きのPyTorchでもCPUで推論するため、GPUは必須ではありません。lockfileのLinux向け依存にはCUDAライブラリが含まれ、ダウンロードと展開に数GB以上の容量を使います。large-v3の重みも別途ダウンロードします。

VOICEVOX ENGINEは別プロセスで起動します。既定の接続先は`http://127.0.0.1:50021`です。起動方法は[公式ドキュメント](https://github.com/VOICEVOX/voicevox_engine)を参照してください。接続と使用する話者を確認できます。

```bash
curl --fail http://127.0.0.1:50021/version
curl --fail http://127.0.0.1:50021/speakers
```

記事で使った音声はspeaker ID 2の四国めたん（ノーマル）です。エンジンのバージョンと話者IDの対応を確認し、実行時の`/version`の応答を手元に保存してください。元実験ではVOICEVOX ENGINEのバージョンを固定していなかったため、再合成した明瞭語が当時のWAVと完全一致することは保証できません。

## 明瞭語の合成

「朝・花・山・坂」を一度だけ合成します。合成音声のクレジットは**VOICEVOX:四国めたん**です。

```bash
uv run --frozen python scripts/synthesize_anchor_words.py \
  --profile rhyme-final --speaker 2 \
  --url http://127.0.0.1:50021 \
  --output-dir generated_audio/article_words \
  --manifest generated_audio/article_words/manifest.json
```

`rhyme_final_0.wav`〜`rhyme_final_3.wav`と、単語・音声パスを対応させた`manifest.json`を生成します。読み上げ速度1.12、pitchScale −0.02、intonationScale 0.8、16 kHz monoなどの設定は[合成スクリプト](../scripts/synthesize_anchor_words.py)にあります。

この合成コマンドは既存の音声を上書きします。実験開始後や再開時には実行せず、別の合成設定を試す場合は新しい保存先にしてください。

## 一括実行と再開

```bash
# 実行順だけを確認（音声生成・モデルロードなし）
uv run --frozen python scripts/reproduce_article.py \
  --anchor-manifest generated_audio/article_words/manifest.json \
  --output-dir results/qiita --dry-run

# 記事と同じモデル・標本数で実行
uv run --frozen python scripts/reproduce_article.py \
  --anchor-manifest generated_audio/article_words/manifest.json \
  --output-dir results/qiita

# 中断後は同じコマンドに--resumeを付ける
uv run --frozen python scripts/reproduce_article.py \
  --anchor-manifest generated_audio/article_words/manifest.json \
  --output-dir results/qiita --resume
```

基準6条件×20回、一部明瞭語3条件×20回、押韻文脈1条件×20回、冒頭明瞭語2条件×20回の計240 decodeです。各段階でモデルをロードし、同じ音声のEncoder出力はその段階内で再利用します。CPUでの所要時間は環境に依存します。

再開時は同じモデル・標本数・seed・スレッド数を指定してください。完了済みの段階とsampleは検証後に再利用し、まだ始めていない段階は新規実行します。各段階のmanifestで入力、ソース、設定、環境が変わっていると再開を拒否します。実験が終わるまで、音声・出力ディレクトリを移動したり、推論コードやライブラリを更新したりしないでください。

既存ディレクトリへの新規実行は拒否します。manifestがない中途半端な段階も自動削除せず停止するため、そのデータを保ったまま別の出力先を選びます。

最初の動作確認は、別ディレクトリで`tiny`・各1回にできます。

```bash
uv run --frozen python scripts/reproduce_article.py \
  --anchor-manifest generated_audio/article_words/manifest.json \
  --output-dir results/qiita_smoke --model tiny --n-samples 1
```

この実行のdecoder seedは3579だけです。記事の例のseed 3585は含まれず、記事の出力文や数値とも比較できません。

## 段階ごとに実行する場合

一括コマンドは次の既存CLIを順に呼びます。同じ出力先で一括実行と重ねて新規実行する必要はありません。

```bash
# 1. 基準音声と6条件（120 decode）
uv run --frozen python scripts/run_article_experiment.py \
  --anchor-manifest generated_audio/article_words/manifest.json \
  --output-dir results/qiita/baseline \
  --model large-v3 --n-samples 20 --seed 3579 --threads 8

# 2. 2・4フレーズだけを明瞭語にする（60 decode）
uv run --frozen python scripts/run_article_experiment.py \
  --anchor-manifest generated_audio/article_words/manifest.json \
  --frozen-baseline-dir results/qiita/baseline \
  --output-dir results/qiita/partial \
  --model large-v3 --n-samples 20 --seed 3579 --threads 8

# 3. 一部明瞭語と同じ音声に4行の押韻文脈を与える（20 decode）
uv run --frozen python scripts/run_article_experiment.py \
  --anchor-manifest generated_audio/article_words/manifest.json \
  --frozen-baseline-dir results/qiita/baseline --rhyming-context-only \
  --prompt-file prompts/rap_aa_four_phrases.txt \
  --output-dir results/qiita/context \
  --model large-v3 --n-samples 20 --seed 3579 --threads 8

# 4. 同じ録音をフレーズ冒頭へ移す（40 decode）
uv run --frozen python scripts/run_onset_experiment.py \
  --baseline-dir results/qiita/baseline --partial-dir results/qiita/partial \
  --anchor-manifest generated_audio/article_words/manifest.json \
  --output-dir results/qiita/onset

# 5. 記事の6条件を取り出す（推論なし）
uv run --frozen python scripts/summarize_article.py results/qiita
```

## 設定と出力の確認

|項目|記事の設定|
|---|---|
|音声|12秒、16 kHz mono、90 BPM、4フレーズ×12 CV相当|
|音声seed / decoder seed|3579 / 3579〜3598|
|モデル / 推論|多言語版large-v3 / CPU・float32・8スレッド|
|decoder|日本語、temperature 0.6、best_of 1、最大64 token、length_penalty 1.0|
|token抑制|blank抑制off、モデル固有のnon-speech・制御token抑制on|
|記事の文脈|[rap_aa_four_phrases.txt](../prompts/rap_aa_four_phrases.txt)、prefixなし|

`baseline`と`partial`で併せて実行する旧文脈・prefix付きの対照は、[lexical_anchor_context.txt](../prompts/lexical_anchor_context.txt)と[vowel_rhyme_prefix.txt](../prompts/vowel_rhyme_prefix.txt)を使います。記事の最後の押韻文脈条件と混ぜて集計しません。

各段階の`manifest.json`は入力WAV、音節イベントや録音、ソースのハッシュ、decoder設定、実行環境を記録します。`samples.jsonl`は1 decodeにつき1行で、完了ごとにflushします。`summary.json`はその段階の対照条件や対応seedのbootstrap区間を含む集計です。

一括実行の最後に、記事の6条件だけを`article_summary.json`と`article_summary.md`へ出力します。記事中の例は共通seed 3585で、レポートでは表示用の改行を追加しない生出力を示します。保存済みJSONLから集計だけをやり直すこともできます。

```bash
uv run --frozen python scripts/summarize_article.py results/qiita --example-seed 3585
```

当時の集計・例は次の報告に残しています。

- [基準6条件](../reports/article-20261028.md)
- [一部明瞭語とモーラ数](../reports/partial-article-20261028.md)
- [4行の押韻文脈](../reports/rhyming-context-20261028.md)
- [冒頭の明瞭語](../reports/onset-article-20261028.md)

## 再現できる範囲

この手順は合成から推論・集計までを再実行します。元音声・全出力JSONLはGitに含まれないため、保存済みデータをダウンロードして当時の全数値を再集計する手順ではありません。VOICEVOXの再合成音声、OS・CPU、ライブラリのビルドや乱数処理に差があると、seedを揃えても認識文や集計値が変わる場合があります。

各条件20回は一つの音声のサンプリング反復です。20種類の独立音源を使った評価ではありません。語尾率は空白・句読点で区切った出力句の母音一致を数え、意味の自然さや音楽の拍・発音時刻への対応は評価していません。明瞭語の直接転写、疑似CVと自然TTSの声質の差、64 tokenでの打ち切りも含めて解釈してください。
