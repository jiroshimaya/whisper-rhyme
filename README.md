# Whisper Rap Hallucination Experiment

OpenAI Whisper `large-v3` を通常の音声認識器ではなく、曖昧な音響入力で条件付ける日本語テキスト生成器として調べるための最小実験環境です。外部音源や既存歌詞は使わず、ビートと人声風の疑似ボーカルをすべてローカル合成します。

この実験は「良いラップを自動生成する」ことを最初の成功条件にしていません。まず、ビートと疑似発声に対して `large-v3` が入力に存在しない日本語文を継続的に生成するか、その量・多様性・decoderの確信度を記録します。

## 実験の仕組み

公式 `openai-whisper` の `model.decode()` に30秒のlog-Mel spectrogramを直接渡します。`model.transcribe()` の no-speech判定、logprob threshold、timestampに基づくsilence/hallucination除外は通りません。デフォルトではblank tokenおよびnon-speech tokenの抑制も無効です（CLIで有効化できます）。

4条件を比較します。

| 条件 | ビート | 疑似ボーカル | decoder prompt |
|---|---:|---:|---:|
| A | ✓ |  |  |
| B |  | ✓ |  |
| C | ✓ | ✓ |  |
| D | ✓ | ✓ | ✓ |

Dの音声はCと同一です。違いはWhisper decoderに渡すオリジナルのラップ風文脈だけです。

疑似ボーカルには次の3方式があります。

- `vowel`: `a/i/u/e/o` の倍音・formantを持つ母音列
- `cv`: `ta/ka/ra/na/ma/sa/da/ba/pa` 風の、ノイズ化した子音とformant-shiftした母音
- `hum`: 「んー／あー／うー」に近い低域寄りの曖昧な発声
- `degraded_tts`: オリジナルの無意味仮名TTSを帯域制限・速度揺らぎ・短いdropoutで崩した発声

## セットアップ

Python 3.10以上、FFmpeg、十分なRAMまたはVRAMが必要です。`large-v3` は大きいため、初回はモデルのダウンロードにも時間と空き容量を要します。

```bash
uv sync --extra dev
```

通常のpip環境では次も使えます。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

以下の例は `uv run` を使います。pip環境では先頭の `uv run` を外してください。

## 音声生成

```bash
uv run python scripts/generate_inputs.py \
  --condition C \
  --duration 30 \
  --bpm 90 \
  --seed 1234
```

出力はデフォルトで `generated_audio/condition_C_seed_1234_cv.wav`、設定と実際の疑似音節イベントは同名のJSONです。A〜D、`--voice-mode vowel|cv|hum`、`--vowels aioai` を切り替えられます。同じ設定とseedからは同じ音声が得られます。

### 指定モーラタイミング

第二段階用に、開始時刻・長さ・任意の疑似音節をJSONで指定できます。

```json
[
  {"start": 0.00, "duration": 0.20, "symbol": "ta"},
  {"start": 0.25, "duration": 0.20, "symbol": "ra"},
  {"start": 0.50, "duration": 0.40, "symbol": "na"},
  {"start": 1.00, "duration": 0.20, "symbol": "ma"}
]
```

```bash
uv run python scripts/generate_inputs.py \
  --condition C --seed 1234 --rhythm-file rhythm.json
```

母音条件付けには `--voice-mode vowel --vowels aioai` を使います。イベント側の `symbol` に母音を指定することもできます。

## Whisper decode実験

音声がまだなければ同じ引数で自動生成します。

```bash
uv run python scripts/run_experiment.py \
  --condition C \
  --model large-v3 \
  --temperature 0.8 \
  --n-samples 50 \
  --seed 1234
```

多言語版の `tiny`、`small`、`large-v3` に対応します。`openai/whisper-<name>` も同じcheckpointへ正規化します。英語専用の `.en` と `large-v3-turbo` は比較対象外として拒否します。既定値は `large-v3` です。

温度スイープ（`0.0, 0.2, 0.4, 0.6, 0.8, 1.0`）は次の通りです。

```bash
uv run python scripts/run_experiment.py \
  --condition C --temperature-sweep --n-samples 50 --seed 1234
```

複数の任意温度は `--temperature` を繰り返します。`temperature > 0` では `--best-of N` も指定できます。各sampleには独立した `trial_seed = seed + sample_id` を設定します。`temperature=0` はgreedy decodeなので、同じ入力の反復結果が同一になるのが正常です。

### promptあり（条件D）

```bash
uv run python scripts/run_experiment.py \
  --condition D \
  --model large-v3 \
  --temperature 0.8 \
  --n-samples 50 \
  --seed 1234 \
  --prompt-file prompts/rap_01.txt
```

promptはinstructionではなく、decoderの直前文脈として渡されます。リポジトリ内のpromptはすべて実験用のオリジナル短文です。

`--prefix-file`を併用すると、直前文脈とは別に次の出力の書き出しを条件付けできます。生成結果にはprefix自体を含まない`text`と、結合した`full_text`の両方を保存します。既定ではWhisperのnon-speech記号と言語切替tokenを抑制しますが、no-speech確率による結果の破棄は行いません。

## ラップ誘導2×2×2実験

ローカルVOICEVOXから著作物を含まない無意味仮名列を作り、劣化TTS入力を生成します。音声を外部配布する場合は使用したVOICEVOX話者の利用規約とクレジット条件を確認してください。

```bash
uv run python scripts/synthesize_gibberish.py \
  --output generated_audio/gibberish_source.wav

uv run python scripts/generate_inputs.py \
  --condition C --duration 16 --bpm 90 --seed 1234 \
  --voice-mode degraded_tts \
  --tts-source generated_audio/gibberish_source.wav \
  --output generated_audio/matrix_degraded_tts.wav
```

現行CVと劣化TTS、promptのみとprompt+prefix、temperature 0.6と0.8を各20回比較します。モデルは一度だけロードされ、各行は完了直後に保存されます。

```bash
uv run python scripts/run_conditioning_matrix.py \
  --cv-audio generated_audio/matrix_cv.wav \
  --tts-audio generated_audio/matrix_degraded_tts.wav \
  --prompt-file prompts/rap_context.txt \
  --prefix-file prompts/rap_prefix.txt \
  --temperature 0.6 0.8 --n-samples 20 \
  --device cpu \
  --output results/conditioning_matrix.jsonl
```

中断後は同じコマンドへ`--resume`を加えます。結果には`acoustic_mode`、`conditioning_mode`、定型的ASR hallucinationの検出結果、最低限の日本語候補条件を満たす`usable_candidate`を追加します。

## 出力

結果は1 decodeにつき1行のJSONLです。長時間実験で途中停止しても、完了済みの行は都度flushされます。

```json
{
  "condition": "C",
  "seed": 1234,
  "trial_seed": 1236,
  "temperature": 0.8,
  "sample_id": 2,
  "audio_file": "generated_audio/condition_C_seed_1234_cv.wav",
  "prompt": null,
  "text": "...",
  "tokens": [50364],
  "avg_logprob": -1.2,
  "no_speech_prob": 0.82,
  "compression_ratio": 1.1,
  "character_count": 12,
  "japanese_character_ratio": 1.0,
  "reading": "...",
  "moras": ["..."],
  "mora_count": 10,
  "mora_per_second": 0.33,
  "phrase_mora_counts": [10],
  "phrase_end_moras": [{"last_1": ["..."], "last_2": ["..."], "last_3": ["..."]}]
}
```

`no_speech_prob` が高くても結果は破棄しません。これにより、モデルが音声なしと判断しながらどの程度の確信度でテキストを生成するかを後から調べられます。

## 母音による押韻制御実験

無意味なCV音声の `/a/` の量と位置を変え、認識結果のフレーズ末尾が「あ」または「ああ」になる割合を比較できます。4条件は同じ子音、タイミング、ビート、音量、decoder prompt、生成seedを共有します。

| 条件 | 入力母音 |
|---|---|
| `balanced` | 5母音をほぼ均等に配置 |
| `a_heavy` | 全体の70%を `/a/` にする |
| `a_shuffled` | `a_final` と同じ母音数をランダム配置 |
| `a_final` | 8つの各フレーズ末尾2モーラを `/a a/` にする |

`a_final` と `a_shuffled` は母音のヒストグラムが完全に同じなので、両者の差は `/a/` の量ではなく配置によるものです。

```bash
uv run python scripts/run_vowel_rhyme_experiment.py \
  --prompt-file prompts/vowel_rhyme_context.txt \
  --prefix-file prompts/vowel_rhyme_prefix.txt \
  --temperature 0.6 --n-samples 20 --device cpu \
  --output results/vowel_rhyme.jsonl \
  --summary results/vowel_rhyme_summary.json
```

JSONLには認識テキストの全母音に占める「あ」の割合、フレーズ末尾1母音・2母音のtarget一致率、フレーズ末尾同士の一致率を保存します。summaryは条件ごとの合算値です。空白もフレーズ境界として扱うため、Whisperが句読点の代わりに出す全角・半角スペースも評価に含まれます。

seed 2468、temperature 0.6、各20 decodeの探索的実行では次の結果でした。

| 条件 | 入力の `/a/` 率 | 出力の平均 `/a/` 率 | 語尾 `/a/` | 語尾 `/a a/` |
|---|---:|---:|---:|---:|
| `balanced` | 20.3% | 25.2% | 28.6% | 8.6% |
| `a_heavy` | 70.3% | 31.2% | 35.1% | 8.1% |
| `a_shuffled` | 43.8% | 31.8% | 28.1% | 12.5% |
| `a_final` | 43.8% | 32.3% | 43.5% | 17.4% |

同じ母音数の `a_final` と `a_shuffled` で、出力全体の `/a/` 率はほぼ同じまま、語尾 `/a/` は15.4ポイント、語尾 `/a a/` は4.9ポイント高くなりました。ただしdecode単位のペア差は語尾 `/a/` が+7.1ポイント（bootstrap 95%区間 -9.0〜+22.1）、語尾 `/a a/` が+2.6ポイント（-1.8〜+8.8）で、各20本では統計的に確定していません。母音の位置による弱い誘導信号として扱い、seedや標本数を増やして再検証する必要があります。

## 明瞭語による語彙アンカー実験

各フレーズの冒頭へ短い明瞭語を混ぜ、残りを無意味CV音声にすると、Whisperがアンカー語を含む長い行として補完するかを調べられます。テーマ語 `夜・街・影・鼓動`、同じ2〜3モーラの中立語 `これ・ただ・いま・ことば`、アンカーなしの3条件を、語尾 `/a a/` の有無と組み合わせた3×2実験です。

ローカルVOICEVOXでアンカー音声とmanifestを作ります。

```bash
uv run python scripts/synthesize_anchor_words.py \
  --output-dir generated_audio/lexical_anchors/words \
  --manifest generated_audio/lexical_anchors/manifest.json
```

音声を外部配布する場合は、使用したVOICEVOX話者の利用規約とクレジット条件を確認してください。実験は次のコマンドで実行します。

```bash
uv run python scripts/run_lexical_anchor_experiment.py \
  --anchor-manifest generated_audio/lexical_anchors/manifest.json \
  --prompt-file prompts/lexical_anchor_context.txt \
  --prefix-file prompts/vowel_rhyme_prefix.txt \
  --temperature 0.6 --n-samples 20 --device cpu \
  --output results/lexical_anchor.jsonl \
  --summary results/lexical_anchor_summary.json
```

seed 3579、各20 decodeの実行結果は次の通りです。「アンカー認識」は各条件で挿入した語の読みが1語以上出た割合、「テーマ語出現」は `夜・街・影・鼓動` の表記が1語以上出た割合です。

| アンカー | 語尾 `/a a/` | アンカー認識 | テーマ語出現 | 平均テーマ語数 | 平均文字数 | 語尾 `/a/` | 語尾 `/a a/` |
|---|---:|---:|---:|---:|---:|---:|---:|
| なし | なし | 0% | 0% | 0.00 | 14.1 | 18.0% | 2.0% |
| 中立語 | なし | 40% | 0% | 0.00 | 9.8 | 54.8% | 11.9% |
| テーマ語 | なし | **80%** | **85%** | **3.05** | **33.8** | 14.4% | 0.7% |
| なし | あり | 0% | 5% | 0.05 | 12.3 | 22.5% | 5.0% |
| 中立語 | あり | 40% | 0% | 0.00 | 5.5 | 54.8% | 12.9% |
| テーマ語 | あり | **80%** | **85%** | **2.70** | **26.7** | 34.5% | 8.4% |

テーマ語ありと中立語ありのペア差は、テーマ語出現率が+85ポイント（bootstrap 95%区間 +70〜+100）、平均テーマ語数が語尾操作なしで+3.05語（+2.05〜+4.15）でした。明瞭語による語彙制御は明確です。一方、事前定義した関連語への波及はテーマ条件5%、アンカーなし条件でも最大5%で、直接挿入していない意味語まで増える証拠は得られませんでした。

テーマ語条件で語尾操作を有効にすると、フレーズ加重の語尾 `/a/` は14.4%から34.5%、`/a a/` は0.7%から8.4%へ増えました。decode単位のペア差はそれぞれ+9.6ポイント（-1.8〜+22.1）、+4.4ポイント（0.0〜+9.6）なので、語彙アンカーと押韻の同時制御は可能性があるものの、韻側は追加検証が必要です。

## モデルサイズ比較

保存済みのアンカー音声6条件を、`tiny`、`small`、`large-v3`へ同じ設定で渡します。入力音声を再合成せず、WAVのSHA-256、prompt、各sampleのseed、ライブラリversionを保存します。音声は上記のアンカー実験で生成した `none/neutral/theme_rhyme_off/on.wav` を使用します。

```bash
uv run python scripts/run_model_size_comparison.py \
  --audio-dir generated_audio/lexical_anchor \
  --output-dir results/model_size_n20 \
  --models tiny small large-v3 \
  --temperature 0.6 --n-samples 20 --seed 3579 \
  --device cpu --threads 8

uv run python scripts/summarize_model_comparison.py results/model_size_n20
```

同じコマンドへ `--resume` を加えると完了済みsampleを再計算せず再開します。音声、設定、ライブラリversionが変わっていれば再開を拒否します。既存結果は明示的な再開なしには上書きしません。

同一音声のEncoder出力は1回計算して再利用し、Decoderはsampleごとにseedを設定します。モデルごとのMel bin数と特殊tokenに対応しています。`large-v3`は言語数が異なるため、比較実装ではtoken抑制リストもモデル自身のtokenizerから取得します。過去の実行結果とは混ぜず、3モデルを同じ実装・device・精度で再実行してください。

`samples.jsonl`には全出力、`manifest.json`には入力条件、`analysis.json`にはモデル別・条件別の集計を保存します。表層的な日本語候補率、反復率、promptのコピー率、アンカー出現率、語尾母音率を確認できます。日本語候補の判定は意味の自然さを保証せず、文字の出現率はハルシネーションの正確な発生率ではありません。

## 集計と可視化

```bash
uv run python scripts/summarize_results.py results/ \
  --output results/summary.json \
  --plot-dir results/plots
```

全体・condition別・temperature別に次を集計します。

- 文字数、日本語文字率、empty/non-empty数
- unique output率と頻出出力上位10件（重複数・率）
- `pykakasi` による読み変換と `jamorasep` 基準のモーラ数
- mora/sec、フレーズ別モーラ数、語尾1/2/3モーラ

PNGにはcondition別平均文字数、日本語文字率、mora/sec、temperature別平均文字数とunique率を保存します。

## 仮説の読み方

1. AとCの平均文字数・non-empty率を比較し、疑似ボーカル追加で生成量が増えるかを見る。
2. mora eventの密度やdurationを変え、出力モーラ数との対応を見る。
3. CとDを同じ音声・seed・temperatureで比較し、promptによる語彙や行構造の変化を見る。
4. temperature別unique率を比較し、多様性が増えるかを見る。

出力モーラ数と入力event数の一致だけで音響的な追従とは断定できません。まず相関を探索し、その後はtoken timestampやforced alignmentを使う検証が必要です。

## 開発時の確認

```bash
uv run pytest
uv run ruff check .
```

既定値は [`configs/default.yaml`](configs/default.yaml) に一覧化しています。CLI引数が実行時の正本です。
