from whisper_rap.comparison import output_diagnostics
from whisper_rap.metrics import analyze_text


def diagnose(text):
    return output_diagnostics({**analyze_text(text), "text": text, "tokens": [], "sample_len": 64})


def test_repetition_is_distinguished_from_long_japanese() -> None:
    repeated = diagnose("ありがとう" * 6)
    assert repeated["repetitive"]
    assert not repeated["nontrivial_japanese_candidate"]
    sentence = diagnose("夜の街を歩きながら明日のことを考える")
    assert not sentence["repetitive"]
    assert sentence["nontrivial_japanese_candidate"]


def test_punctuation_empty_and_boilerplate_are_not_candidates() -> None:
    for text in ("", "、。", "ご視聴ありがとうございました", "夜"):
        assert not diagnose(text)["nontrivial_japanese_candidate"]
    assert not diagnose("、。")["japanese_output"]
    assert diagnose("夜")["japanese_output"]


def test_short_repeat_is_not_treated_as_dominant_loop() -> None:
    assert not diagnose("ははは、今日は街を歩こうと思っています")["repetitive"]


def test_prompt_copy_is_detected_without_treating_one_anchor_as_copy() -> None:
    text = "答えはまだ遠いですね"
    row = {**analyze_text(text), "text": text, "tokens": [], "sample_len": 64,
           "prompt": "迷いを越えて　答えはまだ遠い"}
    assert output_diagnostics(row)["prompt_echo"]
    assert not output_diagnostics(row)["nontrivial_japanese_candidate"]
    row["prompt"] = "夜"
    assert not output_diagnostics(row)["prompt_echo"]
