from threading import Lock

from sudachipy import Dictionary, SplitMode

from app.normalization import normalize_for_search


SEARCHABLE_PARTS_OF_SPEECH = {
    "名詞",
    "動詞",
    "形容詞",
    "副詞",
    "接頭辞",
}


class SudachiTokenizer:
    def __init__(self) -> None:
        self._tokenizer = Dictionary().create(
            mode=SplitMode.C,
            projection="normalized",
        )
        self._lock = Lock()

    def tokenize(self, text: str) -> list[str]:
        return [
            normalized
            for _, normalized in self.tokenize_with_surfaces(text)
        ]

    def tokenize_with_surfaces(self, text: str) -> list[tuple[str, str]]:
        normalized_text = normalize_for_search(text)

        with self._lock:
            morphemes = self._tokenizer.tokenize(normalized_text)

        return [
            (morpheme.surface(), morpheme.normalized_form())
            for morpheme in morphemes
            if morpheme.part_of_speech()[0] in SEARCHABLE_PARTS_OF_SPEECH
        ]
