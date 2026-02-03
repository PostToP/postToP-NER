import re


class TokenizerCustom:
    name = "TokenizerCustom"

    def encode(self, text):
        matches = re.finditer(
            r"([一-龠ぁ-ゔァ-ヴーａ-ｚＡ-Ｚ０-９々〆〤0-9゙゚]+|[a-zA-Z0-9]+)[.!]*",
            text,
        )

        tokens = []
        current_pos = 0

        for match in matches:
            start, end = match.span()

            if current_pos < start:
                tokens.extend(list(text[current_pos:start]))

            tokens.append(text[start:end])
            current_pos = end

        if current_pos < len(text):
            tokens.extend(list(text[current_pos:]))

        return [t for t in tokens if t == "\n" or not t.isspace()]

    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]
