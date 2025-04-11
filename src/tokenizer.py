import re


class TokenizerCustom:
    name = "TokenizerCustom"

    def encode(self, text):
        matches = re.finditer(
            r'([一-龠ぁ-ゔァ-ヴーａ-ｚＡ-Ｚ０-９々〆〤0-9゙゚]+|[a-zA-Z0-9]+)[.!]*', text)
        match_positions = [match.span() for match in matches]
        tokens = []
        current_pos = 0
        for start, end in match_positions:
            if current_pos < start:
                tokens.extend([*text[current_pos:start]])
            tokens.append(text[start:end])
            current_pos = end
        if current_pos < len(text):
            tokens.extend([*text[current_pos:]])

        return [t.strip() for t in tokens if t.strip()]

    def encode_batch(self, texts):
        return [self.encode(text) for text in texts]
