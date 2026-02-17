from transformers import AutoTokenizer


class TransformerTokenizer:
    def __init__(self, model_name: str, max_length: int = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.tokenizer.add_special_tokens(
        #     {"additional_special_tokens": ["[SEP]", "[NL]"]}
        # )
        self.max_length = max_length

    def encode(self, text: str):
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        return encoding
