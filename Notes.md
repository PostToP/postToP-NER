# Notes

```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```

## Problems & Thoughts

MeCab & other libraries split text way too much and poorly.
**Solution**: Custom tokenization using regex.

NER Tagging: IO is good enough for my use case, since token's are usually well separated.

LSTM vs CNN vs GRU

- LSTM: My inputs are too short, so LSTM is not the best choice.
- CNN: CNN is good for short inputs, but it requires a lot of data to train.
- GRU: GRU is a good choice for short inputs and requires less data than LSTM.

## Possible Features Extractions

- [x] Token is present in channel name **Improvement**
- [x] Number of occurrences of token in description
- [ ] Length of Token
- [ ] Token capitalization & Japanese character type
- [ ] Token is a stop word
- [ ] Token's classification by Spacy or other NLP library (noun, verb, etc.)
- [ ] Token is inside a delimiter (e.g. "【】")
- [ ] Video language
- [ ] Last resort: Token is present in any known channel name

{'loss': 0.07074445486068726, 'accuracy': 0.8577607870101929, 'f1_micro': 0.8782505910165485, 'f1_macro': 0.8571988199983384, 'f1_weighted': 0.8776672283152609, 'f1_per_class': array([0.91551629, 0.79864636, 0.85743381])}
{'loss': 0.0650283694267273, 'accuracy': 0.8659644722938538, 'f1_micro': 0.8918439716312057, 'f1_macro': 0.8699934506499689, 'f1_weighted': 0.8914531816728084, 'f1_per_class': array([0.93274041, 0.81144781, 0.86579213])}

HUNGARY ÚÁÉŐ -etc
