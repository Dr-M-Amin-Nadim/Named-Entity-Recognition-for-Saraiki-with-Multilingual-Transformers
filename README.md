# Saraiki NER Dataset

A Saraiki corpus in Perso-Arabic script manually annotated for named entity recognition (NER) using the BIO tagging scheme.

The dataset extends the Saraiki UPOS corpus introduced by Nawaz et al. (2026) with named entity annotations while preserving its original sentences, tokenization, and fixed train, development, and test splits.

## Dataset Statistics

| Split | Sentences | Tokens |
|---|---:|---:|
| Train | 1,666 | 28,377 |
| Dev | 208 | 3,636 |
| Test | 209 | 3,439 |
| **Total** | **2,083** | **35,452** |

## Annotation Scheme

The dataset covers five entity categories:

- PERSON
- LOCATION
- ORGANIZATION
- DATE
- OTHER

Annotations use 11 BIO labels:

`B-PERSON`, `I-PERSON`, `B-LOCATION`, `I-LOCATION`,
`B-ORGANIZATION`, `I-ORGANIZATION`, `B-DATE`, `I-DATE`,
`B-OTHER`, `I-OTHER`, and `O`.

`B-` marks the first token of an entity, `I-` marks a continuation token, and `O` marks a token outside any entity.

## Files

- `saraikiner/train.conllu`
- `saraikiner/dev.conllu`
- `saraikiner/test.conllu`

The files preserve the tokenization and sentence boundaries of the source corpus. NER labels are separate from UPOS tags.

## Data Source

The source text and predefined dataset splits come from the Saraiki UPOS corpus described in:

Nawaz, U., Iqbal, M. J., Asaf, M., Mir, M. M., Rehman, F., Alyas, T., & Hasnain, M. (2026). Neural POS Tagging for the Low-Resource Saraiki. arXiv preprint arXiv:2610.20279.

Our contribution is the manually annotated NER layer.

## Citation

When using this dataset, please cite the original Saraiki UPOS corpus paper:

Nawaz, U., Iqbal, M. J., Asaf, M., Mir, M. M., Rehman, F., Alyas, T., & Hasnain, M. (2026). Neural POS Tagging for the Low-Resource Saraiki. arXiv preprint arXiv:2610.20279.

Citation details for the Saraiki NER dataset will be added when available.

## License

The NER annotations are released under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
for non-commercial use, with attribution and share-alike requirements.

The underlying corpus remains subject to its original license.
