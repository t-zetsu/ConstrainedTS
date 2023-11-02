# Lexically Constrained Decoding with Edit Operation Prediction for Controllable Text Simplification
We explain the details in the following paper. [[PDF]](https://aclanthology.org/2022.tsar-1.13.pdf)
```
@inproceedings{zetsu-etal-2022,
    author      = {Tatsuya, Zetsu and Kajiwara, Tomoyuki and Arase, Yuki},
    title       = {Lexically Constrained Decoding with Edit Operation Prediction for Controllable Text Simplification},
    booktitle   = {Proceedings of the Workshop on Text Simplification, Accessibility, and Readability (TSAR-2022)},
    year        = {2022},
    pages       = {147--153},
    url         = {https://aclanthology.org/2022.tsar-1.13/}
}
```

### Description
Our method predicts edit operations conditioned to a target level and creates positive/negative constraints for words that should/should not appear in an output sentence.
Refer to the [input data](https://github.com/t-zetsu/ConstrainedTS/tree/main/data/datasets/sample) and [sample.sh](https://github.com/t-zetsu/ConstrainedTS/tree/main/sample.sh) to create level-based constraints

You generate sentences with [neurologic decoding](https://github.com/GXimingLu/neurologic_decoding) using the constraints you have created.
