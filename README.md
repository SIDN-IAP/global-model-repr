# global-model-repr
Repository for the session on global-, model-, and representation-level interpretations. 

## Outline

### Probing classifiers for NLP models
1. Experiment 1 (20 minutes):
   - Load one pretrained model from the [Transformers library](https://huggingface.co/transformers/index.html)
   - Load dataset of texts with part-of-speech (POS) annotations
   - Run pretrained model on texts and extract representations
   - Train and evaluate linear classifier on classifying representations to POS tags
1. Experiment 2 (10 minutes):
   - Repeat the same for representations from all layers and compare accuracy across layers
1. Experiment 3 (10 minutes):
   - Repeat the same for non-linear classifier
1. Experiment 4 (10 minutes):
   - Create control experiment with random labels as per [Hewitt and Liang](https://arxiv.org/pdf/1909.03368.pdf)
   - Calculate selectivity and compare to previous results
1. Other topics as time permits:
   - Other word-level linguistic properties besides parts-of-speech
   - Sentence-level properties using aggregation of word-level representations or using sentence tokens
   - [Structural probe](https://github.com/john-hewitt/structural-probes/)
   - Methods for finding linguistic information in attention weights
   - Other models from the Transformers library 
