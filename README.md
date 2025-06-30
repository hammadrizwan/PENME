# PENME - Resolving Lexical Bias in Model Editing
![Alt Text](PENME.png)

<br>
Model editing aims to modify the outputs of large language models after they are trained. Previous approaches have often involved direct alterations to model weights, which can result in model degradation. Recent techniques avoid making modifications to the model's weights by using an adapter that applies edits to the model when triggered by semantic similarity in the representation space. We demonstrate that current adapter methods are critically vulnerable to strong lexical biases, leading to issues such as applying edits to irrelevant prompts with overlapping words. This paper presents a principled approach to learning a disentangled representation space that facilitates precise localization of edits by maintaining distance between irrelevant prompts while preserving proximity among paraphrases. In our empirical study, we show that our method (Projector Editor Networks for Model Editing - PENME) achieves state-of-the-art model editing results while being more computationally efficient during inference than previous methods and adaptable across different architectures.<br /> <br /> 

📄 **Paper ICML 2025**: [Resolving Lexical Bias in Model Editing](https://www.arxiv.org/abs/2408.10411)


##  Evaluation Overview

We evaluate PENME (Projector Editor Networks for Model Editing) across two major benchmarks: Counterfact and zsRE, and on multiple transformer architectures including T5-small, GPT2-XL, and LLaMA-2 7B. The evaluation focuses on three key criteria: **Edit Success**, **Generalization to paraphrases**, and **Locality**, i.e., preventing undesired changes to unrelated prompts. 

We provide pretrained projector networks and codebooks at **URL TBD**, and dataset vector files at **URL TBD**.

## Environment Setup
Setting up the Python environment:

   `pip install -r requirements.txt`

## ZsRE
For ZsRE, there are two modes of evaluation: (1) batch editing and (2) cross-dataset generalisation using PENME, trained on the Counterfact dataset.


1. T5 batch editing
   
   `python main.py --eval_type zsre_t5 --evaluation_mode batch`
3. T5 cross-dataset editing
   
   `python main.py --eval_type zsre_t5 --evaluation_mode transferlearning`
5. GPT-2XL batch editing
   
   `python main.py --eval_type zsre_gpt2xl --evaluation_mode batch`
7. GPT-2XL cross-dataset editing
   
   `python main.py --eval_type zsre_gpt2xl --evaluation_mode transferlearning`
9. Llama-2-7b batch editing
    
    `python main.py --eval_type zsre_llama --evaluation_mode batch`
11. Llama-2-7b cross-dataset editing
    
    `python main.py --eval_type zsre_llama --evaluation_mode transferlearning`

## Counterfact
For Counterfact, there are two modes of evaluation: (1) batch editing and (2) stream editing/incremental editing


1. T5 batch editing
   
   `python main.py --eval_type counterfact_t5 --evaluation_mode batch`
3. T5 cross-dataset editing
   
   `python main.py --eval_type counterfact_t5 --evaluation_mode stream`
5. GPT-2XL batch editing
   
   `python main.py --eval_type counterfact_gpt2xl --evaluation_mode batch`
7. GPT-2XL cross-dataset editing
   
   `python main.py --eval_type counterfact_llama --evaluation_mode stream`
9. Llama-2-7b batch editing
    
    `python main.py --eval_type counterfact_llama --evaluation_mode batch`
11. Llama-2-7b cross-dataset editing
    
    `python main.py --eval_type counterfact_llama --evaluation_mode stream`
##
All projectors can be retrained by using the `--force_train` flag. Example usage:
`python main.py --eval_type counterfact_llama --evaluation_mode stream --force_train`

All results are stored in the results subfolder.


- - -

### Contact

For further assistance, email: 

*hammad.rizwan@dal.ca*

### 📎 Citation
```bibtex
@inproceedings{rizwan2025resolving,
  title     = {Resolving Lexical Bias in Edit Scoping with Projector Editor Networks},
  author    = {Rizwan, Hammad and Rosati, Domenic and Wu, Ga and Sajjad, Hassan},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  note      = {To appear}
}

```

### License

Shield: [![CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by/4.0/)

This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).

[![CC BY 4.0](https://i.creativecommons.org/l/by/4.0/88x31.png)](http://creativecommons.org/licenses/by/4.0/)
