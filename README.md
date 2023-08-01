# Revision-Quality-Prediction

Paper: [Predicting the Quality of Revisions in Argumentative Writing](https://aclanthology.org/2023.bea-1.24/)

Abstract: The ability to revise in response to feedback is critical to students’ writing success. In the case of argument writing in specific, identifying whether an argument revision (AR) is successful or not is a complex problem because AR quality is dependent on the overall content of an argument. For example, adding the same evidence sentence could strengthen or weaken existing claims in different argument contexts (ACs). To address this issue we developed Chain-of-Thought prompts to facilitate ChatGPT-generated ACs for AR quality predictions. The experiments on two corpora, our annotated elementary essays and existing college essays benchmark, demonstrate the superiority of the proposed ACs over baselines.
## Dataset
Please download college revision desirability corpus from [https://petal-cs-pitt.github.io/data.html](https://petal-cs-pitt.github.io/data.html).
Please run `preprocess.py` to preprocess the data.
## Experiments
Please run `main.py` for classifying revision quality. Please provide appropriate parameters for each experiments. 

```angular2html
python main.py \
              --exp-dir=${OUTPUT_DIR} \
              --data-source=${DATA_SOURCE} \
              --purpose-type=${PURPOSE_TYPE} \
              --model-type=${MODEL_TYPE} \
              --context-type=${CONTEXT_TYPE} \
              --epochs=10 \
              --batch-size=16 \
              --learning-rate=5e-5 \
              --step-size=4 \
              --gamma=0.95 \
              --k-fold=10 \
              --random-seed=${SEED} \
              --model-name=${MODEL_NAME} 2>&1 | tee ${OUTPUT_DIR}/results.log
```

## Citation
```angular2html
@inproceedings{liu-etal-2023-predicting,
    title = "Predicting the Quality of Revisions in Argumentative Writing",
    author = "Liu, Zhexiong and Litman, Diane and Wang, Elaine and Matsumura, Lindsay and Correnti, Richard",
    booktitle = "Proceedings of the 18th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2023)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.bea-1.24",
    pages = "275--287",
}
```
