# Revision-Quality-Prediction

The code for our paper Predicting the Quality of Revisions in Argumentative Writing

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

