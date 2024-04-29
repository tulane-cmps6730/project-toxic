# CMPS 6730 Safety classification using LM

This repository contains the codes for the nlp project named Safety classification using LM.

Large language models (LLMs) have demonstrated remarkable proficiency in tasks like chat completion, instruction following, coding, problem-solving, and decision-making. Considering the potential for broad societal impact, responses generated by LLMs must not contain harmful content, such as discrimination, misinformation, or violations of social norms and morals. One of the important steps is to train a binary classifier to identify if a sentence contains harmful content~\citep{dai2023safe}. In this project, we will train a simple binary classifier using LM to identify if the sentence contains harmful language.

### Contents

- [docs](docs): contains the slides for presentation
- [nlp](nlp): contains the demo for the web interface and command line interface and all the images for the demo
- [notebooks](notebooks): The main codes for training the cost model
- [report](report): report_final.pdf contains the final report
- [tests](tests): unit tests for project code (I don't use this code)


### Installation

Install the necessary dependencies:

```
pip install -r requirements.txt
```

Notice that we need gpu to run all the experiments and tests.

### Training

First, we train a sft model to build the cost model. (This step is optional.)

```
python3 notebooks/sft_llama2.py
```

Next, we train the cost model:

```
python3 notebooks/cost_trainer.py
```

