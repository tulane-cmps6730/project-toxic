# CMPS 6730 Safety classification using LM

This repository contains the codes for the nlp project named Safety classification using LM.

### Goals
Large language models (LLMs) have demonstrated remarkable proficiency in tasks like chat completion, instruction following, coding, problem-solving, and decision-making. Considering the potential for broad societal impact, responses generated by LLMs must not contain harmful content, such as discrimination, misinformation, or violations of social norms and morals. An essential component of safety alignment involves minimizing the tendency of a model to generate harmful responses through fine-tuning. One of the important steps is to train a binary classifier to identify if a sentence contains harmful content. In this project, we will train a simple binary classifier using LM to identify if the sentence contains harmful language. 

### Methods
We introduce a cost model to discriminate between safe and unsafe responses. We learn the model using the following pairwise comparison loss:


It’s worth noting that in the cost model, a response $y$ that is more harmful to the same prompt $x$ will yield a higher cost value. For unsafe responses, the cost value is positive; otherwise, it is negative.


### Conclusions
In this project, we train a simple binary classifier to identify if the sentence contains harmful language. The cost classifier will also give a cost value for each response. If the cost value is negative, it means the response is safe, otherwise, it means the response is unsafe. A higher cost value means the response is unsafer.  Experiments on the test datasets show that our cost model performs well for the safety classification task, which achieves an accuracy of 81.83\%. However, the accuracy of the ranking accuracy is low. In the future, we will use more complex models such as Llama2 instead of gpt-neo-1.3B for training the cost model in order to improve the performance.


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

