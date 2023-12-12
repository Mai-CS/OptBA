# OptBA: Optimizing Hyperparameters with the Bees Algorithm for Improved Medical Text Classification
Mai A. Shaaban, Mariam Kashkash, Maryam Alghfeli, Adham Ibrahim

**Mohamed bin Zayed University of Artificial Intelligence, Abu Dhabi, UAE**

[![Static Badge](https://img.shields.io/badge/Paper-Link-yellowgreen?link=https%3A%2F%2Fzenodo.org%2Frecords%2F10104139)](https://arxiv.org/abs/2303.08021)
[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![tensorflow](https://img.shields.io/badge/TensorFlow-2.11-FF6F00.svg?style=flat&logo=tensorflow)](https://www.tensorflow.org)
![Static Badge](https://img.shields.io/badge/License-Apache-blue?link=https%3A%2F%2Fgithub.com%2FMai-CS%2FOptBA%2Fblob%2Fmain%2FLICENSE)

## Abstract
One of the challenges that artificial intelligence engineers face, specifically in the field of deep learning is obtaining the optimal model hyperparameters. The search for optimal hyperparameters usually hinders the progress of solutions to real-world problems such as healthcare. To overcome this hurdle, the proposed work introduces a novel mechanism called "OptBA" to automatically fine-tune the hyperparameters of deep learning models by leveraging the Bees Algorithm, which is a recent promising swarm intelligence algorithm. In this paper, the optimization problem of OptBA is to maximize the accuracy in classifying ailments using medical text, where initial hyperparameters are iteratively adjusted by specific criteria. Experimental results demonstrate a noteworthy enhancement in accuracy with approximately 1.4%. This outcome highlights the effectiveness of the proposed mechanism in addressing the critical issue of hyperparameter optimization and its potential impact on advancing solutions for healthcare and other societal challenges. 


## Prerequisites
Before you begin, ensure you have met the requirements by running the following command: ```pip install -r requirements.txt```

## Dataset
Data files are included in the ```data``` folder.

*Link*: https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent

## Usage
To reproduce results, just run the following command: ```python BA.py```

### Customization:
- To apply different LSTM structures or other deep learning models, use ```lstm_module.py```
- For text/data preprocessing, use ```preprocessing_module.py```

## License
This project uses [Apache License Version 2.0](https://github.com/Mai-CS/OptBA/blob/main/LICENSE).
