# DAT-550

## Bag of Words Document Classification using Feedforward Neural Network and Recurrent Neural Network

### Objective: 
The goal of this assignment is to train a simple feed-forward neural network and Recurrent Neural Network for document classification, focusing on various configurations of bags-ofwords features. 
The experimentation involves evaluating two different bags-of-words representations (for example, TF_IDF and CountVectorizer) and exploring the impact of the number
of hidden layers on the neural network's performance.

### Goal: 
The goal is to train neural networks on textual data. You will have to address a classification task with different architectures of different complexities including FFNN and RNN. The recommended machine learning framework in this course is PyTorch;

### Dataset and Classification Task: 
The dataset consists of two text columns: "Abstract" and "Field". You will be working with a dataset of scientific articles in English, containing the article abstracts and the field of the research (10 subcategories of computer science, physics, and math are considered). 

Your responsibility involves forecasting the research field by analyzing the abstract of the paper leading to a multiclass classification task.

The training data is as a gzipped comma-separated text file, where each data instance represents a paper and consists of two columns:
- abstract (input)
- label (output)

### Feature Extraction: 
Train a classifier to forecast a paper's field based on its abstract by converting abstracts into feature vectors using the bags-of-words representation technique mentioned earlier, suitable for feeding into a machine learning algorithm, such as a feed-forward neural network.

For this assignment, you must write the code that takes the provided dataset and extracts bags of words for each abstract, feeds these features into an NN-based classifier, and Outputs the field of the abstract. 

You must report the following metrics: 
- Accuracy
- macro-F1-score
- Precision
- Recall

You should experiment with:
- Two different variations of BoW features (for example, TF_IDF and CountVectorizer)
- Experiment with two NN architectures (FFNN, RNN)
- Different number of layers
  
### Citation: 
[Benchmark dataset for abstracts and titles of 100,000 ArXiv scientific papers.](https://paperswithcode.com/dataset/arxiv-10)
>@inproceedings{farhangi2022protoformer,
>  title={Protoformer: Embedding Prototypes for Transformers},
>  author={Farhangi, Ashkan and Sui, Ning and Hua, Nan and Bai, Haiyan and Huang, Arthur and Guo, Zhishan},
>  booktitle={Advances in Knowledge Discovery and Data Mining: 26th Pacific-Asia Conference, PAKDD 2022, Chengdu, China, May 16--19, 2022, Proceedings, Part I},
>  pages={447--458},
>  year={2022}
>}


You can find the Dataset files on [OneDrive.](https://liveuis-my.sharepoint.com/:u:/g/personal/2926110_uis_no/EShorcz3759Jr_3SEm0B1BUBy4PQKFml6DFbyuwozcor_w?e=3mU0TA)
For further information, contact arezo.shakeri@uis.no 
