# SpringBoard

Project description
•	Binary text classification is supervised learning problem in which we try to predict whether a piece of text of sentence falls into one category or other.
•	This project is focused on implementation of one of the most widely used NLP Task " Text classification " using BERT Language model and Pytorch framework.
Applications of BERT
BERT can be used for a variety of NLP tasks such as Text Classification or Sentence Classification , Semantic Similarity between pairs of Sentences , Question Answering Task with paragraph , Text summarization etc. but there are some NLP task where BERT can’t be used due to its bidirectional information retrieval property. Some of those task are Machine translation ,Text Generator , Normal Question answering task etc. because it needs to get the information from both sides. These application generally achieved by fine tuning the BERT model for our task. Fine tuning is little bit analogous to Transfer learning in which we take a pre-trained model and retrained it on our small dataset by freezing some original layers and adding some new ones , but in fine tuning there is no concept adding or freezing layers we can simply training the model on similar dataset ,it is a form of transfer learning.

Implementation of Binary Text Classification

We are going to use pre-trained BERT model for fine tuning so let's first install transformer from Hugging face library ,because it  provide us pytorch interface for the BERT model .Instead of using a model from variety of pre-trained transformer, library also provides with models for specific task so we are going use " BertForSequenceClassification " for this task. Next step in process is to loading the dataset.

Data Preparation
•	Transforming the Dataset: Next step is to getting BERT tokenizer as we have to split the sentences into token and mapped these token to the BERT tokenizer vocabulary to feed into the BERT model.
•	Data : The real and fake news dataset from Kaggle
•	Tokenization : bert-base-uncased
•	Input size : 128 tokens 
•	Batch size : 16


Model

•	Pretrained Bert for sequence classification
•	Optimizer: Adam
•	Epochs: 5
•	Loss function: Binary cross entropy

 









Evaluation

Using test data set calculate accuracy with prediction against true labels

 









Conclusion

Fine-tuning BERT performs extremely well on our dataset and is very  simple to implement.  The accuracy is about 96 percent for the data e used here.

