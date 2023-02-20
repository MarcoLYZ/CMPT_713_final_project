# CMPT_713_final_project
## Dataset
The original dataset and cleaned dataset are stored [here](https://drive.google.com/file/d/1aWB_1qGshZQjG_8aJ7H_AteXzdAGy1e4/view?usp=sharing)

``` classes.txt ```: The names of 10 classes.

``` train.txt ```: The original train dataset, which has 4 columns: label, question, content and answer.

``` test.txt ```: The original test dataset.

``` train_cleaned.txt ```: The cleaned train dataset, remove stop words, symbols etc. And combine question, content and answer into text_clean

``` test_cleaned.txt ```:The cleaned test dataset

## Model
### BERT
To train the BERT model, follow the instructions in ``` BERT.ipynb ```, after 6:48:27 training(3 epoches), BERT reachs 76% accuracy on test dataset.

### LSTM
To train the LSTM model, follow the instructions in ``` torch_LSTM.ipynb ```, after 10 epoches training, BiLSTM reachs 69.52% accuracy on test dataset.
