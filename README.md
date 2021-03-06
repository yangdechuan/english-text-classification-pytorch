# english-text-classification-pytorch
A PyTorch Implementation of English Text Classification.

## Requirement
* python3
* pytorch >= 0.4
    * Follow setup steps in https://pytorch.org/
* nltk
* numpy
* pandas

## Usage
* Step1: Put train and test data to `./data/` folder.
* Step2: Download [google word2vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) to `./resources/` folder and modify `embedding_file` in `settings.ini`.
* Step3: Adjust hyper parameters in `settings.ini` if necessary.
* Step4: Generate vocabulary file to the `./results/` folder.
```
python main.py --make-vocab
```
* Step5: Train model.
    * Model will be saved in `./models/` folders
    * Default Use all gpu to train mdoel. If you only want to use gpu 0, you can add `CUDA_VISIBLE_DEVICES=0` before the following instruction.
```
python main.py --do-train
```
* Step6: Predict labels with saved model.
    * `epoch_idx` is the saved model's epoch id.
    * labels will be saved in `./results/` folder.
```
python main.py --do-predict --epoch-idx 10
```

## File Description
* `cnn.py` includes CNN text classifier.
* `lstmattention.py` includes LSTM+Attention text classifier.
* `utils.py` contains function and class regarding loading and batching data.
* `main.py` for preprocess, train or predict.
* `data/`: dataset dir
* `models/`: saved models dir
* `results/`: vocab dict file and predict result file dir
* `resources`: word2vec file