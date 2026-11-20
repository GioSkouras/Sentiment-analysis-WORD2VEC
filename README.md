# Sentiment Analysis with Word2Vec

This project implements a complete tweet sentiment-classification workflow using
**Word2Vec embeddings** and a small **PyTorch feed-forward neural network**.
The main experiment is contained in [`WORD2VEC.ipynb`](WORD2VEC.ipynb), and the
written analysis is available in [`WORD2VEC-Analysis.pdf`](WORD2VEC-Analysis.pdf).

## What the project does

The notebook:

1. Loads the train, validation, and test CSV files.
2. Explores the data with label-distribution plots, word clouds, and token-frequency
   visualizations.
3. Cleans and tokenizes tweet text.
4. Replaces URLs, mentions, numbers, email addresses, emoticons, and repeated
   characters with meaningful normalized tokens.
5. Detects common bigrams with `gensim.models.Phrases`.
6. Trains a Word2Vec model on the training tweets.
7. Represents each tweet by the mean of its token vectors.
8. Classifies the resulting 400-dimensional vectors with a PyTorch neural network.
9. Evaluates the model using accuracy, precision, recall, F1 score, ROC-AUC, and
   a confusion matrix.
10. Creates a competition-ready `submission.csv` file.

## Model configuration

| Component | Configuration |
| --- | --- |
| Word2Vec vector size | 400 |
| Word2Vec window | 7 |
| Minimum token count | 2 |
| Training epochs | 10 |
| Word2Vec architecture | Skip-gram (`sg=1`) |
| Negative samples | 15 |
| Classifier | Fully connected PyTorch network |
| Hidden layer size | 256 |
| Batch size | 256 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Classifier epochs | 20 |
| Prediction threshold | 0.5 |
| Random seed | 42 |

The classifier uses ReLU activations and dropout, followed by a sigmoid output
for binary sentiment prediction.

## Dataset format

The notebook was designed for the Kaggle dataset mounted at:

```text
/kaggle/input/ai-2-dl-for-nlp-2025-homework-2/
```

It expects these files:

```text
train_dataset.csv
val_dataset.csv
test_dataset.csv
sample_submission.csv
```

The training and validation files must contain:

```text
Text, Label
```

The test file must contain at least:

```text
ID, Text
```

Labels are expected to be binary values (`0` and `1`).

## Running the notebook

### Kaggle

1. Open `WORD2VEC.ipynb` in a Kaggle Notebook.
2. Attach the dataset so it is available under the path shown above.
3. Run all cells from top to bottom.
4. Find the generated predictions in `submission.csv`.

### Local Jupyter installation

Install the main Python dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk \
    contractions emoji gensim wordcloud torch jupyter
```

Then download the NLTK resources used by the notebook:

```python
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

Because the notebook uses Kaggle-specific absolute paths, either run it on
Kaggle or update the three `pd.read_csv(...)` paths in the loading cell to point
to your local dataset directory.

## Output

After inference, the notebook writes:

```text
submission.csv
```

with the following columns:

```text
ID, Label
```

## Reproducibility

The experiment sets the NumPy, Python, and PyTorch random seeds to `42`.
Training Word2Vec only on the training split also helps keep validation and test
data separate from embedding training.

## Repository structure

```text
.
├── WORD2VEC.ipynb          # EDA, preprocessing, training, evaluation, inference
├── WORD2VEC-Analysis.pdf   # Project analysis and discussion
└── README.md               # Project documentation
```

## Notes

The notebook also contains an experimental CNN-based sequence model section.
That code is commented out; the active solution uses averaged Word2Vec vectors
with the feed-forward classifier described above.
