#!/usr/bin/env python3
# coding: utf-8

"""
Implement BERT feature extractor.

Authors:
- Bereket A. Yilma <name.surname@uni.lu>
"""


import torch
import pickle
import numpy as np
from string import punctuation
from transformers import BertTokenizer, BertModel
from transformers import logging

logging.set_verbosity_error()

"""
Define a method to obtain the vocab IDs of the tokenized sentence and the segment IDs vector of the sentence.
"""


def tokenize_and_segment(sentence, preprocess=False):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    if preprocess:
        sentence = "".join([c for c in sentence if c not in punctuation]).lower()
    marked_text = "[CLS] " + sentence + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)  # A vector of 1s.
    return indexed_tokens, segments_ids


"""
Method to apply the BERT model on one sentence composed of indexes of tokens and segments.
"""


def get_hidden_states(bert_model, idx_tokens, idx_segments):
    tokens_tensor = torch.tensor([idx_tokens])
    segments_tensor = torch.tensor([idx_segments])
    with torch.no_grad():
        outputs = bert_model(tokens_tensor, segments_tensor)
    hidden_states = torch.stack(outputs.hidden_states, dim=0)
    return hidden_states  # Output shape: [layers, batches, seq_len, features]


""" 
Average the last N layers (last_layers) of each token producing a single N-length vector.
"""


def create_sentence_embeddings(hidden_states, layer_idx=-2):
    """
    Parameters
        hidden_states: output from the BERT model.
        layer_idx: the layer to be used for obtaining the features (default=second to last layer).
    """

    # Remove the batch dim, since we work with individual sentences.
    token_embeddings = torch.squeeze(
        hidden_states, dim=1
    )  # Output shape: [layers, seq_len, features]

    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(
        1, 0, 2
    )  # Output shape: [seq_len, layers, features]

    # Obtain average of features from all tokens.
    sentence_embeddings = torch.mean(hidden_states[layer_idx][0], dim=0)

    # Return embeddings.
    return sentence_embeddings


""" 
Use the create_sentence_embeddings function as feature extractor
 """


def bert_feature_extractor(
    sentences, bert_model, preprocess=False, as_numpy=False, layer_idx=-2
):
    """
    Parameters
        sentences: list of sentences.
        bert_model: pre-trained BERT model.
        as_numpy: return tensors as numpy or not. If False, then dataset[i] is a torch tensor.

    Output
        dataset: a list with N tensors (torch or numpy) with features, with N=len(sentences).
    """

    # Our dataset.
    dataset = []

    # Put the model in "evaluation" mode.
    bert_model.eval()

    # Go through the sentences and get the features of each.
    for sentence in sentences:
        idx_tokens, idx_segments = tokenize_and_segment(sentence, preprocess)
        hidden_bert = get_hidden_states(bert_model, idx_tokens, idx_segments)
        feature_vector = create_sentence_embeddings(hidden_bert, layer_idx)
        dataset.append(feature_vector.numpy() if as_numpy else feature_vector)

    # Return features list.
    return dataset


def extract_features():
    # Load pre-trained model (weights)
    bert_model = BertModel.from_pretrained(
        "bert-base-uncased", output_hidden_states=True
    )

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    bert_model.eval()

    # Read the dataset
    DATASET_FILE = "data/text data/preprocessed/painting_descriptions.pickle"
    dataset_path = DATASET_FILE
    dataset = pickle.load(open(dataset_path, "rb"))
    print(dataset.keys())

    painting_ids = dataset["painting_ids"]
    sentences = dataset["painting_descriptions:"]

    """
    Extract the BERT features and
    Forward the train set to BERT to obtain the feature vectors for each sentence.
    """
    dataset = bert_feature_extractor(
        sentences=sentences, bert_model=bert_model, preprocess=True, as_numpy=True
    )
    dataset = np.asarray(dataset)
    print("Dataset shape:", dataset.shape)

    """
    Generate a new pickle file with the BERT features and the painting indeces.
    """
    DATASET_BERT_FILE = "data/text data/preprocessed/paintings_bert_features.pickle"

    # Dump both lists into a dictionary within a .pickle file.
    file = open(DATASET_BERT_FILE, "wb")
    pickle_rick = {"painting_ids": painting_ids, "bert_features": dataset}
    pickle.dump(pickle_rick, file)
    file.close()
    print("BERT feature extraction done!")


def main():
    extract_features()


if __name__ == "__main__":
    main()
