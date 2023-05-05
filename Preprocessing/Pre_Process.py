#!/usr/bin/env python3
# coding: utf-8

"""
Implement text preprocessing for Paitings descriptions.

Authors:
- Bereket A. Yilma <name.surname@uni.lu>
"""

import re
import spacy
import pickle
import pandas as pd

"""This file is used to perform text cleaning on a painting description dataset.
    Multiple Natural Language Processing (NLP) techniques are used in order remove
    meaningless information from the dataset."""

path_to_dataset = "data/text data/ng-dataset.xlsx"


def clean_dataframe(df):
    """We apply a transformation to each row of the dataframe using the function replace_break_balise
    Input:
            df: dataframe of paintings
    Output:
            df: dataframe were break lines are removed
    """
    pd.set_option("display.max_colwidth", 1000)
    df["merged_description"] = df["merged_description"].apply(replace_break_balise)
    return df


def replace_break_balise(text):
    """From the index, returns the painting ID from the paintings dataframe
    Input:
            text
    Output:
            text: breaklines + <p> </p> balises,... removed
    """
    text = text.replace("\n", "")
    text = text.replace("<p>", "")
    text = text.replace("</p>", "")
    text = text.replace("<br>", "")
    text = text.replace("<br />", "")
    text = text.replace("< p >", "")
    text = text.replace("</p >", "")
    return text


def dataframe2text(df):
    """We transform the dataframe into a raw text format
    Input:
            df: dataframe of paintings
    Output:
            text
    """
    text = df["merged_description"].to_string(index=False)
    text = re.sub(" +", " ", text).replace("<br>", "")
    text = text.replace("\n ", " \n")
    return text


def add_stopwords(text, nlp):
    my_stop_words = [
        "'s",
        "be",
        "work",
        "painting",
        "early",
        "small",
        "know",
        "appear",
        "depict",
        "tell",
        "type",
        "apparently",
        "paint",
        "show",
        "probably",
        "picture",
        "left",
        "right",
        "date",
        "suggest",
        "hold",
        "de",
        "see",
        "represent",
        "paint",
    ]
    for stopword in my_stop_words:
        nlp.vocab[stopword].is_stop = True
    doc = nlp(text)
    return doc, nlp


def clean_text(text, nlp):
    """Remove stopwords, punctuation and numbers from the text"""
    doc = nlp(text)
    article = ""
    for w in doc:
        # If it's not a space or a stop word or a punctuation mark, add it to our article.
        if (
            not w.is_stop
            and not nlp.vocab[w.lemma_].is_stop
            and not w.is_punct
            and not w.like_num
        ):
            article += w.lemma_ + " "  # Use the lematized version of the word.
    return article


# We wrap all the preprocessing in one function for future usability


def preprocess():
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 10000000
    df = pd.read_excel(path_to_dataset)
    df = clean_dataframe(df)
    doc, nlp = add_stopwords(dataframe2text(df), nlp)
    # df = df[['painting_id','merged_description']]
    sentences = []
    painting_ids = []
    for i in range(len(df)):
        idx = df["painting_id"][i]
        cleaned = clean_text(df["merged_description"][i], nlp)
        cleaned = cleaned.replace("   ", " ")
        sentences.append(cleaned)
        painting_ids.append(idx)

    # Dump both lists into a dictionary within a .pickle file.
    PREPROCESSED_FILE = "data/text data/preprocessed/painting_descriptions.pickle"
    file = open(PREPROCESSED_FILE, "wb")
    pickle_rick = {"painting_ids": painting_ids, "painting_descriptions:": sentences}
    pickle.dump(pickle_rick, file)
    file.close()
    print("Preprocessing done!")


def main():
    preprocess()


if __name__ == "__main__":
    main()
