import wikipediaapi
import pandas as pd
import nltk
nltk.download('punkt')
import numpy as np
import re



def build_df_from_words(words):
    """
    input:
        words: list[str]
            list of words from which to get the abstracts
    output:
        df: pandas.DataFrame
            Dataframe with 2 columns: title and abstract. Title is the word,
            while abstract is its Wikipedia abstract as a string. 
    """
    wiki = wikipediaapi.Wikipedia('en')
    abstract_list = [{"title": word, "abstract": wiki.page(word).summary}
               for word in words]
    df = pd.DataFrame(abstract_list)
    return df



def build_embedding(df):
    """
    Build embedding matrix from dataframe of titles and abstracts
    input:
        df: pandas.DataFrame
            dataframe with 2 columns: title which contains the word,
            and abstract whose elements are wikipedia abstract strings.
    output:
        embedding: np.array
            ndarray of shape (len(df), len(df), nb_features) that embedds
            every word of the dataframe's title column.
    """
    def _tokenize(abstract_str):
        """
        Tokenize a abstract string as a lit of words.
        input: str
        output: list[str]
        """
        abstract_list = nltk.word_tokenize(abstract1_str)
        return abstract_list

    nb_features = 10
    embedding = np.zeros((len(df), len(df), nb_features))

    for i1, row1 in df.iterrows():
        for i2, row2 in df.iterrows():
            if i1 == i2:
                continue
            word1, abstract1_str = row1["title"].lower(), row1["abstract"].lower()
            word2, abstract2_str = row2["title"].lower(), row2["abstract"].lower()
            
            # Transform abstracts strings into lists of tokens
            abstract1 = _tokenize(abstract1_str)
            abstract2 = _tokenize(abstract2_str)

            # Surface features
            # Not implemented

            # Word N-gramms features
            # Not implemented
            
            # 3.2.2 Wikipedia abstract features
            # Il faut créer un pandas avec les abstracts des articles contenant l'un des mots.
            embedding[i1, i2, 0] = 1 if word1 in abstract2 else 0

            # Presence and distance
            if word1 in abstract2 and word2 in abstract2:
                # distance = abs(abstract2.index(word1) - abstract2.index(word2))
                distance = min(
                    [abs(pos_word1 - pos_word2)
                        for (pos_word1, pos_word2)
                        in zip(
                            [pos_word1 for pos_word1, word in enumerate(abstract2)
                                if word == word1],
                            [pos_word2 for pos_word2, word in enumerate(abstract2)
                                if word == word2])
                    ])
                embedding[i1, i2, 1] = 1 if distance < 20 else 0

            # count
            
            # min distance

            # Patern
    return embedding



def process_embedding_from_words(words):
    df = build_df_from_words(words)
    embedd = build_embedding(df)
    return embedd