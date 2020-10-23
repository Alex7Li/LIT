import pandas as pd

def tokenize_array(supports):
    tokens = []
    for support in supports:
        tokens.extend(tokenizer.tokenize(support))
        tokens.append(tokenizer.sep_token)
    return tokens


def load_summaries():
    summaries = pd.read_csv('narrativeqa/summaries.csv', sep=',')
    train = summaries.loc[summaries['set'] == 'train']
    test = summaries[summaries['set'] == 'test']
    return train[['document_id', 'summary']], test[['document_id', 'summary']]


def load_train():
    """
    Load data in from narrativeqa.
    """
    qa = pd.read_csv('narrativeqa/qaps.csv', sep=',')

    train = qa.loc[qa['set'] == 'train']
    test = qa[qa['set'] == 'test']

    train_x = train[['document_id', 'question']]
    train_y = train[['answer1', 'answer2']]
    test_x = test[['document_id', 'question']]
    test_y = test[['answer1', 'answer2']]

    return train_x, train_y, test_x, test_y

