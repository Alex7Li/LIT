import numpy as np



def functions_tokenizer(tokenizer):
    def update(example):
        """
        Update the supports of a training example to be just a single array,
        and add the possible answers to the query
        """
        # Not sure we should join with the separating token but there's nothing better
        example['supports'] = tokenizer.sep_token.join(example['supports'])
        question = example['query'].split(" ")[0]
        subject = " ".join(example['query'].split(" ")[1:])
        query = subject + " " + " ".join(question.split("_"))
        example['query'] = query + tokenizer.sep_token + tokenizer.sep_token.join(example['candidates'])
        return example

    def get_encodings_concat(dataset):
        # We use the + sign to concateate the encodings instead of doing it as a pair because the tokenizer's char to token method doesn't
        # work on the second parameter. Later we make new encodings.
        encodings = tokenizer(
            [sup.lower() + tokenizer.sep_token + q.lower() for sup, q in zip(dataset['supports'], dataset['query'])],
            return_tensors='tf', return_token_type_ids=True, padding=True, truncation=False)
        return encodings

    return update, get_encodings_concat


window_size = 512


def earliest_entity_ind(mentions):
    # I think first_mention_ind is always 0 but just to be safe
    assert 0 == np.argmin([mention.start_char for mention in mentions])
    return mentions[0].start_char


class TextInstance(object):
    """
    This is just to be compatible with the spacy API
    """

    def __init__(self, start, end):
        self.start_char = start
        self.end_char = end


def find_instances_of_word_in_text(word, text):
    instances = []
    start = 0
    start = text.find(word, start)
    while start != -1:
        end = start + len(word)
        instance = TextInstance(start, end)
        instances.append(instance)
        start = text.find(word, start + 1)
    return instances

"""
def clusters_with_nlp(text):
    # Get clusters from the coreference library
    clusters_nlp = nlp(text)._.coref_clusters
    return [cluster.mentions for cluster in clusters_nlp]
"""

def clusters_from_candidates(text, candidates):
    # Get clusters by finding elements with the names of the objects we wanna classify
    clusters_candidates = []
    for candidate in candidates:
        clusters_candidates.append(find_instances_of_word_in_text(candidate, text))
    return clusters_candidates


def prepare_with_tokenizer_and_encodings(tokenizer, get_encoding):
    def update_st_end(example, encoding):
        """
        Compute the correct start/end positions
        """
        # Find the start/end positions for the answer in the supports.
        start = (example['supports'] + tokenizer.sep_token + example['query']).lower() \
            .find(example['answer'].lower(), len(example['supports']))
        assert start != -1
        end = start + len(example['answer']) - 1

        # We only predict on the last 512 tokens, so let's adjust the start and end positions to be
        # in that window.
        encoding_length = len(encoding.ids)
        start_token_ind = encoding.char_to_token(start) + window_size - encoding_length - 1
        end_token_ind = encoding.char_to_token(end) + window_size - encoding_length - 1
        # The padding side is left and we search from the start of the supports. Since there's never more
        # than 512 tokens of support, this should always work
        assert start_token_ind >= 0
        return start_token_ind, end_token_ind

    def find_corefs(example, encoding):
        """
        find coreferences in the text
        returns
        entity_ends[i] = the end of the entity starting at token i, or -1
        to_entity_ind[i] = the index corresponding to the entity starting at token i, or -1
        """
        n_tokens = len(encoding.ids)
        entity_ends = [0 for _ in range(n_tokens)]
        to_entity_ind = [0 for _ in range(n_tokens)]
        # the first entity is at index 1, not 0. so that sparse tensors work easily.
        cur_entity_ind = 1

        # find clusters with 2 different methods.
        clusters = clusters_from_candidates((example['supports'] + tokenizer.sep_token + example['query']).lower(),
                                            example['candidates'])  # + clusters_with_nlp(example['supports'])
        for cluster in sorted(clusters, key=earliest_entity_ind):
            for mention in cluster:
                st = encoding.char_to_token(mention.start_char)
                if entity_ends[st] != 0:
                    continue  # this start position has already been used in another entity
                end = encoding.char_to_token(mention.end_char - 1)
                assert st != None and end != None
                entity_ends[st] = end
                to_entity_ind[st] = cur_entity_ind
            cur_entity_ind += 1
        return entity_ends, to_entity_ind

    def prepare(example):
        encoding = get_encoding(example)
        example['start_positions'], example['end_positions'] = update_st_end(example, encoding)
        assert example['start_positions'] is not None
        example['entity_ends'], example['to_embed_ind'] = find_corefs(example, encoding)
        return example
    return prepare


