import argparse
import logging
import os

import numpy as np
import tensorflow as tf
from datasets import load_dataset
from transformers import DistilBertTokenizerFast

from base_model import TFDistilBertForQuestionAnswering
from lit_model import LIT
from prep_data import functions_tokenizer, prepare_with_tokenizer_and_encodings

if __name__ == "__main__":
    BERT_DIR = "uncased_L-12_H-768_A-12/"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_to_use", default='base', action='store_true',
                        help="Either 'base' for the base model or 'lit' for the lit model."
                             "Type 'train' for training, 'eval' for evaluating")
    parser.add_argument("--type", default='train', action='store_true',
                        help="Whether to run training or evaluation."
                             "Type 'train' for training, 'eval' for evaluating")
    parser.add_argument("--output_dir", default="out", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="The batch size to use while training the model."
                             "Set it to be as big as possible without crashing anything!")
    parser.add_argument('--debug', action="store_true", default=False)

    args = parser.parse_args()
    print(args)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)

    # Load in the qangaroo dataset
    train_dataset = load_dataset('qangaroo', 'wikihop', split='train')
    val_dataset = load_dataset('qangaroo', 'wikihop', split='validation')
    if args.debug:
        val_dataset = val_dataset.filter(lambda _, ind: ind < 10, with_indices=True)
        train_dataset = train_dataset.filter(lambda _, ind: ind < 10, with_indices=True)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    update, get_encodings_concat = functions_tokenizer(tokenizer)

    train_dataset = train_dataset.map(update)
    val_dataset = val_dataset.map(update)

    train_encodings_concat = get_encodings_concat(train_dataset)
    val_encodings_concat = get_encodings_concat(val_dataset)

    def get_encoding(example):
        # Example ids: WH_train_33, WH_test_4
        _, train, id = example['id'].split('_')
        id = int(id)
        if train == 'train':
            return train_encodings_concat[id]
        elif train == 'dev':
            return val_encodings_concat[id]
        else:
            raise KeyError(f"didn't expect the data set type to be {train}")


    prepare = prepare_with_tokenizer_and_encodings(tokenizer, get_encoding)
    train_dataset = train_dataset.map(prepare)
    val_dataset = val_dataset.map(prepare)


    def get_encodings(dataset):
        encodings = tokenizer([sup.lower() for sup in dataset['supports']], [q.lower() for q in dataset['query']],
                              return_tensors='tf', return_token_type_ids=True, padding=True, truncation=False)
        return encodings


    def update_encodings(dataset, encodings):
        encodings.update({
            'start_positions': dataset['start_positions'],
            'end_positions': dataset['end_positions'],
            'entity_ends': dataset['entity_ends'],
            'to_embed_ind': dataset['to_embed_ind'],
        })
        return encodings


    train_encodings = get_encodings(train_dataset)
    val_encodings = get_encodings(val_dataset)
    # Recalculate the encodings and store everything
    train_encodings = update_encodings(train_dataset, train_encodings)
    val_encodings = update_encodings(val_dataset, val_encodings)

    BASE_KEYS = ['input_ids', 'attention_mask']
    LIT_KEYS = ['input_ids', 'attention_mask', 'entity_ends', 'to_embed_ind']

    train_dataset_for_model = tf.data.Dataset.from_tensor_slices((
        {key: train_encodings[key] for key in LIT_KEYS},
        {key: train_encodings[key] for key in ['start_positions', 'end_positions']}
    ))

    val_dataset_for_model = tf.data.Dataset.from_tensor_slices((
        {key: val_encodings[key] for key in LIT_KEYS},
        {key: val_encodings[key] for key in ['start_positions', 'end_positions']}
    ))

    # Keras will expect a tuple when dealing with labels.
    train_dataset_for_model = train_dataset_for_model.map(lambda x, y: (x, (y['start_positions'], y['end_positions'])))
    val_dataset_for_model = val_dataset_for_model.map(lambda x, y: (x, (y['start_positions'], y['end_positions'])))

    if args.model_to_use == 'base':
        model = TFDistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased', return_dict=True)
    elif args.model_to_use == 'lit':
        model = LIT.from_pretrained('distilbert-base-uncased', return_dict=True)
    else:
        raise NotImplementedError

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.distilbert.return_dict = False  # if using 🤗 Transformers >3.02, make sure outputs are tuples

    batch_size = 16
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss=loss)  # can also use any keras loss fn
    # https://www.tensorflow.org/guide/keras/train_and_evaluate#training_evaluation_from_tfdata_datasets
    model.fit(train_dataset_for_model.shuffle(1000).batch(batch_size), epochs=1,
              validation_data=val_dataset_for_model.batch(batch_size), validation_steps=4)
    model.distilbert.return_dict = True
    print("Saving")
    model.save_pretrained('distilbert-quangaroo')

    window_size = 512

    def best_answer(start_logits, end_logits, data, encoding):
        answers = [answer.lower() for answer in data['candidates']]
        text = (data['supports'] + tokenizer.sep_token + data['query']).lower()
        best_answer_str = ''
        best_answer_score = -100000
        encoding_length = len(encoding.ids)
        offset = window_size - encoding_length - 1
        for answer in answers:
            start = text.find(answer, len(data['supports']))
            while start != -1:
                end = start + len(answer)
                start_token = encoding.char_to_token(start)
                end_token = encoding.char_to_token(end - 1)
                if start_token is None or end_token is None:
                    break
                score = start_logits[start_token + offset] + end_logits[end_token + offset]
                if score > best_answer_score:
                    best_answer_score = score
                    best_answer_str = answer
                # Check the next occurance of this answer
                start = text.find(answer, start + 1)
        return best_answer_str


    def score_prediction(pred, encodings, dataset):
        start_logits = pred['start_logits']
        end_logits = pred['end_logits']
        answers = []
        for i in range(len(start_logits)):
            answers.append(best_answer(start_logits[i], end_logits[i], dataset[i], encodings[i]))
        # print(list(zip(np.array(answers), dataset['answer'], dataset['candidates'])))
        correct = np.count_nonzero(np.array(answers) != dataset['answer'])
        return correct / len(answers)

    # Predict stuff with the model
    pred_train = model.predict(train_dataset_for_model.batch(batch_size))
    score_train = score_prediction(pred_train, train_encodings_concat, train_dataset)
    print(f"Train Error: {score_train}")

    pred_val = model.predict(val_dataset_for_model.batch(batch_size))
    score_test = score_prediction(pred_val, val_encodings_concat, val_dataset)
    print(f"Test Error: {score_test}")

