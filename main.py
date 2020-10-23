import argparse
import logging
import os

from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering

from process_data import load_train, load_summaries


def train(model, tokenizer, x, y, summaries):
    for document_id, question in x:


if __name__ == "__main__":
    BERT_DIR = "uncased_L-12_H-768_A-12/"
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default='train', action='store_true',
                        help="Whether to run training or evaluation."
                             "Type 'train' for training, 'eval' for evaluating")
    parser.add_argument("--output_dir", default="out", type=str,
                        help="The output directory where the model checkpoints will be written.")
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

    train_x, train_y, test_x, test_y = load_train()
    test_sum, train_sum = load_summaries()
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        'distilbert-base-cased')

    # Load the model in training mode
    model = DistilBertForQuestionAnswering()
    model.train()
    train(None, tokenizer, train_x, train_y, train_sum)
