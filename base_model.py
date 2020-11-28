# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team, The Google AI Language Team and Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
 TF 2.0 DistilBERT model
"""

import tensorflow as tf
from transformers.configuration_distilbert import DistilBertConfig
from transformers.modeling_tf_distilbert import (
    TFTransformer, TFEmbeddings
)
from transformers.modeling_tf_outputs import (
    TFQuestionAnsweringModelOutput,
)
from transformers.modeling_tf_utils import (
    TFQuestionAnsweringLoss,
    TFPreTrainedModel,
    get_initializer,
    shape_list,
)
from transformers.tokenization_utils import BatchEncoding

from helpers import dict_subset


class TFDistilBertMainLayer(tf.keras.layers.Layer):
    config_class = DistilBertConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.num_hidden_layers = config.num_hidden_layers
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict

        self.embeddings = TFEmbeddings(config, name="embeddings")  # Embeddings
        self.transformer = TFTransformer(config, name="transformer")  # Encoder

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
        self.embeddings.vocab_size = value.shape[0]

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    def call(
            self,
            inputs,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            training=False,
    ):
        assert isinstance(inputs, (dict, BatchEncoding));
        in_len = inputs['input_ids'].shape[1]
        inputs = dict_subset(inputs, in_len - 512, in_len)
        input_ids = inputs.get("input_ids")
        if attention_mask is not None:
            attention_mask = inputs.get("attention_mask", attention_mask)
        head_mask = inputs.get("head_mask", head_mask)
        inputs_embeds = inputs.get("inputs_embeds", inputs_embeds)
        output_attentions = inputs.get("output_attentions", output_attentions)
        output_hidden_states = inputs.get("output_hidden_states", output_hidden_states)
        return_dict = inputs.get("return_dict", return_dict)
        assert len(inputs) <= 7, "Too many inputs."

        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = tf.ones(input_shape)  # (bs, seq_length)
        attention_mask = tf.cast(attention_mask, dtype=tf.float32)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.num_hidden_layers

        embedding_output = self.embeddings(input_ids, inputs_embeds=inputs_embeds)  # (bs, seq_length, dim)
        tfmr_output = self.transformer(
            embedding_output,
            attention_mask,
            head_mask,
            output_attentions,
            output_hidden_states,
            return_dict,
            training=training,
        )
        return tfmr_output  # last-layer hidden-state, (all hidden_states), (all attentions)


class TFDistilBertPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = DistilBertConfig
    base_model_prefix = "distilbert"


class TFDistilBertForQuestionAnswering(TFDistilBertPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.distilbert = TFDistilBertMainLayer(config, name="distilbert")
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        assert config.num_labels == 2, f"Incorrect number of labels {config.num_labels} instead of 2"
        self.dropout = tf.keras.layers.Dropout(config.qa_dropout)

    def call(
            self,
            inputs=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            start_positions=None,
            end_positions=None,
            training=False,
    ):
        """
        start_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.distilbert.return_dict
        if isinstance(inputs, (tuple, list)):
            start_positions = inputs[7] if len(inputs) > 7 else start_positions
            end_positions = inputs[8] if len(inputs) > 8 else end_positions
            if len(inputs) > 7:
                inputs = inputs[:7]
        elif isinstance(inputs, (dict, BatchEncoding)):
            start_positions = inputs.pop("start_positions", start_positions)
            end_positions = inputs.pop("end_positions", end_positions)
        distilbert_output = self.distilbert(
            inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        hidden_states = distilbert_output[0]  # (bs, max_query_len, dim)
        hidden_states = self.dropout(hidden_states, training=training)  # (bs, max_query_len, dim)
        logits = self.qa_outputs(hidden_states)  # (bs, max_query_len, 2)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        loss = None
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            loss = self.compute_loss(labels, (start_logits, end_logits))

        if not return_dict:
            output = (start_logits, end_logits) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )