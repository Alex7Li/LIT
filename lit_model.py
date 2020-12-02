# https://huggingface.co/transformers/_modules/transformers/modeling_distilbert.html#DistilBertForQuestionAnswering
from transformers import DistilBertConfig, BatchEncoding
from transformers.modeling_tf_distilbert import (
    TFEmbeddings, TFDistilBertPreTrainedModel, TFTransformerBlock
)
from transformers.modeling_tf_outputs import (
    TFQuestionAnsweringModelOutput, TFBaseModelOutput
)
from transformers.modeling_tf_utils import (
    shape_list,
    get_initializer,
    TFQuestionAnsweringLoss,
)

from helpers import *


class TFTransformer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.n_layers = config.n_layers
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions

        self.layer = [TFTransformerBlock(config, name="layer_._{}".format(i)) for i in range(config.n_layers)]
        self.entity_memory = EntityMemory(config, name="EntityMemory")
        self.layer_norm = tf.keras.layers.LayerNormalization(config, name="LayerNorm")

    def call(self, x, attn_mask, head_mask, output_attentions, output_hidden_states,
             return_dict, entity_matrix, entity_ends, to_embed_ind, training=False, do_all=True, optimizer=None):
        """
        Parameters
        ----------
        x: tf.Tensor(bs, seq_length, dim)
            Input sequence embedded.
        attn_mask: tf.Tensor(bs, seq_length)
            Attention mask on the sequence.

        Outputs
        -------
        hidden_state: tf.Tensor(bs, seq_length, dim)
            Sequence of hidden states in the last (top) layer
        all_hidden_states: Tuple[tf.Tensor(bs, seq_length, dim)]
            Tuple of length n_layers with the hidden states from each layer.
            Optional: only if output_hidden_states=True
        all_attentions: Tuple[tf.Tensor(bs, n_heads, seq_length, seq_length)]
            Tuple of length n_layers with the attention weights from each layer
            Optional: only if output_attentions=True
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            layer_outputs = layer_module(hidden_state, attn_mask, head_mask[i], output_attentions, training=training)
            hidden_state = layer_outputs[-1]

            if output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1, f"Incorrect number of outputs {len(layer_outputs)} instead of 1"

            if len(self.layer) // 2 == i and entity_ends is not None:
                # It's time for the entity memory layer, the main contribution of this repo!
                outputs, entity_matrix = self.entity_memory(hidden_state, entity_matrix,
                                                            entity_ends, to_embed_ind, training=training,
                                                            optimizer=optimizer)
                hidden_state += outputs
                hidden_state = self.layer_norm(hidden_state)
                if not do_all:
                    print("Exit early!")
                    # There's no need to go through the rest of the hidden layers
                    break
                else:
                    print("Predicted!")

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)
        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states, all_attentions] if v is not None)
        return TFBaseModelOutput(
            last_hidden_state=hidden_state, hidden_states=all_hidden_states, attentions=all_attentions
        )


# @keras_serializable
class LITMainLayer(tf.keras.layers.Layer):
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
            entity_matrix=None,
            training=False,
            optimizer=None
    ):
        assert (isinstance(inputs, (dict, BatchEncoding)))
        return_dict = inputs.get("return_dict", return_dict)

        return_dict = return_dict if return_dict is not None else self.return_dict

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = [None] * self.num_hidden_layers

        input_shape = shape_list(inputs.get('input_ids'))
        start = inputs.get("start")
        end = inputs.get("end")
        in_len = input_shape[1]
        input_ids = inputs.get('input_ids')[:, start:end]
        attention_mask = get_from_map_subrange(inputs, 'attention_mask', start, end, offset=False)
        input_embeds = get_from_map_subrange(inputs, 'input_embeds', start, end, offset=False)
        to_embed_ind = get_from_map_subrange(inputs, 'to_embed_ind', start, end, offset=False)
        entity_ends = get_from_map_subrange(inputs, 'entity_ends', start, end, offset=True)

        assert input_ids is not None
        input_shape = shape_list(input_ids)
        assert inputs_embeds is None
        if attention_mask is None:
            attention_mask = tf.ones(input_shape)  # (bs, seq_length)

        attention_mask = tf.cast(attention_mask, dtype=tf.float32)
        embedding_output = self.embeddings(input_ids, inputs_embeds=inputs_embeds)  # (bs, seq_length, dim)
        tfmr_output = self.transformer(
            embedding_output,
            attention_mask,
            head_mask,
            None,  # output_attentions,
            None,  # output_hidden_states,
            return_dict,
            entity_matrix,
            entity_ends,
            to_embed_ind,
            training=training,
            optimizer=optimizer,
            do_all=end == in_len  # We only care about the return value in the last pass through
        )
        return tfmr_output  # last-layer hidden-state, (all hidden_states), (all attentions)


class EntityMemory(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # Same as for the entity as experts paper
        self.entity_dim = 256
        self.dim = config.dim
        self.to_entity_embed = tf.keras.layers.Dense(self.entity_dim, name='to_entity')
        self.from_entity_embed = tf.keras.layers.Dense(config.dim, name='from_entity')
        self.merge_entity_embed = tf.keras.layers.Dense(self.entity_dim, name='merge_entity')

    def call(self, x, entity_matrix_batch, entity_ends_batch, to_embed_ind_batch, training=False, optimizer=None):
        """
        Parameters
        ----------
        x: tf.Tensor(bs, seq_length, dim)
            Input sequence embedded.
        entity_matrix_batch: tf.RaggedTensor(bs, (n_entities), entity_dim)
            The matrix of entities in the context
        entity_ends_batch: tf.Tensor(bs, seq_length)
            entity_ends[bs][i] = -1 if no entity starts here, or the index of the end of the entity here
        to_embed_ind_batch: tf.Tensor(bs, seq_length)
            The index of the entity in the entity matrix referred to at this point of the sequence, or -1
        Outputs
        -------
        hidden_state: tf.Tensor(bs, seq_length, dim)
            The hidden state result after passing through this layer
            Sequence of hidden states in the last (top) layer
        updated_embed_matrix: tf.RaggedTensor(bs, (n_entities), entity_dim)
            An updated matrix of entity embeddings
        """
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        cross_entropy_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

        # TF2.0 seems to have problems updating slices of variable tensor,
        # https://stackoverflow.com/questions/39157723/how-to-do-slice-assignment-in-tensorflow/43139565#43139565
        # So we use a python list
        entity_loss = [[tf.Variable(0.0) for i in range(seq_length)] for j in range(batch_size)]
        update_embed_matrix = [[tf.Variable(entity_matrix_batch[i][j], shape=(self.entity_dim))
                                for j in range(entity_matrix_batch[i].shape[0])] for i in range(batch_size)]
        zero_int = tf.constant(0)

        def entity_update(inputs):
            i, entity_ends, to_embed_ind = inputs
            # Get the ith entity matrix as a tensor, it won't be ragged anymore.
            entity_matrix = tf.Variable(entity_matrix_batch[i])

            # print(shape_list(entity_matrix)) # (n_entities, entity_dim)
            def get_embed_else_zero(inputs):
                j, entity_end, embed_ind = inputs

                def get_embed():
                    concat_context = tf.expand_dims(tf.concat((x[i][j], x[i][entity_end]), axis=0),
                                                    axis=0)  # (1, 2*dim)
                    entity_embed = self.to_entity_embed(concat_context)  # (1, entity_dim, )
                    embed_distances = tf.tensordot(entity_embed, update_embed_matrix[i],
                                                   axes=[[1], [1]])  # (1, n_entities,)
                    embed_dist_softmax = tf.nn.softmax(embed_distances)  # (1, n_entities,)

                    not_first_time = tf.math.greater(
                        tf.math.count_nonzero(update_embed_matrix[i][embed_ind], dtype=tf.int32), zero_int)
                    condition = tf.math.logical_and(training, not_first_time)
                    entity_loss[i][j].assign(tf.cond(training,
                                                     # We compute the cross entropy loss with the current prediction. Maybe shouldn't
                                                     # be here for the first time
                                                     lambda: cross_entropy_loss(
                                                         embed_ind,  # The entity value is the true prediction.
                                                         embed_dist_softmax[0]  # Our 'prediction' of the right entity
                                                     ),
                                                     lambda: tf.zeros(shape=())
                                                     ))
                    entity_rep = tf.tensordot(embed_dist_softmax, update_embed_matrix[i],
                                              axes=[[1], [0]])  # (1, entity_dim,)
                    # Update the entity matrix with the embedding
                    concat_embeds = tf.expand_dims(
                        tf.concat([update_embed_matrix[i][embed_ind], entity_embed[0]], axis=0),
                        axis=0)
                    # print(concat_embeds.shape) # (1, 2 * entity_dim)
                    updated_entity = self.merge_entity_embed(concat_embeds)  # (1, entity_dim)

                    update_embed_matrix[i][embed_ind] = update_embed_matrix[i][embed_ind].assign(
                        tf.squeeze(updated_entity))

                    return tf.squeeze(self.from_entity_embed(entity_rep))

                rval = tf.cond(embed_ind > 0, get_embed, lambda: tf.zeros(shape=(self.dim)))
                return rval

            output_signature_inner = tf.TensorSpec(tf.shape(x[0][0]))
            # The output of this layer applied to this batch element
            entity_output_values = tf.map_fn(
                get_embed_else_zero, (tf.range(seq_length), entity_ends, to_embed_ind),
                fn_output_signature=output_signature_inner)
            return entity_output_values

        with tf.GradientTape() as tape:
            trainable_variables = self.trainable_variables
            tape.watch(trainable_variables)
            # Map the function over each batch element. (tf.vectorized_map may be faster?)
            layer_output = tf.map_fn(
                entity_update, (tf.range(batch_size), entity_ends_batch, to_embed_ind_batch),
                fn_output_signature=tf.TensorSpec(tf.shape(x[0])))

            print("Minimize time")
            gradients = tape.gradient(entity_loss, trainable_variables)
            # Bug - these are None when there are trainable variables. Perhaps there's a non-differentiable
            # function hiding somewhere above?
            print(gradients)
            optimizer.apply_gradients(zip(gradients, trainable_variables))
            print("We did it boys")
        return layer_output, update_embed_matrix


class LIT(TFDistilBertPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.entity_dim = 256
        self.distilbert = LITMainLayer(config, name="lit")
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        assert config.num_labels == 2, f"Incorrect number of labels {config.num_labels} instead of 2"
        self.dropout = tf.keras.layers.Dropout(config.qa_dropout)

    def compile(self, **kwargs):
        super(LIT, self).compile(**kwargs)
        self.optimizer = kwargs['optimizer']
        self.loss_fn = kwargs['loss']

    """
    def train_step(self, data):
        This method can be overridden to support custom training logic.
        This method is called by `Model.make_train_function`.

        This method should contain the mathemetical logic for one step of training.
        This typically includes the forward pass, loss calculation, backpropagation,
        and metric updates.

        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_train_function`, which can also be overridden.

        Arguments:
        data: A nested structure of `Tensor`s.

        Returns:
        A `dict` containing values that will be passed to
        `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
        values of the `Model`'s metrics are returned. Example:
        `{'loss': 0.2, 'accuracy': 0.7}`.
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided. These utilities will be exposed
        # publicly.
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
        y_pred = self(x, training=True)
        loss = self.compiled_loss(
            y, y_pred, sample_weight, regularization_losses=self.losses)
        # For custom training steps, users can just write:
        #   trainable_variables = self.trainable_variables
        #   gradients = tape.gradient(loss, trainable_variables)
        #   self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        # The _minimize call does a few extra steps unnecessary in most cases,
        # such as loss scaling and gradient clipping.
        _minimize(self.distribute_strategy, tape, self.optimizer, loss,
                self.trainable_variables)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}
    """

    def train_step(self, inputs):
        x, y = inputs
        input_ids = x['input_ids']
        batch_size = shape_list(input_ids)[0]
        in_len = shape_list(input_ids)[1]
        entity_loss = 0
        if 'entity_ends' in x:
            row_lengths = get_row_lengths(x['to_embed_ind'])
            x['entity_matrix'] = init_entity_matrix(batch_size, row_lengths, self.entity_dim)
        else:
            x['entity_matrix'] = None
        for start, end in split_into_intervals(in_len, 512):
            x["start"] = start
            x["end"] = end
            if end == in_len:
                x['start_positions'] = y[0]
                x['end_positions'] = y[1]
                entity_loss, prediction_loss = self(x, training=True, optimizer=self.optimizer)
            else:
                num_entities = tf.math.count_nonzero(x["entity_ends"][:, start:end])
                if num_entities < 2:
                    # No point in running the layers, the loss will be 0
                    continue
                entity_loss += self.call(x, training=True, optimizer=self.optimizer)
        return {'prediction_loss': prediction_loss, 'entity_loss': entity_loss}

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
            entity_matrix=None,
            training=False,
            optimizer=None,
    ):
        return_dict = return_dict if return_dict is not None else self.distilbert.return_dict
        if isinstance(inputs, (tuple, list)):
            start_positions = inputs[7] if len(inputs) > 7 else start_positions
            end_positions = inputs[8] if len(inputs) > 8 else end_positions
            if len(inputs) > 7:
                inputs = inputs[:7]
        elif isinstance(inputs, (dict, BatchEncoding)):
            start_positions = inputs.get("start_positions", start_positions)
            end_positions = inputs.get("end_positions", start_positions)
            entity_matrix = inputs.get("entity_matrix", entity_matrix)
        distilbert_output = self.distilbert(
            inputs,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
            entity_matrix=entity_matrix,
            optimizer=optimizer,
        )
        hidden_states = distilbert_output[0]  # (bs, max_query_len, dim)
        hidden_states = self.dropout(hidden_states, training=training)  # (bs, max_query_len, dim)
        logits = self.qa_outputs(hidden_states)  # (bs, max_query_len, 2)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)

        loss = tf.reduce_sum(self.distilbert.losses)
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions,
                      "end_position": end_positions}
            loss['prediction'] = tf.reduce_sum(loss, self.compute_loss(labels, (start_logits, end_logits)))

        if not return_dict:
            output = (start_logits, end_logits) + distilbert_output[1:]
            return ((loss,), output) if loss is not None else output

        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )
