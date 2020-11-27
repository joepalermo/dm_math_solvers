import os
from preprocessing import load_tokenizers
from transformer.transformer_utils import positional_encoding, create_masks, scaled_dot_product_attention, \
    point_wise_feed_forward_network
import tensorflow as tf
import numpy as np
import datetime
import time


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, params):
        super(Transformer, self).__init__()
        self.char2idx, self.idx2char = load_tokenizers('output/tokenizers')

        # Model config
        self.params = params
        num_layers = params.num_layers
        d_model = params.d_model
        num_heads = params.num_heads
        dff = params.dff
        input_vocab_size = params.vocab_size
        target_vocab_size = params.vocab_size
        pe_input = params.questions_max_length
        pe_target = params.answer_max_length
        attention_dropout = params.attention_dropout

        # Training loop config
        self.learning_rate = CustomSchedule(params.d_model)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean('train_accuracy')
        self.val_loss = tf.keras.metrics.Mean('val_loss')
        self.val_accuracy = tf.keras.metrics.Mean('val_accuracy')

        # Instantiate the model
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, attention_dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, attention_dropout)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

        # tensorboard writers
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        val_log_dir = 'logs/gradient_tape/' + current_time + '/val'
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def train(self, params, train_ds, val_ds, logger):
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        checkpoint_path = params.checkpoint_dir
        ckpt = tf.train.Checkpoint(transformer=self, optimizer=self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

        val_acc_list = []
        best_val_accuracy = 0
        best_val_loss = np.inf
        for epoch_i in range(params.num_epochs):
            start = time.time()
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.val_loss.reset_states()
            self.val_accuracy.reset_states()
            accuracy_list = []
            for batch, (input_batch, target_batch) in enumerate(train_ds):
                probs_batch = self.train_step(input_batch, target_batch)
                preds_batch = tf.argmax(probs_batch, axis=-1, output_type=tf.int32)
                accuracy = self.accuracy(target_batch, preds_batch)
                self.train_accuracy(accuracy)
                accuracy_list.append(accuracy)
                if batch % self.params.batches_per_inspection == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch_i + 1, batch, self.train_loss.result(), np.mean(accuracy_list[-50:])))
                    self.inspect_inference(input_batch, target_batch, logger, num_to_inspect=3)

            # at end of epoch write for tensorboard
            with self.train_summary_writer.as_default():
                tf.summary.scalar('train_loss', self.train_loss.result(), step=epoch_i)
                tf.summary.scalar('train_accuracy', self.train_accuracy.result(), step=epoch_i)

            # at end of some epochs run validation metrics
            if epoch_i % self.params.min_epochs_until_checkpoint == 0:
                val_accuracy, val_loss = self.get_validation_metrics(val_ds)
                self.val_loss(val_loss)
                self.val_accuracy(val_accuracy)
                logger.info(f'Validation Accuracy: {val_accuracy}, Validation Loss: {val_loss}')
                val_acc_list.append(val_accuracy)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logger.info(f'Saving on batch {batch}')
                    logger.info(f'New best validation loss: {best_val_loss}')
                    ckpt_save_path = ckpt_manager.save()
                    logger.info('Saving checkpoint for epoch {} at {}'.format(epoch_i + 1,
                                                                        ckpt_save_path))
                with self.val_summary_writer.as_default():
                    tf.summary.scalar('val_loss', self.val_loss.result(), step=epoch_i)
                    tf.summary.scalar('val_accuracy', self.val_accuracy.result(), step=epoch_i)

            # print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch_i + 1,
            #                                                     self.train_loss.result(),
            #                                                     np.mean(accuracy_list[-50:])))

            # print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

            # # early stopping
            # if all([val_accuracy > best_val_accuracy for val_accuracy in val_acc_list[-5:]]):
            #     break

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                                  tf.TensorSpec(shape=(None, None), dtype=tf.int32), ])
    def train_step(self, inputs, targets):
        teacher_forcing_targets = targets[:, :-1]
        actual_targets = targets[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inputs, teacher_forcing_targets)
        with tf.GradientTape() as tape:
            probs, _ = self.call(inputs, teacher_forcing_targets,
                                   True,
                                   enc_padding_mask,
                                   combined_mask,
                                   dec_padding_mask)
            loss = self.loss_function(actual_targets, probs)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_loss(loss)
        return probs

    def inference(self, encoder_input):
        decoder_input = [self.params.start_token]
        output = tf.expand_dims(decoder_input, 0)
        for i in range(self.params.answer_max_length-1):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, output)
            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.call(encoder_input, output, False,
                                                       enc_padding_mask, combined_mask, dec_padding_mask)
            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            # return the result if the predicted_id is equal to the end token
            if predicted_id == self.params.end_token:
                return tf.squeeze(output, axis=0), attention_weights
            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output = tf.concat([output, predicted_id], axis=-1)

        return tf.squeeze(output, axis=0), attention_weights

    def batch_inference(self, encoder_input):
        all_preds = tf.ones((self.params.batch_size, 1), dtype=tf.int32)
        all_probs = tf.ones((self.params.batch_size, 1, self.params.vocab_size), dtype=tf.float32)
        for i in range(self.params.answer_max_length-1):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, all_preds)
            # predictions.shape == (batch_size, seq_len, vocab_size)
            probs, attention_weights = self.call(encoder_input, all_preds, False,
                                                       enc_padding_mask, combined_mask, dec_padding_mask)
            # select the last word from the seq_len dimension
            probs = probs[:, -1:, :]  # (batch_size, 1, vocab_size)
            preds = tf.cast(tf.argmax(probs, axis=-1), tf.int32)
            # concatentate the predicted_id to the all_preds which is given to the decoder
            # as its input.
            all_probs = tf.concat([all_probs, probs], axis=1)
            all_preds = tf.concat([all_preds, preds], axis=-1)
        return all_preds[:, 1:], all_probs[:, 1:, :], attention_weights

    def inspect_inference(self, input_batch, target_batch, logger, num_to_inspect=1):
        for i in range(num_to_inspect):
            first_inp = input_batch[i-1: i]
            first_target = target_batch[i]
            first_output, _ = self.inference(first_inp)
            logger.info("pred: " + decode(first_output, self.idx2char))
            logger.info("targ: " + decode(first_target, self.idx2char))

    def get_validation_metrics(self, val_ds):
        metrics_dict = {'accuracy': [], 'loss': []}
        for batch, (input_batch, target_batch) in enumerate(val_ds):
            if input_batch.shape[0] < self.params.batch_size:
                continue
            preds_batch, probs_batch, _ = self.batch_inference(input_batch)
            accuracy = self.accuracy(target_batch, preds_batch)
            loss = self.loss_function(target_batch[:,1:], probs_batch)
            metrics_dict['accuracy'].append(accuracy)
            metrics_dict['loss'].append(loss)
        accuracy = sum(metrics_dict['accuracy'])/len(metrics_dict['accuracy'])
        loss = sum(metrics_dict['loss'])/len(metrics_dict['loss'])
        return accuracy, loss

    def accuracy(self, target_batch, preds):
        first_padding_positions = tf.argmax(
            tf.cast(tf.equal(tf.cast(tf.zeros(target_batch.shape), dtype=tf.float32),
                             tf.cast(target_batch, dtype=tf.float32)),
                    tf.float32), axis=1)
        padding_mask = tf.sequence_mask(lengths=first_padding_positions, maxlen=self.params.answer_max_length - 1,
                                        dtype=tf.int32)
        preds_to_compare = preds * padding_mask
        targets_to_compare = target_batch[:, 1:] * padding_mask
        # Compare row-by-row for exact match between preds / true target sequences
        correct_pred_mask = tf.reduce_all(tf.equal(preds_to_compare, targets_to_compare), axis=1)
        accuracy = tf.reduce_sum(tf.cast(correct_pred_mask, dtype=tf.int32)) / tf.shape(correct_pred_mask)[0]
        return accuracy

def decode(encoding, idx2char):
    return "".join([idx2char[idx] for idx in encoding.numpy() if idx not in [0,1,2]])


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

