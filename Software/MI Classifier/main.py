set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

import tensorflow as tf
import numpy as np

tf.compat.v1.enable_eager_execution()

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, total_steps, warmup_steps):
        super(CustomSchedule, self).__init__()

        self.peak_lr = peak_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = step / self.warmup_steps
        arg2 = (self.total_steps - step) / (self.total_steps - self.warmup_steps)
        return self.peak_lr * tf.math.minimum(arg1, arg2)

# Masking
def create_padding_mask(seq):
    # To be consistent with RoBERTa, the padding index is set to 1.
    seq = tf.cast(tf.math.equal(seq, 1), tf.float32)

    # Add extra dimensions so that we can add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_masks(inp):
    enc_padding_mask = create_padding_mask(inp)
    return enc_padding_mask

def build_model(model, max_length, vocab_size):
    inp = np.ones((1, max_length), dtype = np.int32)
    inp[0,:max_length//2] = np.random.randint(2, vocab_size, size = max_length//2)
    inp = tf.constant(inp)
    enc_padding_mask = create_masks(inp)
    _ = model(inp, True, enc_padding_mask)

import math


# Multi-Head Attention
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dropout_rate, name = 'multi_head_attention'):
        super().__init__(name = name)

        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, name = 'query')
        self.wk = tf.keras.layers.Dense(d_model, name = 'key')
        self.wv = tf.keras.layers.Dense(d_model, name = 'value')

        self.dropout = tf.keras.layers.Dropout(dropout_rate, name = 'mha_dropout')
        self.dense = tf.keras.layers.Dense(d_model, name = 'mha_output')

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm = [0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask):
        """
        Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead) 
        but it must be broadcastable for addition.

        Args:
            q: query shape == (..., seq_len_q, depth)
            k: key shape == (..., seq_len_k, depth)
            v: value shape == (..., seq_len_v, depth_v)
            mask: Float tensor with shape broadcastable 
                to (..., seq_len_q, seq_len_k). Defaults to None.
        
        Returns:
            output, attention_weights
        """

        matmul_qk = tf.matmul(q, k, transpose_b = True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis = -1)  # (..., seq_len_q, seq_len_k)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # (As claimed in the RoBERTa implementation.)
        attention_weights = self.dropout(attention_weights)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

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
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm = [0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def gelu(x):
    """
    Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + tf.math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tf.math.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + tf.math.erf(x / math.sqrt(2.0)))

act_funcs = {'gelu': gelu, 'relu': tf.nn.relu}

# Pointwise Feed Forward Network
def point_wise_feed_forward_network(d_model, dff, hidden_act):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation = act_funcs[hidden_act],
            name = 'ff_hidden'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model, name = 'ff_output')  # (batch_size, seq_len, d_model)
    ], name = 'ff_network')


# Encoder Layer
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, hidden_act, dropout_rate, layer_norm_eps, layer_num):
        super().__init__(name = 'encoder_layer_{:02d}'.format(layer_num))

        self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.ffn = point_wise_feed_forward_network(d_model, dff, hidden_act)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon = layer_norm_eps,
            name = 'layernorm_1')
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon = layer_norm_eps,
            name = 'layernorm_2')

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate, name = 'dropout_1')
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate, name = 'dropout_2')

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training = training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training = training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2



def loss_function(real_emot, pred_emot):
    scce = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits = True, reduction = 'none')
    loss_ = scce(real_emot, pred_emot)
    return loss_

class EmoBERT(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, hidden_act, dropout_rate,
                 layer_norm_eps, max_position_embed, vocab_size, num_emotions):
        super().__init__(name = 'emo_bert')

        self.padding_idx = 1

        # Embedding layers
        self.word_embeddings = tf.keras.layers.Embedding(vocab_size, d_model, name = 'word_embed')
        self.pos_embeddings = tf.keras.layers.Embedding(max_position_embed, d_model, name = 'pos_embed')
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon = layer_norm_eps,
            name = 'layernorm_embed')
        self.dropout = tf.keras.layers.Dropout(dropout_rate, name = 'dropout_embed')

        # Encoder layers
        self.num_layers = num_layers
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, hidden_act, dropout_rate, layer_norm_eps, i)
            for i in range(num_layers)
        ]

        # Output layers
        self.attention_v = tf.keras.layers.Dense(1, use_bias = False, name = 'attention_v')
        self.attention_layer = tf.keras.layers.Dense(d_model, activation = 'tanh', name = 'attention_layer')
        self.hidden_layer = tf.keras.layers.Dense(d_model, activation = 'tanh', name = 'hidden_layer')
        self.output_layer = tf.keras.layers.Dense(num_emotions, name = 'output_layer')

    def call(self, x, training, mask):
        # x.shape == (batch_size, seq_len)

        seq_len = tf.shape(x)[1]

        # Add word embedding and position embedding.
        pos = tf.range(self.padding_idx + 1, seq_len + self.padding_idx + 1)
        pos = tf.broadcast_to(pos, tf.shape(x))
        x = self.word_embeddings(x)  # (batch_size, seq_len, d_model)
        x += self.pos_embeddings(pos)

        x = self.layernorm(x)
        x = self.dropout(x, training = training)

        # x.shape == (batch_size, seq_len, d_model)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        # Compute the attention scores
        projected = self.attention_layer(x)  # (batch_size, seq_len, d_model)
        scores = tf.nn.softmax(tf.squeeze(self.attention_v(projected), 2))
        scores = tf.expand_dims(scores, 1)  # (batch_size, 1, seq_len)

        # x.shape == (batch_size, d_model)
        x = tf.squeeze(tf.matmul(scores, x), 1)

        x = self.hidden_layer(x)
        x = self.output_layer(x)

        return x  # (batch_size, num_emotions)

from tqdm import tqdm
from os.path import join, exists

import numpy as np

def create_datasets(tokenizer, data_path, buffer_size, batch_size, max_length):
    print('Vocabulary size is {}.'.format(tokenizer.vocab_size))

    SOS_ID = tokenizer.encode('<s>')[0]
    EOS_ID = tokenizer.encode('</s>')[0]

    def create_dataset(read_path):
        print('Reading data from \"{}\"...'.format(read_path))

        with open(read_path, 'r') as f:
            lines = f.readlines()
            inputs = np.ones((len(lines), max_length), dtype = np.int32)
            labels = np.zeros(len(lines), dtype = np.int32)

            for i, line in tqdm(enumerate(lines), total = len(lines)):
                label, uttr = line.strip().split(' <SEP> ')
                uttr_ids = [SOS_ID] + tokenizer.encode(uttr)[:(max_length - 2)] + [EOS_ID]
                inputs[i,:len(uttr_ids)] = uttr_ids
                labels[i] = int(label)

            print('Created dataset with {} examples.'.format(inputs.shape[0]))

        return inputs, labels

    train_inputs, train_labels = create_dataset(join(data_path, 'train.txt'))
    val_inputs, val_labels = create_dataset(join(data_path, 'valid.txt'))
    test_inputs, test_labels = create_dataset(join(data_path, 'test.txt'))

    train_inputs = np.concatenate([train_inputs, val_inputs])
    train_labels = np.concatenate([train_labels, val_labels])

    train_dataset = (tf.data.Dataset.from_tensor_slices(train_inputs),
        tf.data.Dataset.from_tensor_slices(train_labels))
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_inputs),
        tf.data.Dataset.from_tensor_slices(test_labels))

    train_dataset = tf.data.Dataset.zip(train_dataset).shuffle(buffer_size).batch(batch_size)
    test_dataset = tf.data.Dataset.zip(test_dataset).batch(batch_size)

    return train_dataset, test_dataset


# ========================================================================== main text for training ========================================================================

import time
from os import mkdir
from os.path import exists
from pytorch_transformers import RobertaTokenizer

# Some hyper-parameters
num_layers = 12
d_model = 768
num_heads = 12
dff = d_model * 4
hidden_act = 'gelu'  # Use 'gelu' or 'relu'
dropout_rate = 0.1
layer_norm_eps = 1e-5
max_position_embed = 514
num_emotions = 15  # Number of dialogue acts

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
vocab_size = tokenizer.vocab_size

max_length = 100  # Maximum number of tokens
buffer_size = 100000
batch_size = 32
num_epochs = 10


peak_lr = 2e-5
total_steps = 7000
warmup_steps = 700


adam_beta_1 = 0.9
adam_beta_2 = 0.98
adam_epsilon = 1e-6

dataset = 'mi_gold' # 'mi_augmented_intersection'; 'mi_augmented_union'

checkpoint_path = './checkpoints/'+dataset
log_path = './log/'+dataset+'/classifier.log'
data_path = './data/'+dataset

f = open(log_path, 'w', encoding = 'utf-8')

mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
    train_dataset, test_dataset = create_datasets(tokenizer, data_path, buffer_size,
        batch_size, max_length)
    train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)

    # Define the model.
    emobert = EmoBERT(num_layers, d_model, num_heads, dff, hidden_act, dropout_rate,
        layer_norm_eps, max_position_embed, vocab_size, num_emotions)

    # Build the model and initialize weights from PlainTransformer pre-trained on OpenSubtitles.
    build_model(emobert, max_length, vocab_size)
    #emobert.load_weights(path + 'BERTActClassifier/pretrained_weights/roberta2bertact.h5')
    emobert.load_weights('./pretrained_weights/roberta2bertact.h5')
    print('Weights initialized from RoBERTa.')
    f.write('Weights initialized from RoBERTa.\n')

    # Define optimizer and metrics.
    learning_rate = CustomSchedule(peak_lr, total_steps, warmup_steps)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = adam_beta_1, beta_2 = adam_beta_2,
        epsilon = adam_epsilon)

    train_loss = tf.keras.metrics.Mean(name = 'train_loss')


    # Define the checkpoint manager.
    ckpt = tf.train.Checkpoint(model = emobert, optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = None)

    # If a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
        f.write('Latest checkpoint restored!!\n')

    @tf.function
    def train_step(dist_inputs):
        def step_fn(inputs):
            # inp.shape == (batch_size, seq_len)
            # tar_emot.shape == (batch_size,)
            inp, tar_emot = inputs
            enc_padding_mask = create_masks(inp)

            with tf.GradientTape() as tape:
                pred_emot = emobert(inp, True, enc_padding_mask)  # (batch_size, num_emotions)
                losses_per_examples = loss_function(tar_emot, pred_emot)
                loss = tf.reduce_sum(losses_per_examples) * (1.0 / batch_size)

            gradients = tape.gradient(loss, emobert.trainable_variables)    
            optimizer.apply_gradients(zip(gradients, emobert.trainable_variables))
            return loss

        losses_per_replica = mirrored_strategy.experimental_run_v2(
            step_fn, args = (dist_inputs,))
        mean_loss = mirrored_strategy.reduce(
            tf.distribute.ReduceOp.SUM, losses_per_replica, axis = None)

        train_loss(mean_loss)
        return mean_loss

    def validate():
        accuracy = []
        for (batch, inputs) in enumerate(test_dataset):
            inp, tar_emot = inputs
            enc_padding_mask = create_masks(inp)
            pred_emot = emobert(inp, False, enc_padding_mask)
            pred_emot = np.argmax(pred_emot.numpy(), axis = 1)
            accuracy.append(np.mean(tar_emot.numpy() == pred_emot))
        return np.mean(accuracy)

    # Start training
    for epoch in range(num_epochs):
        start = time.time()

        train_loss.reset_states()

        for (batch, inputs) in enumerate(train_dataset):
            current_loss = train_step(inputs)
            current_mean_loss = train_loss.result()
            print('Epoch {} Batch {} Mean Loss {:.4f} Loss {:.4f}'.format(
                epoch + 1, batch, current_mean_loss, current_loss))
            f.write('Epoch {} Batch {} Mean Loss {:.4f} Loss {:.4f}\n'.format(
                epoch + 1, batch, current_mean_loss, current_loss))

        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
        f.write('Saving checkpoint for epoch {} at {}\n'.format(epoch + 1, ckpt_save_path))

        epoch_loss = train_loss.result()
        print('Epoch {} Loss {:.4f}'.format(epoch + 1, epoch_loss))
        f.write('Epoch {} Loss {:.4f}\n'.format(epoch + 1, epoch_loss))

        current_time = time.time()
        print('Time taken for 1 epoch: {} secs'.format(current_time - start))
        f.write('Time taken for 1 epoch: {} secs\n'.format(current_time - start))

        val_ac = validate()
        print('Current accuracy on validation set: {}\n'.format(val_ac))
        f.write('Current accuracy on validation set: {}\n\n'.format(val_ac))

        

f.close()










