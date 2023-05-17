import time
import numpy as np
import tensorflow as tf
from os import mkdir
from os.path import exists
from pytorch_transformers import RobertaTokenizer

from optimize import CustomSchedule
from model_utils import *
from model_emobert import EmoBERT, loss_function
from datasets import create_datasets
import csv

miti_arr = ['Closed Question',
'Open Question',
'Simple Reflection',
'Complex Reflection',
'Give Information',
'Advise with Permission',
'Affirm',
'Emphasize Autonomy',
'Support',
'Advise without Permission',
'Confront',
'Direct',
'Warn',
'Self-Disclose',
'Other']

# Some hyper-parameters
num_layers = 12
d_model = 768
num_heads = 12
dff = d_model * 4
hidden_act = 'gelu'  # Use 'gelu' or 'relu'
dropout_rate = 0.1
layer_norm_eps = 1e-5
max_position_embed = 514
num_emotions = 15  # Number of emotion categories

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


# =========================== Change ===========================
dataset = 'mi_gold' # 'mi_augmented_intersection'; 'mi_augmented_union'

optimal_epoch = {
    'mi_gold': 7,
    'mi_augmented_intersection': 2,
    'mi_augmented_union': 13,
}
checkpoint_path = './checkpoints/'+dataset
data_path = './data/'+dataset
output_path = './output/'+dataset+'/test_output.csv'
# ===============================================================

mirrored_strategy = tf.distribute.MirroredStrategy()

f_csv = open(output_path, 'a', encoding = 'utf-8')
writer = csv.writer(f_csv)
writer.writerow(['Text', 'Ground truth (index)', 'Ground truth', 'Prediction (index)', 'Prediction'])

ground_indices = []
texts = []

with open(data_path+'/test.txt') as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
for line in lines:
    arr = line.split(' <SEP> ')
    ground_indices.append(int(arr[0]))
    texts.append(arr[1])

with mirrored_strategy.scope():

    train_dataset, val_dataset, test_dataset = create_datasets(tokenizer, data_path, buffer_size,
        batch_size, max_length)

    # Define the model.
    emobert = EmoBERT(num_layers, d_model, num_heads, dff, hidden_act, dropout_rate,
        layer_norm_eps, max_position_embed, vocab_size, num_emotions)


    def build_model(model, max_length, vocab_size):
        inp = np.ones((1, max_length), dtype = np.int32)
        inp[0,:max_length//2] = np.random.randint(2, vocab_size, size = max_length//2)
        inp = tf.constant(inp)
        enc_padding_mask = create_masks(inp)
        _ = model(inp, True, enc_padding_mask)

    build_model(emobert, max_length, vocab_size)

    # Define optimizer and metrics.
    learning_rate = CustomSchedule(peak_lr, total_steps, warmup_steps)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = adam_beta_1, beta_2 = adam_beta_2,
        epsilon = adam_epsilon)


    # Define the checkpoint manager.
    ckpt = tf.train.Checkpoint(model = emobert, optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep = None)

    print(ckpt_manager.checkpoints[optimal_epoch[dataset]-1])
    ckpt.restore(ckpt_manager.checkpoints[optimal_epoch[dataset]-1])
    print('Checkpoint at epoch '+str(optimal_epoch[dataset])+' restored!!')

    

    def test():
        accuracy = []
        predicted_indices = []
        for (batch, inputs) in enumerate(test_dataset):
            inp, tar_emot = inputs
            enc_padding_mask = create_masks(inp)
            pred_emot = emobert(inp, False, enc_padding_mask)
            pred_emot = np.argmax(pred_emot.numpy(), axis = 1)
            print(batch, pred_emot)
            for emot in pred_emot:
                predicted_indices.append(emot)
            accuracy.append(np.mean(tar_emot.numpy() == pred_emot))
        return np.mean(accuracy), predicted_indices

    start = time.time()

    test_ac, pred_indices = test()
    print('Current accuracy on test set: {}\n'.format(test_ac))

    current_time = time.time()
    print('Time taken: {} secs'.format(current_time - start))


    for i in range(0, len(texts)):
        ground_label = miti_arr[ground_indices[i]]
        pred_label = miti_arr[pred_indices[i]]
        writer.writerow([texts[i], ground_indices[i], ground_label, pred_indices[i], pred_label])










