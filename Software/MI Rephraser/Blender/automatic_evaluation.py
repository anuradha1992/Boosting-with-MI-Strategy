import numpy as np
import os
import csv
import random
import json
import math

import numpy as np

from sentence_transformers import SentenceTransformer, util
#from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download('omw-1.4')

from nltk.translate.bleu_score import sentence_bleu

from nltk.translate.meteor_score import single_meteor_score

from rouge import Rouge

from bert_score import score

from nltk.translate.chrf_score import sentence_chrf

import gensim.downloader as api
model = api.load('word2vec-google-news-300')

#sbert = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
sbert = SentenceTransformer('sentence-transformers/roberta-base-nli-stsb-mean-tokens', device='cuda')

from nltk.corpus import stopwords
from nltk import download
download('stopwords')  # Download stopwords list.
stop_words = stopwords.words('english')

def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]


# ========================== POS =============================
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag, word_tokenize
import torchtext.vocab as torch_vocab
import torch
dic_glove = torch_vocab.GloVe(name='twitter.27B',dim=100)
# ========================== POS =============================

folder = 'train_PP_test_PP'
path = './output/'+folder+'/'

files = ['template.csv', 'retrieve.csv', 'template_retrieve.csv', 'template_retrieve_prompt.csv', 'template_retrieve_ngram_prompt.csv']
#files = ['template.csv']

#print("================================================================")

for file in files:

	#print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")

	print("================ " + file + " ====================")

	wmd_list = []
	chrf_list = []
	meteor_list = []
	b1_list = []
	b2_list = []
	b3_list = []
	b4_list = []
	emb_list = []

	refs = []
	cands = []

	# ========================== POS =============================
	loss_nn  = 0
	count = 0
	# ========================== POS =============================

	with open(path+file, 'r', encoding='utf-8') as infile:
	    readCSV = csv.reader(infile, delimiter=',')
	    next(readCSV)

	    print("-----------------------------------------------")
	    
	    for row in readCSV:

	        ground_truth = row[1].strip()
	        predicted = row[2].strip()

	        refs.append(ground_truth)
	        cands.append(predicted)

	        # compute word mover distance
	        ground_truth_wo_stopwords = preprocess(ground_truth)
	        predicted_wo_stopwords = preprocess(predicted)
	        wmd = model.wmdistance(ground_truth_wo_stopwords, predicted_wo_stopwords)
	        wmd_list.append(wmd)
	        #print('word mover distance = %.4f' % wmd)

	        #print("*********************************************")

	        chrf = sentence_chrf(ground_truth, predicted)
	        chrf_list.append(chrf)
	        #print('chrf = %.4f' % chrf)

	        #print("&&&&&&&&&&&&&&&&&&&&")

	        meteor = single_meteor_score(ground_truth.split(), predicted.split())
	        meteor_list.append(meteor)

	        #print("ççççççççççççççççç")

	        bleu_1 = sentence_bleu([ground_truth.split()], predicted.split(), weights=(1, 0, 0, 0))
	        b1_list.append(bleu_1)
	        #print("11111111111111")
	        bleu_2 = sentence_bleu([ground_truth.split()], predicted.split(), weights=(0.5, 0.5, 0, 0))
	        b2_list.append(bleu_2)
	        #print("2222222222")
	        bleu_3 = sentence_bleu([ground_truth.split()], predicted.split(), weights=(0.33, 0.33, 0.33, 0))
	        b3_list.append(bleu_3)
	        #print("333333333")
	        bleu_4 = sentence_bleu([ground_truth.split()], predicted.split(), weights=(0.25,0.25,0.25,0.25))
	        b4_list.append(bleu_4)
	        #print("444444444444")

	        # ========================== POS =============================
	        temp_res_ori = pos_tag(word_tokenize(ground_truth))
	        temp_res_gen = pos_tag(word_tokenize(predicted))
	        temp_nn_ori = []
	        temp_nn_gen = []
	        temp_nn_vector_ori = []
	        temp_nn_vector_gen = []
	        for tube in temp_res_ori:
	        	if tube[1] == 'NN':
	        		temp_nn_ori.append(tube[0])
	        for tube in temp_res_gen:
	        	if tube[1] == 'NN':
	        		temp_nn_gen.append(tube[0])
	        for word in temp_nn_ori:
	        	try:
	        		temp_nn_vector_ori.append(dic_glove.vectors[dic_glove.stoi[word]])
	        	except KeyError:
	        		print("Key error 1")
	        for word in temp_nn_gen:
	        	try:
	        		temp_nn_vector_gen.append(dic_glove.vectors[dic_glove.stoi[word]])
	        	except KeyError:
	        		print("Key error 2")
	        if temp_nn_vector_ori != [] and temp_nn_vector_gen != []:
	        	loss_list = []
	        	for vector_target in temp_nn_vector_ori:
	        		for vector_gen in temp_nn_vector_gen:
	        			tensor_gen = torch.FloatTensor(vector_gen)
	        			tensor_target = torch.FloatTensor(vector_target)
	        			temp_loss = torch.dist(tensor_gen,tensor_target)
	        			loss_list.append(temp_loss)
	        	loss_list_new = sorted(loss_list)
	        	loss_list_new1 = loss_list_new[:min(len(temp_nn_vector_ori),len(temp_nn_vector_gen))]
	        	loss_nn += (sum(loss_list_new1)/len(loss_list_new1))*(1+abs(len(temp_nn_vector_ori)-len(temp_nn_vector_gen))/len(temp_nn_vector_ori))
	        	count+=1
	        # ========================== POS =============================

	print("File name: ", file)
	print("Average word mover distance = %.4f" % np.mean(wmd_list))
	print("Average chrf = %.4f" % np.mean(chrf_list))

	P, R, F1 = score(cands, refs, lang="en", verbose=True)
	print(f"System level Bert (P) score: {P.mean():.4f}")
	print(f"System level Bert (R) score: {R.mean():.4f}")
	print(f"System level Bert (F1) score: {F1.mean():.4f}")

	#cands, refs = map(list, zip(cands, refs))
	rouge = Rouge()
	scores = rouge.get_scores(cands, refs, avg=True)
	print("Rouge all scores = ", scores)
	print("Average ROUGE-L (F) score = %.4f" % scores['rouge-l']['f'])

	print("Average meteor score = %.4f" % np.mean(meteor_list))

	print("Average bleu-1 score = ", np.mean(b1_list))
	print("Average bleu-2 score = ", np.mean(b2_list))
	print("Average bleu-3 score = ", np.mean(b3_list))
	print("Average bleu-4 score = ", np.mean(b4_list))

	ref_embed = sbert.encode(refs, convert_to_tensor=True)
	cand_embed = sbert.encode(cands, convert_to_tensor=True)
	#print("sssssssssssssssss")
	cosine_scores = util.cos_sim(ref_embed, cand_embed)
	for i in range(len(refs)):
		#print("{} \t\t {} \t\t Score: {:.4f}".format(refs[i], cands[i], cosine_scores[i][i]))
		emb_list.append(float(cosine_scores[i][i]))
	#print("ccccccccccccccccc")
	print("Average cosine embedding similarity = ", np.mean(emb_list))

	# ========================== POS =============================
	print("Average POS distance = ", float(loss_nn/count))
	# ========================== POS =============================

	with open('./metrics/'+folder+'/'+file+'.csv', 'a') as f:
	    # create the csv writer
	    writer = csv.writer(f)
	    writer.writerow(['folder', 'file', 'bleu-1', 'bleu-2', 'bleu-3', 'bleu-4', 'rouge-1 (r)', 'rouge-1 (p)', 'rouge-1 (f)', 'rouge-2 (r)', 'rouge-2 (p)', 'rouge-2 (f)', 'rouge-L (r)', 'rouge-L (p)', 'rouge-L (f)', 'meteor', 'cos sim', 'WMD', 'chrf', 'BERTSCore (P)', 'BERTSCore (R)', 'BERTSCore (F1)', 'POS distance'])
	    writer.writerow([folder, file, np.mean(b1_list), np.mean(b2_list), np.mean(b3_list), np.mean(b4_list), scores['rouge-1']['r'], scores['rouge-1']['p'], scores['rouge-1']['f'], scores['rouge-2']['r'], scores['rouge-2']['p'], scores['rouge-2']['f'], scores['rouge-l']['r'], scores['rouge-l']['p'], scores['rouge-l']['f'], np.mean(meteor_list), np.mean(emb_list), np.mean(wmd_list), np.mean(chrf_list), float(P.mean()), float(R.mean()), float(F1.mean()), float(loss_nn/count)])


# {'rouge-1': {'r': 0.4632963306147853, 'p': 0.6712418030855859, 'f': 0.5396043091885412}, 'rouge-2': {'r': 0.36835728035702997, 'p': 0.575838540544423, 'f': 0.44327793529287773}, 'rouge-l': {'r': 0.4536832631342984, 'p': 0.6609510831678725, 'f': 0.5299945083903064}}













