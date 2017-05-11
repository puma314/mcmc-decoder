import numpy as np
from numpy.core.umath_tests import inner1d
import csv

import matplotlib.pyplot as plt #REMOVE THIS LATER

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '.']
alphabet_to_num = dict(zip(alphabet,range(len(alphabet))))
num_to_alphabet = dict(zip(range(len(alphabet)), alphabet))
transition_matrix = np.loadtxt('letter_transition_matrix.csv',delimiter=",").T
og_transition_matrix = np.copy(transition_matrix)
transition_matrix[transition_matrix==0.] = 1e-200 #THIS IS A HACK!!!!!
letter_probabilities = np.loadtxt('letter_probabilities.csv',delimiter=",")
log_transition_matrix = np.log(transition_matrix)
#log_transition_matrix[log_transition_matrix==-np.inf] = -1e200 #THIS IS A HACK!!!!!
log_letter_probs = np.log(letter_probabilities)

def check_plausible(perm, text_transitions):
	permutation_mat = np.identity(len(perm))[perm]
	updated_text_transition = np.dot(np.dot(permutation_mat.T,text_transitions),permutation_mat)
	#print updated_text_transition
	#print np.count_nonzero(updated_text_transition)
	non_zero = np.nonzero(updated_text_transition)
	#print non_zero
	trans_mat_entries = og_transition_matrix[non_zero]
	if trans_mat_entries.size > np.count_nonzero(trans_mat_entries): #if any entries are 0, then its not possible
		#	print 'hey'
		return False
	return True

def identify(text_transitions): #identifies the . and space
	counts_arr = np.count_nonzero(text_transitions, axis=1)
	period_candidates = np.where(counts_arr==1.)[0]
	if len(period_candidates) == 0:
		return (None, None)
	period_rows = text_transitions[period_candidates]
	space_freqs = np.max(period_rows,axis=1)
	most_freq = np.argmax(space_freqs)
	return ((period_candidates[most_freq],np.argmax(period_rows[most_freq])))

def construct_transition(num_text):
	transition = np.zeros((len(alphabet),len(alphabet)))
	for i in range(len(num_text)-1):
		char1,char2 = num_text[i], num_text[i+1]
		transition[char1][char2] += 1
	return transition

def check_permuted_transition(perm, trans):
	new_trans = np.zeros(trans.shape)
	for i in range(len(trans)):
		for j in range(len(trans)):
			new_trans[perm[i]][perm[j]] = trans[i][j]
	return new_trans

def check_LL(perm, text):
	LL = log_letter_probs[perm[text[0]]]
	for i in range(len(text)-1):
		LL += log_transition_matrix[perm[text[i]]][perm[text[i+1]]]
	return LL

def LL(perm, text_transitions,char0):
	permutation_mat = np.identity(len(perm))[perm]
	updated_text_transition = np.dot(np.dot(permutation_mat.T,text_transitions),permutation_mat) #check that this is correct
	#updated_text_transition = check_permuted_transition(perm, text_transitions) #holy shit just changing this goes from 25 seconds to 3 seconds
	#assert(np.array_equal(check_permuted_transition(perm, text_transitions),updated_text_transition))
	#check for zero mask
	#zero_mask = np.nonzero(updated_text_transition)
	#if transition_matrix[zero_mask].any(0):
	#	print 'BAD'
	likelihood = log_letter_probs[perm[char0]]
	likelihood += np.sum(np.multiply(log_transition_matrix,updated_text_transition)) + likelihood
	#checkLL = likelihood
	#for i in range(len(updated_text_transition)):
	#	for j in range(len(updated_text_transition)):
	#		checkLL += updated_text_transition[i][j]*log_transition_matrix[i][j] #this matches up with check_LL
	#print checkLL
	#print np.sum(np.multiply(log_transition_matrix,updated_text_transition)) + likelihood
	return likelihood #check for -inf

def convert(perm, num_text):
	perm_dic = dict(zip(range(len(perm)), perm))
	text_perm = np.vectorize(perm_dic.__getitem__)(num_text)
	return np.vectorize(num_to_alphabet.__getitem__)(text_perm)

def check_subset(perm, num_text, amt, all_words):
	converted = convert(perm, num_text)
	converted = ''.join(list(converted))
	converted_split = converted.split(" ")

	not_in = 0
	if amt > len(converted_split):
		start = 0
		amt = len(converted_split)

	start = np.random.randint(len(converted_split)-amt)

	for w in converted_split[start:start+amt]:
		w = w.rstrip('.')
		if w not in super_words:
			not_in += 1

	return float(not_in)/amt

def decode(text):
	all_words = None
	text = np.array(list(text)) #this can be made faster using numpy
	char0 = alphabet_to_num[text[0]]
	num_text = np.vectorize(alphabet_to_num.__getitem__)(text)
	text_transitions = construct_transition(num_text)
	
	period,space = identify(text_transitions)
	#print period, space

	tot_starts = 30
	mini_epochs = 3000

	best_perm = None
	best_perm_LL = -np.inf
	perm_to_LL = {}

	def gen_perm():
		if period is not None:
			cur_perm = np.zeros(28)
			mask = [x for x in range(28) if x != period and x != space]
			cur_perm[period]=27
			cur_perm[space]=26
			cur_perm[mask] = np.random.permutation(26)
			cur_perm = np.rint(cur_perm)
			cur_perm = cur_perm.astype(int)
		else:
			cur_perm = np.random.permutation(28)
		return cur_perm
	
	stop = True
	while start < tot_starts and stop: 
		LL_array = []

		cur_perm = gen_perm()
		while not check_plausible(cur_perm, text_transitions):
			#print 'infeasible'
			cur_perm = gen_perm()
		#print 'generated cur perm'

		cur_perm_LL = LL(cur_perm,text_transitions,char0)
		perm_to_LL[tuple(cur_perm)] = cur_perm_LL

		for i in range(mini_epochs):
			switch1 = np.random.randint(0,28) 
			switch2 = switch1
			while switch2 == switch1:
				switch2 = np.random.randint(0,28)
			next_perm = list(cur_perm)
			next_perm[switch1], next_perm[switch2] = next_perm[switch2], next_perm[switch1]
			next_perm_LL = perm_to_LL.get(tuple(next_perm))
			if next_perm_LL is None:
				next_perm_LL = LL(next_perm,text_transitions,char0)
				perm_to_LL[tuple(next_perm)] = next_perm_LL
			
			if next_perm_LL >= cur_perm_LL:
				cur_perm = list(next_perm)
				cur_perm_LL = next_perm_LL
			else:
				uni_draw = np.random.uniform()
				log_ratio = next_perm_LL-cur_perm_LL
				ratio = np.exp(log_ratio)
				if uni_draw <= ratio: 
					cur_perm = list(next_perm)
					cur_perm_LL = next_perm_LL
			
			LL_array.append(cur_perm_LL)

			if cur_perm_LL >= best_perm_LL:
				best_perm_LL = cur_perm_LL
				best_perm = list(cur_perm)
		
		if check_language:
			if all_words is None:
				with open('common_words.txt', 'r') as f:
					comm_words = f.read().splitlines()
				comm_words = set(comm_words)
				with open('all_words.txt','r') as f:
					scrabble_words = f.read().splitlines()
				scrabble_words = set(scrabble_words)
				all_words = scrabble_words.union(comm_words)

			frac_no_match = check_subset(best_perm, num_text, amt=100, all_words)
			if frac_no_match > 0.1:
				stop = False

	print best_perm_LL/len(num_text)

	return (best_perm, convert(best_perm,num_text))

if __name__=="__main__":
	with open('plaintext1.txt') as f:
		cipher = f.readlines()[0][:-1]

	best_perm, converted = decode(cipher)
	print best_perm