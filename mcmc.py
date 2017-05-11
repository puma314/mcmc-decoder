import numpy as np
import csv
import random
import matplotlib.pyplot as plt
import pickle

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '.']


alphabet_map = dict(zip(range(len(alphabet)),alphabet))
num_to_alphabet = {v:k for k,v in alphabet_map.iteritems()}
print alphabet
exit()
#755 nonzero out of 784 entries
transition_matrix = np.loadtxt('letter_transition_matrix.csv',delimiter=",")
letter_probabilities = np.loadtxt('letter_probabilities.csv',delimiter=",")

with open('ciphertext.txt') as f:
	cipher = f.readlines()[0][:-1]

with open('plaintext.txt') as f:
	plaintext = f.readlines()[0][:-1]

cipher_num = [num_to_alphabet[ch] for ch in cipher]

with open('cipher_function.csv', 'rb') as cipher_function:
	real_perm_letters = csv.reader(cipher_function)
	for row in real_perm_letters:
		real_perm_letters = row


def inverse_perm(p):
	s = [0 for _ in range(len(p))]
	for i in range(len(p)):
	    s[p[i]] = i
	return s

real_perm = [num_to_alphabet[a] for a in real_perm_letters]
real_perm = inverse_perm(real_perm)
print 'real perm', real_perm

def probability(perm, y):
	#print 'perm', perm
	f_inv = perm#inverse_perm(perm)
	y_inv = [f_inv[c] for c in y]
	log_prob = 0
	log_prob += np.log(letter_probabilities[y_inv[0]])
	#print log_prob
	for i in range(1,len(y)):
		transition_prob = transition_matrix[y_inv[i]][y_inv[i-1]] 
		if transition_prob == 0:
			#print 'alert', i
			log_prob += np.log(1e-10) #hack
		else:
			log_prob += np.log(transition_prob)
		#if i % 3000 == 0:
			#print log_prob
	return log_prob

def eval_accuracy(perm, y, ground):
	assert(len(y) == len(ground))
	wrong = 0
	for i, ch in enumerate(y):
		convert = alphabet_map[perm[ch]]
		if convert != ground[i]:
			wrong += 1


	return 1.-float(wrong)/len(ground)

cur_perm = np.random.permutation(28)
log_likelihoods = []
transition_acceptance = []
accuracy = []

print 'correct perm log-likelihood', probability(real_perm, cipher_num)
print 'test eval acc', eval_accuracy(real_perm, cipher_num, plaintext)

for i in range(3000):

	switch1 = np.random.randint(0,28)
	switch2 = switch1
	while switch2 == switch1:
		switch2 = np.random.randint(0,28)
	assert(switch1 != switch2)

	next_perm = list(cur_perm)
	next_perm[switch1], next_perm[switch2] = next_perm[switch2], next_perm[switch1]
	segment_length = 1000
	p_next = probability(next_perm,cipher_num[2000:2000+segment_length])
	p_cur = probability(cur_perm,cipher_num[2000:2000+segment_length])

	#assert(p_next != p_cur)

	log_likelihoods.append(p_cur)
	acc = eval_accuracy(cur_perm, cipher_num, plaintext)
	accuracy.append(acc)

	if p_next >= p_cur:
		cur_perm = next_perm[:]
		transition_acceptance.append(1)
	else:
		uni_draw = np.random.uniform()
		log_ratio = p_next-p_cur
		ratio = np.exp(log_ratio)
		#print 'log_ratio, ratio', log_ratio, ratio
		if uni_draw <= ratio: #there is probably something wrong with this step
			print 'advanced from the uni draw'
			transition_acceptance.append(1)
			cur_perm = next_perm[:]
		else:
			transition_acceptance.append(0)

	cont = True
	if i % 10 == 0:
		num_diff = 0
		for j in range(len(cur_perm)):
			if cur_perm[j] != real_perm[j]:
				num_diff += 1
		if num_diff == 0:
			cont = False
	
		if len(log_likelihoods) > 0:
			print i, log_likelihoods[-1], num_diff, acc


	#arr = [transition_acceptance, accuracy, log_likelihoods]
	#f = open('arrays.py','w')
	#pickle.dump(arr,f)
	#f.close()

	#if not cont:
	#	break


# def plot_acceptance(transition_acceptance, T):
# 	arr = []
# 	for i in range(T,len(transition_acceptance)):
# 		yes = sum(transition_acceptance[i-T:i])/float(T)
# 		arr.append(yes)
# 	return arr

#import pickle
#pickle.dump(arr, 'arrays.p')

#plt.plot(plot_acceptance(transition_acceptance,T=20))
#plt.show()
#plt.plot(accuracy)
#plt.show()
#plt.plot(log_likelihoods)
#plt.show()

print p_cur

