import sys
import csv
import string
import operator
import random
import re
import scipy
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import chain, combinations
from scipy.stats import chi2_contingency

#punctuations
punc = set(string.punctuation)
freq = {} #freq dic
featureToIndex = {}
minsupp = 0.03
minconf = 0.25
alpha = 0.05
p_value = alpha/682
allFreq = defaultdict(int)
ONESIZE = 2002    # 2002 choose 1
TWOSIZE = 2003001   # 2002 choose 2
THRSIZE = 1335334000    # 2002 choose 3
totalConsidered = 0
totalInfreq = 0
chisquaremode = True
def regex_split(s):
	rgx = re.compile("([\w][\w']*\w)")
	return rgx.findall(s)

def calcfreq(lowerStr):
	for word in lowerStr:
		if word in freq:
			freq[word]+=1
		else:
			freq[word]=1

def clearpunc(string):
	result=""
	for ch in string:
		if ch not in punc:
			result=result+ch
		# if ch in punc:
		# 	result=result+" "
	return result

def loadcsv(filename):
	f=open(filename, "rb")
	lines = csv.reader(f)
	dataset = list(lines)
	f.close()
	return dataset

def subsets(arr):
	return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])

def kminusonesubsets(arr, k):
	return chain(*[combinations(arr, k-1)])

def getMinSuppMatch (D, minsupp, itemset_arr, allFreq, previousL, k):
	global totalConsidered
	global totalInfreq
	print "Itemset length before candidate pruning: " + str(len(itemset_arr))
	# candidate pruning
	pruned_itemset_arr = set(itemset_arr)
	if previousL is not None:
		for itemset in itemset_arr:
			kminusone_subsets = map(frozenset, [x for x in kminusonesubsets(itemset, k)])
			kminusone_subsets = set(kminusone_subsets)
			if not kminusone_subsets.issubset(previousL):
				pruned_itemset_arr.remove(itemset)
	print "Itemsets length considered (after prunn): " + str(len(pruned_itemset_arr))
	if k==1:
		ratio = float(ONESIZE - len(pruned_itemset_arr)) / ONESIZE
	if k==2:
		ratio = float(TWOSIZE - len(pruned_itemset_arr)) / TWOSIZE
	if k==3:
		ratio = float(THRSIZE - len(pruned_itemset_arr)) / THRSIZE
	print str(k) + "-itemset Pruning ratio: " + str(ratio)

	totalConsidered = len(pruned_itemset_arr) + totalConsidered
	C = defaultdict(int)
	for itemset in pruned_itemset_arr:
		for transaction in D:
			if itemset.issubset(transaction):
				C[itemset] += 1
				allFreq[itemset] += 1
	
	L_itemset = set()
	# prune out stuffs
	for itemset, count in C.items():
		supp = float(count)/len(D)
		if supp >= minsupp:
			L_itemset.add(itemset)
	print "Itemsets length considered frequent: " + str(len(L_itemset))
	print "Itemsets length considered infrequent: " + str(len(pruned_itemset_arr) - len(L_itemset))
	far = float(len(pruned_itemset_arr) - len(L_itemset)) / len(pruned_itemset_arr)
	print str(k) + "-itemset False alarm rate: " + str(far)
	totalInfreq += len(pruned_itemset_arr) - len(L_itemset)
	print "------------------------------------------------"
	return L_itemset

def selfJoin(itemset_arr ,length):
	return set([i.union(j) for i in itemset_arr for j in itemset_arr if len(i.union(j)) == length])

def getSupp(item, allFreq, D):
	return float(allFreq[item])/len(D)


def FrequentItemsetGeneration (D, minsupp, selectedWords):
	globalL = dict()
	itemset_arr = set()  # build 1-itemsets (candidate itemset)
	for word in set(selectedWords):
		itemset_arr.add(frozenset([word]))
	itemset_arr.add(frozenset(["isPositive"]))
	itemset_arr.add(frozenset(["isNegative"]))
	k = 1
	
	currentL_itemset = getMinSuppMatch(D, minsupp, itemset_arr, allFreq, None, k)
	globalL[k] = currentL_itemset
	#print currentL_itemset

	k = 2
	while (currentL_itemset!=set([]) and k <= 3):
		itemset_arr = selfJoin(currentL_itemset, k)
		currentL_itemset = getMinSuppMatch(D, minsupp, itemset_arr, allFreq, globalL[k-1], k)
		globalL[k] = currentL_itemset
		# print currentL_itemset
		k=k+1
	L = []
	for key, value in globalL.items():
		L.append([(tuple(item), getSupp(item, allFreq, D)) for item in value])
	return L, globalL


def RuleGeneration (D, globalL, minconf):
	Rules = []
	for key, value in globalL.items()[1:]:
		for item in value:
			#_subsets = map(frozenset, [x for x in subsets(item)])
			for consequence in item:
				consequence = frozenset([consequence])
				antecedent = item.difference(consequence)
				if len(consequence) > 0:
					if (chisquaremode):
						#calculate chi square value
						#A->B
						A = getSupp(antecedent, allFreq, D)*len(D)
						B = getSupp(consequence, allFreq, D)*len(D)
						AB = getSupp(item, allFreq, D)*len(D)
						# print "AB: " + str(AB)
						A_B = A-AB
						# print "A_B: " + str(A_B)
						_AB = B-AB
						# print "_AB: " + str(_AB)
						_A_B = len(D) - AB - A_B - _AB
						# print "_A_B: " + str(_A_B)
						chistatistics = chi2_contingency(np.array([[AB,A_B],[_AB,_A_B]]))
						if chistatistics[1] <= p_value:
							Rules.append(((tuple(antecedent), tuple(consequence)), chistatistics[0], getSupp(item, allFreq, D), getSupp(antecedent, allFreq, D), getSupp(consequence, allFreq, D), chistatistics[1]))
					else:
						confidence =  getSupp(item, allFreq, D) / getSupp(antecedent, allFreq, D)
						if confidence >= minconf:
							Rules.append(((tuple(antecedent), tuple(consequence)), confidence, getSupp(item, allFreq, D), getSupp(antecedent, allFreq, D), getSupp(consequence, allFreq, D)))
	return Rules


def runApriori(D, minsupp, minconf, selectedWords):
	L, globalL = FrequentItemsetGeneration (D, minsupp, selectedWords)
	R = RuleGeneration (D, globalL, minconf)
	return L, R

if __name__ == '__main__':
	review=[]
	classlabelset=[]
	training="stars_data.csv"
	# testing=sys.argv[2]
	classlabelindex=7
	dataset=loadcsv(training)
	trainsize=len(dataset)
	# NBC model
	for i in range(1,len(dataset)):
		#cleanStr=clearpunc(dataset[i][7]) # hard coded for review column
		classlabel=int(dataset[i][classlabelindex-1])
		cleanStr = clearpunc(dataset[i][7])
		lowerStr=[s.lower() for s in string.split(cleanStr)]
		review.append(lowerStr)  # might cause problem ???????????????????????
		classlabelset.append(classlabel)
		calcfreq(lowerStr)
	#sort words in descending order
	sortedfreq = sorted(freq.items(), key=operator.itemgetter(1),reverse=True)
	selected = sortedfreq[100:2100] # drop the top 100 words
	selectedWords = [item[0] for item in selected]
	topwords = set(selectedWords) #for speeding
	i = 0
	for word in selectedWords:
		featureToIndex[word]=i
		i+=1
	featureToIndex["isPositive"] = 2000
	featureToIndex["isNegative"] = 2001
	
	# for i in range(0,100):
	# 	print "WORD"+str(i+1)+" "+selected[i][0]
	# build matrix
	#feature_matrix=[[0 for x in range(2002)] for x in range(len(review))]
	D = list()
	for i in range(0,len(review)):
		tmp = set()
		for word in review[i]:
			if word in topwords:
				tmp.add(word)
				#feature_matrix[i][featureToIndex[word]]+=1
		if classlabelset[i]==5: 
			tmp.add("isPositive")
			#feature_matrix[i][2000] = 1 # isPositive
			#feature_matrix[i][2001] = 0 # isNegative
		else:
			tmp.add("isNegative")
			#feature_matrix[i][2000] = 0 # isPositive
			#feature_matrix[i][2001] = 1 # isNegative
		D.append(tmp)
	#print D
	L, R = runApriori(D, minsupp, minconf, selectedWords)
	# get the top 30
	def getSortKey(item): 
		return item[1]
	R = sorted(R, key = getSortKey, reverse=True)
	print "Size of association rule set: " + str(len(R))
	topRules = R[:30]


	# printings
	# size of pattern space
	size = ONESIZE + TWOSIZE + THRSIZE
	print "Size of pattern space: " + str(size)

	# Pruning ratio
	ratio = float(size - totalConsidered) / size
	print "Pruning ratio: " + str(ratio)

	# false alarm rate
	far = float(totalInfreq) / totalConsidered
	print "False alarm rate: " + str(far)

	# for i in range(len(L)):
	# 	print str(i+1) + "-itemset:"
	# 	print L[i]
	# 	print "-----------------------------------------------"
	print "Top 30 Rules"
	for rule in topRules:
		if not chisquaremode:
			antecedent = rule[0][0]
			consequence = rule[0][1]
			confidence = rule[1]
			supp = rule[2]
			antSupp = rule[3]
			conseSupp = rule[4]
			print "IF " + str(antecedent) + " THEN " + str(consequence) + ". Confidence is: " + str(confidence) + ", supp is: " + str(supp)
			print "antSupp: " + str(antSupp) + ". ConseSupp: " + str(conseSupp)
		else:
			antecedent = rule[0][0]
			consequence = rule[0][1]
			x2 = rule[1]
			supp = rule[2]
			antSupp = rule[3]
			conseSupp = rule[4]
			p_value = rule[5]
			print "IF " + str(antecedent) + " THEN " + str(consequence) + ". Chi square value: is: " + str(x2) + ", p value: " + str(p_value) +", supp is: " + str(supp)
			print "antSupp: " + str(antSupp) + ". ConseSupp: " + str(conseSupp)