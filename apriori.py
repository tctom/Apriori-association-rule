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
#punctuations
punc = set(string.punctuation)
freq = {} #freq dic
featureToIndex = {}
minsupp = 0.03
minconf = 0.25
allFreq = defaultdict(int)
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

def getMinSuppMatch (D, minsupp, itemset_arr, allFreq):
	print "Itemsets length considered: " + str(len(itemset_arr))
	C = defaultdict(int)
	for itemset in itemset_arr:
		for transaction in D:
			if itemset.issubset(transaction):
				C[itemset] += 1
				allFreq [itemset] += 1
	L_itemset = set()
	# prune out stuffs
	for itemset, count in C.items():
		supp = float(count)/len(D)
		if supp >= minsupp:
			L_itemset.add(itemset)
	print "Itemsets length considered frequent: " + str(len(L_itemset))
	print "Itemsets length considered infrequent: " + str(len(itemset_arr) - len(L_itemset))
	return L_itemset

def selfJoin(itemset_arr ,length):
	return set([i.union(j) for i in itemset_arr for j in itemset_arr if len(i.union(j)) == length])

def getSupp(item, allFreq, D):
	return float(allFreq[item])/len(D)


def FrequentItemsetGeneration (D, minsupp, selectedWords):
	itemset_arr = set()  # build 1-itemsets
	for word in set(selectedWords):
		itemset_arr.add(frozenset([word]))
	itemset_arr.add(frozenset(["isPositive"]))
	itemset_arr.add(frozenset(["isNegative"]))
	k = 1
	
	currentL_itemset = getMinSuppMatch(D, minsupp, itemset_arr, allFreq)
	#print currentL   # L1
	globalL = dict()
	print currentL_itemset
	k = 2
	while (currentL_itemset!=set([]) and k <= 3):
		globalL[k-1] = currentL_itemset
		currentL_itemset = selfJoin(currentL_itemset, k)
		currentL_itemset = getMinSuppMatch(D, minsupp, currentL_itemset, allFreq)
		print currentL_itemset
		k=k+1
	toRetItems = []
	for key, value in globalL.items():
		toRetItems.extend([(tuple(item), getSupp(item, allFreq, D)) for item in value])
	return toRetItems, globalL


def RuleGeneration (D, globalL, minconf):
	Rules = []
	for key, value in globalL.items()[1:]:
		for item in value:
			_subsets = map(frozenset, [x for x in subsets(item)])
			for element in _subsets:
				remain = item.difference(element)
				if len(remain) > 0:
					confidence =  getSupp(item, allFreq, D) / getSupp(element, allFreq, D)
					if confidence >= minconf:
						Rules.append(((tuple(element), tuple(remain)), confidence))
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
		review.append(set(lowerStr))  # might cause problem ???????????????????????
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
	topRules = R[:30]

	print topRules