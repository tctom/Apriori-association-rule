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
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.naive_bayes import BernoulliNB
#punctuations
punc = set(string.punctuation)
freq = {} #freq dic
featureToIndex = {}
minsupp = 0.03
minconf = 0.25

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

def FrequentItemsetGeneration (D, minsupp, selectedWords):
	itemset_arr = set()  # build 1-itemsets
	for word in set(selectedWords):
		itemset_arr.add(frozenset([word]))
	itemset_arr.add(frozenset(["isPositive"]))
	itemset_arr.add(frozenset(["isNegative"]))

	C1 = defaultdict(int)

	for record in feature_matrix:
		for itemset in itemset_arr:
			test = True
			for item in itemset:
				if record[featureToIndex[item]] != 1:
					test = False
			if test == True:
				C1[itemset] += 1
	
	L1 = defaultdict(int)
	# prune out stuffs
	for itemset, count in C1.items():
		supp = float(count)/len(D)
		if supp >= minsupp:
			L1[itemset] = count
	print L1



	# for record in feature_matrix:
	# 	for i in range(len(selectedWords)):
	# 		if i == 2000 && record[i] == 1:
	# 			C1["isPositive"] += 1
	# 		if i == 2001 && record[i] == 1:
	# 			C1["isNegative"] += 1
	# 		if record[i] == 1:
	# 			C1[selectedWords[i]] += 1
	return 0


# def RuleGeneration (L, minconf):


def runApriori(D, minsupp, minconf, selectedWords):
	L = FrequentItemsetGeneration (D, minsupp, selectedWords)
	# R = RuleGeneration (L, minconf)
	# return R

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
		cleanStr = dataset[i][7]
		lowerStr=[s.lower() for s in regex_split(cleanStr)]
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
	
	for i in range(0,100):
		print "WORD"+str(i+1)+" "+selected[i][0]
	# build matrix
	feature_matrix=[[0 for x in range(2002)] for x in range(len(review))]
	for i in range(0,len(review)):
		for word in review[i]:
			if word in topwords:
				feature_matrix[i][featureToIndex[word]]+=1
		if classlabelset[i]==5: 
			feature_matrix[i][2000] = 1 # isPositive
			feature_matrix[i][2001] = 0 # isNegative
		else:
			feature_matrix[i][2000] = 0 # isPositive
			feature_matrix[i][2001] = 1 # isNegative

	runApriori(feature_matrix,minsupp, minconf, selectedWords)