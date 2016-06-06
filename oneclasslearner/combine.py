from sklearn import svm
import re
import nltk
import string
import itertools
import math
import random

def word2ngrams(text, n=3):
    return ["".join(j) for j in zip(*[text[i:] for i in range(n)])]

use_pos = False
use_hitcount_ratio = True
use_ksvalue = True
use_klvalue = True
use_partial_hitcount = True
use_n_gram = True
n_value = 1
alphabet = string.ascii_lowercase
#alphabet += "\'"
#alphabet += " "

ngram_feature_vector = []
ngram_feature_vector_map = {}
counter = 0
if use_n_gram:
    for i in range(n_value + 1):
        if i == 0:
            continue
        #subsets = itertools.combinations_with_replacement(alphabet , i)
        subsets = itertools.permutations(alphabet , i)
        for subset in subsets:
            key = "".join([s for s in subset])
            ngram_feature_vector.append(0)
            ngram_feature_vector_map[key] = counter
            counter = counter + 1

if use_partial_hitcount:
    left_partial_hitcount= {}
    right_partial_hitcount= {}
    with open("partial_hitcount.txt") as f:
        for line in f:
            _phrase = line.rstrip(' ').lstrip(' ').strip('\n')
            splited = _phrase.split("\t")
            left_partial_hitcount[splited[0].strip()] = float(splited[1])
            right_partial_hitcount[splited[0].strip()] = float(splited[2])

if use_ksvalue:
    ksvalues = {}
    with open("ksvalue.txt") as f:
        for line in f:
            _phrase = line.rstrip(' ').lstrip(' ').strip('\n')
            splited = _phrase.split("\t")
            ksvalues[splited[0].strip()] = float(splited[1])

if use_klvalue:
    klvalues = {}
    with open("ksvalue.txt") as f:
        for line in f:
            _phrase = line.rstrip(' ').lstrip(' ').strip('\n')
            splited = _phrase.split("\t")
            klvalues[splited[0].strip()] = float(splited[2])


if use_pos:
    all_pos_tags = ['``', 'CD', 'WP$', "''", '--', 'VB', '$', 'JJR', '.', 'NNS', 'LS', 'TO', 'RB', 'VBD', 'NNP', 'DT', 'VBG', 'NNPS', 'PRP$', 'SYM', 'VBP', 'EX', 'UH', 'POS', 'VBN', 'PRP', 'RBS', ')', ',', 'FW', 'PDT', 'CC', 'NN', '(', 'JJ', ':', 'JJS', 'RP', 'RBR', 'WP', 'VBZ', 'WRB', 'IN', 'WDT', 'MD']
    all_pos_tag_dic = {}
    counter = 0
    for tag in all_pos_tags:
        all_pos_tag_dic[tag] = counter
        counter = counter + 1

all = []
hitcountTrain = {}
hitcountExactTrain = {}
trainset = set()
if use_pos:
    pos_tags_train = {}
    with open("all_taged.txt", 'r') as f:
        for line in f:
            _phrase = line.rstrip(' ').lstrip(' ').strip('\n')
            if len(_phrase) > 0:
                #splitOut = _phrase.rsplit(' ', 1)
                pos_tags_train[re.sub(" \?", "?", re.sub(" 'll ", "'ll ", re.sub(" , ", ", ",re.sub(" ' ", "' ",re.sub("can not" , "cannot", re.sub(" n't ", "n't ", re.sub(" 't ", "'t ",re.sub(" 'm ", "'m ",re.sub(" 's", "'s"," ".join([token.split("_")[0] for token in _phrase.split()]))))))))))] =\
                    " ".join([token.split("_")[1] for token in _phrase.split()])

with open("features_exact.txt", 'r') as f:
    for line in f:
        _phrase = line.rstrip(' ').lstrip(' ').strip('\n')
        if len(_phrase) > 0:
            splitOut = _phrase.rsplit(' ', 1)
            hitcountExactTrain[splitOut[0].rstrip(' ').lstrip(' ').strip('\n')] = int(splitOut[1].strip(' '))
            trainset.add(splitOut[0].rstrip(' ').lstrip(' ').strip('\n'))

with open("features.txt", 'r') as f:
    for line in f:
        _phrase = line.rstrip(' ').lstrip(' ').strip('\n')
        if len(_phrase) > 0:
            splitOut = _phrase.rsplit(' ', 3)
            hitcountTrain[splitOut[0].rstrip(' ').lstrip(' ').strip('\n')] = int(float(splitOut[-3].strip(' ').split('.')[0]))

with open("features_exact_test.txt", 'r') as f:
    for line in f:
        _phrase = line.rstrip(' ').lstrip(' ').strip('\n')
        if len(_phrase) > 0:
            splitOut = _phrase.rsplit(' ', 1)
            hitcountExactTrain[splitOut[0].rstrip(' ').lstrip(' ').strip('\n')] = int(splitOut[1].strip(' '))

with open("features_test.txt", 'r') as f:
    for line in f:
        _phrase = line.rstrip(' ').lstrip(' ').strip('\n')
        if len(_phrase) > 0:
            splitOut = _phrase.rsplit(' ', 3)
            hitcountTrain[splitOut[0].rstrip(' ').lstrip(' ').strip('\n')] = int(float(splitOut[-3].strip(' ').split('.')[0]))

gold_dataset3 = open('GoldBigrams.txt', 'r').readlines()
random.shuffle(gold_dataset3)
trainset_counter = 0
trainsize = len(gold_dataset3) - math.floor(len(gold_dataset3)/5)
gold_dataset3_train = set()
while trainset_counter < trainsize:
    trainset.add(gold_dataset3[trainset_counter])
    gold_dataset3_train.add(gold_dataset3[trainset_counter])
    trainset_counter = trainset_counter + 1

X_train = []
with open("features_total.txt", 'w') as f:
    #for phrase in hitcountExactTrain:
    for phrase in trainset:
        cleanedPhrase = phrase.rstrip(' ').lstrip(' ').strip('\n')
        if use_pos:
            pos = [0 for _ in range(len(all_pos_tags))]
        #oneGram = [0 for _ in range(43)]
		#pos = [0 for _ in range(45)]

        ngram_feature_vector_copy = list(ngram_feature_vector)
        if use_n_gram:
            for i in range(n_value + 1):
                if i == 0:
                    continue
                for gram in word2ngrams(cleanedPhrase, n = i):
                    if gram in ngram_feature_vector_map:
                        ngram_feature_vector_copy[ngram_feature_vector_map[gram]] = 1#ngram_feature_vector_copy[ngram_feature_vector_map[gram]] + 1

            ngram_feature_vector_copy.append(len(cleanedPhrase))
            features = str(ngram_feature_vector_copy)[1:-1]
            instance = [int(feature.strip()) for feature in features.split(',')]
        else:
            instance = []

        #instance = []

        if use_hitcount_ratio:
            instance.append(float(hitcountExactTrain[cleanedPhrase])/float(hitcountTrain[cleanedPhrase]))

        if use_ksvalue:
            instance.append(ksvalues[cleanedPhrase])
        if use_klvalue:
            instance.append(klvalues[cleanedPhrase])

        if use_partial_hitcount:
            left = left_partial_hitcount[cleanedPhrase]
            right = right_partial_hitcount[cleanedPhrase]
            #exact = hitcountExactTrain[cleanedPhrase]
            if right == 0 or left == 0:
                print(cleanedPhrase)
                exit()
            instance.append(hitcountExactTrain[cleanedPhrase]/left)

            instance.append(hitcountExactTrain[cleanedPhrase]/right)
            instance.append(hitcountExactTrain[cleanedPhrase]/((right+left)/2))
            #if exact == 0:
            #    instance.append(0)
            #    instance.append(0)
            #else:
            #    instance.append(float(left)/float(exact))
            #    instance.append(float(right_partial_hitcount[cleanedPhrase])/float(exact))
        #instance.append(hitcountExactTrain[cleanedPhrase])
        #instance.append(hitcountTrain[cleanedPhrase])

        #phrase_tokenized = nltk.word_tokenize(cleanedPhrase)
        #phrase_with_pos = nltk.pos_tag(phrase_tokenized)
        #for tuple in phrase_with_pos:
        #    if(tuple[1][-1] == '$'):
        #        pos[all_pos_tag_dic[tuple[1][:-1]]] = 1
        #        pos[all_pos_tag_dic[tuple[1][-1]]] = 1
        #    else:
        #        pos[all_pos_tag_dic[tuple[1]]] = 1
        if use_pos:
            pos_tags = pos_tags_train[cleanedPhrase]
            for tag in pos_tags.split():
                if(tag[-1] == '$'):
                    pos[all_pos_tag_dic[tag[:-1]]] = pos[all_pos_tag_dic[tag[:-1]]]+1
                else:
                    pos[all_pos_tag_dic[tag]] = pos[all_pos_tag_dic[tag]]+1

            features = str(pos)[1:-1]
            instance.extend(features.split(','))

        exact = hitcountExactTrain[cleanedPhrase]
        normal = hitcountTrain[cleanedPhrase]
        print("%s/----/%s" %(phrase, instance), file=f)
        X_train.append(instance)
        #X_test.append(instance)



#with open("tarinset.txt", 'w') as f:
#    for instance in X_train:
#        print(instance, file = f)

#clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.000000001)
clf = svm.OneClassSVM(nu = 0.2, kernel="linear")
#clf = svm.OneClassSVM(nu=0.1, kernel="sigmoid", gamma=0.01)
clf.fit(X_train)
#y_pred_train = clf.predict(X_train)

hitcountTest = {}
hitcountExactTest = {}
if use_pos:
    pos_tags_test = {}
    with open("Gold_tagged.txt", 'r') as f:
        for line in f:
            _phrase = line.rstrip(' ').lstrip(' ').strip('\n')
            if len(_phrase) > 0:
                #splitOut = _phrase.rsplit(' ', 1)
                pos_tags_test[re.sub(" \?", "?", re.sub(" 'll ", "'ll ", re.sub(" , ", ", ",re.sub(" ' ", "' ",re.sub("can not" , "cannot", re.sub(" n't ", "n't ", re.sub(" 't ", "'t ",re.sub(" 'm ", "'m ",re.sub(" 's", "'s"," ".join([token.split("_")[0] for token in _phrase.split()]))))))))))] =\
                    " ".join([token.split("_")[1] for token in _phrase.split()])

    with open("nonGold_tagged.txt", 'r') as f:
        for line in f:
            _phrase = line.rstrip(' ').lstrip(' ').strip('\n')
            if len(_phrase) > 0:
                #splitOut = _phrase.rsplit(' ', 1)
                pos_tags_test[re.sub(" \?", "?", re.sub(" 'll ", "'ll ", re.sub(" , ", ", ",re.sub(" ' ", "' ",re.sub("can not" , "cannot", re.sub(" n't ", "n't ", re.sub(" 't ", "'t ",re.sub(" 'm ", "'m ",re.sub(" 's", "'s"," ".join([token.split("_")[0] for token in _phrase.split()]))))))))))] =\
                    " ".join([token.split("_")[1] for token in _phrase.split()])

with open("features_exact_test.txt", 'r') as f:
    for line in f:
        _phrase = line.rstrip(' ').lstrip(' ').strip('\n')
        if len(_phrase) > 0:
            splitOut = _phrase.rsplit(' ', 1)
            hitcountExactTest[splitOut[0].rstrip(' ').lstrip(' ').strip('\n')] = int(splitOut[1].strip(' '))

with open("features_test.txt", 'r') as f:
    for line in f:
        _phrase = line.rstrip(' ').lstrip(' ').strip('\n')
        if len(_phrase) > 0:
            splitOut = _phrase.rsplit(' ', 3)
            hitcountTest[splitOut[0].rstrip(' ').lstrip(' ').strip('\n')] = int(float(splitOut[-3].strip(' ').split('.')[0]))

#with open("features_exact_test.txt", 'r') as f:
 #   for line in f:
  #      _phrase = line.rstrip(' ').lstrip(' ').strip('\n')
   #     if len(_phrase) > 0:
    #        splitOut = _phrase.rsplit(' ', 1)
     #       hitcountExactTest[splitOut[0].rstrip(' ').lstrip(' ').strip('\n')] = int(splitOut[1].strip(' '))

X_test = []
testset = set()
with open("GoldBigrams.txt", 'r') as f:
    for line in f:
        cleanedPhrase = line.rstrip(' ').lstrip(' ').strip('\n')
        if cleanedPhrase in gold_dataset3_train:
            continue
        if cleanedPhrase in testset:
            continue
        else:
            testset.add(cleanedPhrase)

        if use_pos:
            pos = [0 for _ in range(len(all_pos_tags))]

        ngram_feature_vector_copy = list(ngram_feature_vector)
        if use_n_gram:
            for i in range(n_value + 1):
                if i == 0:
                    continue
                for gram in word2ngrams(cleanedPhrase, n = i):
                    if gram in ngram_feature_vector_map:
                        ngram_feature_vector_copy[ngram_feature_vector_map[gram]] = 1#ngram_feature_vector_copy[ngram_feature_vector_map[gram]] + 1

            ngram_feature_vector_copy.append(len(cleanedPhrase))
            features = str(ngram_feature_vector_copy)[1:-1]
            instance = [int(feature.strip()) for feature in features.split(',')]
        else:
            instance = []

        #instance = []
        if use_hitcount_ratio:
            instance.append(float(hitcountExactTest[cleanedPhrase])/float(hitcountTest[cleanedPhrase]))

        if use_ksvalue:
            instance.append(ksvalues[cleanedPhrase])
        if use_klvalue:
            instance.append(klvalues[cleanedPhrase])
        if use_partial_hitcount:
            left = left_partial_hitcount[cleanedPhrase]
            right = right_partial_hitcount[cleanedPhrase]
            #exact = hitcountExactTrain[cleanedPhrase]
            #instance.append(left)
            #instance.append(right)
            if right == 0 or left == 0:
                print(cleanedPhrase)
                exit()
            instance.append(hitcountExactTest[cleanedPhrase]/left)

            instance.append(hitcountExactTest[cleanedPhrase]/right)
            instance.append(hitcountExactTest[cleanedPhrase]/((right+left)/2))
        #instance.append(hitcountExactTest[cleanedPhrase])
        #instance.append(hitcountTest[cleanedPhrase])
        #phrase_tokenized = nltk.word_tokenize(cleanedPhrase)
        #phrase_with_pos = nltk.pos_tag(phrase_tokenized)
        #for tuple in phrase_with_pos:
        #    if(tuple[1][-1] == '$'):
        #        pos[all_pos_tag_dic[tuple[1][:-1]]] = 1
        #       pos[all_pos_tag_dic[tuple[1][-1]]] = 1
        #    else:
        #        pos[all_pos_tag_dic[tuple[1]]] = 1

        if use_pos:
            pos_tags = pos_tags_test[cleanedPhrase]
            for tag in pos_tags.split():
                if(tag[-1] == '$'):
                    pos[all_pos_tag_dic[tag[:-1]]] = pos[all_pos_tag_dic[tag[:-1]]]+1
                else:
                    pos[all_pos_tag_dic[tag]] = pos[all_pos_tag_dic[tag]]+1

            features = str(pos)[1:-1]
            instance.extend(features.split(','))
        #features = str(pos)[1:-1]
        #instance.extend(features.split(','))

        X_test.append(instance)

#with open("testset1.txt", 'w') as f:
#    for instance in X_test:
#        print(instance, file = f)

y_pred_test = clf.predict(X_test)
TP = 0
FP = 0
TN = 0
FN = 0
for x in y_pred_test:
    if x == 1:
        TP = TP + 1
    else:
        FN = FN + 1


X_test = []
testset = set()
with open("nonGoldBigrams.txt", 'r') as f:
    for line in f:
        cleanedPhrase = line.rstrip(' ').lstrip(' ').strip('\n')
        if cleanedPhrase in testset:
            continue
        else:
            testset.add(cleanedPhrase)

        if use_pos:
            pos = [0 for _ in range(len(all_pos_tags))]

        ngram_feature_vector_copy = list(ngram_feature_vector)
        if use_n_gram:
            for i in range(n_value + 1):
                if i == 0:
                    continue
                for gram in word2ngrams(cleanedPhrase, n = i):
                    if gram in ngram_feature_vector_map:
                        ngram_feature_vector_copy[ngram_feature_vector_map[gram]] = 1#ngram_feature_vector_copy[ngram_feature_vector_map[gram]] + 1

            ngram_feature_vector_copy.append(len(cleanedPhrase))
            features = str(ngram_feature_vector_copy)[1:-1]
            instance = [int(feature.strip()) for feature in features.split(',')]
        else:
            instance = []


        #instance = []
        if use_hitcount_ratio:
            instance.append(float(hitcountExactTest[cleanedPhrase])/float(hitcountTest[cleanedPhrase]))

        if use_ksvalue:
            instance.append(ksvalues[cleanedPhrase])
        if use_klvalue:
            instance.append(klvalues[cleanedPhrase])
        if use_partial_hitcount:
            left = left_partial_hitcount[cleanedPhrase]
            right = right_partial_hitcount[cleanedPhrase]
            #exact = hitcountExactTrain[cleanedPhrase]
            #instance.append(left)
            #instance.append(right)
            if right == 0 or left == 0:
                print(cleanedPhrase)
                exit()
            instance.append(hitcountExactTest[cleanedPhrase]/left)

            instance.append(hitcountExactTest[cleanedPhrase]/right)
            instance.append(hitcountExactTest[cleanedPhrase]/((right+left)/2))
        #phrase_tokenized = nltk.word_tokenize(cleanedPhrase)
        #phrase_with_pos = nltk.pos_tag(phrase_tokenized)
        #for tuple in phrase_with_pos:
        #    if(tuple[1][-1] == '$'):
        #        pos[all_pos_tag_dic[tuple[1][:-1]]] = 1
        #        pos[all_pos_tag_dic[tuple[1][-1]]] = 1
        #    else:
        #        pos[all_pos_tag_dic[tuple[1]]] = 1

        if use_pos:
            pos_tags = pos_tags_test[cleanedPhrase]
            for tag in pos_tags.split():
                if(tag[-1] == '$'):
                    pos[all_pos_tag_dic[tag[:-1]]] = pos[all_pos_tag_dic[tag[:-1]]]+1
                else:
                    pos[all_pos_tag_dic[tag]] = pos[all_pos_tag_dic[tag]]+1

            features = str(pos)[1:-1]
            instance.extend(features.split(','))
        #features = str(pos)[1:-1]
        #instance.extend(features.split(','))
        #instance.append(hitcountExactTest[cleanedPhrase])
        #instance.append(hitcountTest[cleanedPhrase])
        X_test.append(instance)

#with open("testset2.txt", 'w') as f:
#    for instance in X_test:
#        print(instance, file = f)

y_pred_test = clf.predict(X_test)
for x in y_pred_test:
    if x == 1:
        FP = FP + 1
    else:
        TN = TN + 1

print("TP = %d, FP = %d" % (TP, FP))
print("TN = %d, FN = %d" % (TN, FN))
precision = TP/(TP+FP)
recall = TP/(TP+FN)
print("Precision = %f" % (precision))
print("recall = %f" % (recall))
print("F-Score = %f"% (2*(precision * recall)/(precision + recall)))
