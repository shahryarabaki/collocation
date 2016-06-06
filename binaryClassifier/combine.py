from sklearn import svm
import re
import string
import random
import itertools
import os
import math
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.externals.six import StringIO

use_balanced = True
classifier = "NB"
classifier = "logistic regression"
#classifier = "linear regression"
classifier = "KNN"
classifier = "tree"
classifier = "SVM"
classifier = "randomForsest"
def word2ngrams(text, n=3):
    return ["".join(j) for j in zip(*[text[i:] for i in range(n)])]

create_weka_file = True
use_pos = False
use_hitcount_ratio = True
use_ksvalue = True
use_klvalue = True
use_partial_hitcount = True
use_n_gram = True
n_value = 1
alphabet = string.ascii_lowercase
use_datase3_only = True
use_3gram_dataset = True
use_4gram_dataset = True
use_5gram_dataset = True
use_6gram_dataset = True
use_7gram_dataset = True
use_8gram_dataset = True
#alphabet += "\'"
#alphabet += " "

#Dataset 3
#dataset_size = 2729
#test_size = 545 #2729/5
#train_size = 2184

#Dataset 2 and 3
#dataset_size = 4403
#test_size = 880 #4403/5
#train_size = 3523

#Dataset 1 and 3
dataset_size = 3129
test_size = 625 #4403/5
train_size = 2504

#Dataset 1 and 2 and 3
dataset_size = 4689
test_size = 937 #4403/5
train_size = 3752

if use_3gram_dataset:
    #Dataset 1 and 2 and 3 and 3gram
    dataset_size = 7439
    test_size = 1487 #7424/5
    train_size = 5952

if use_4gram_dataset:
    #Dataset 1 and 2 and 3 and 3gram and 4gram
    dataset_size = 9946
    test_size = 1989 #9931/5
    train_size = 7957

if use_5gram_dataset:
    #Dataset 1 and 2 and 3 and 3gram and 4gram and 5gram
    dataset_size = 12340
    test_size = 2468 #9931/5
    train_size = 9872

if use_6gram_dataset:
    #Dataset 1 and 2 and 3 and 3gram and 4gram and 5gram and 6gram
    dataset_size = 14621
    test_size = 2924 #9931/5
    train_size = 11697

if use_7gram_dataset:
    #Dataset 1 and 2 and 3 and 3gram and 4gram and 5gram and 6gram and 7gram
    dataset_size = 16790
    test_size = 3358 #9931/5
    train_size = 13432

if use_8gram_dataset:
    #Dataset 1 and 2 and 3 and 3gram and 4gram and 5gram and 6gram and 7gram and 8gram
    dataset_size = 18848
    test_size = 3769 #9931/5
    train_size = 15079

if use_datase3_only:
    #Dataset 3 and 3gram and 4gram and 5gram and 6gram and 7gram and 8gram
    dataset_size = 16809
    test_size = 3361 #9931/5
    train_size = 13448


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
if use_pos:
    pos_tags_train = {}
    with open("all_taged.txt", 'r') as f:
        for line in f:
            _phrase = line.rstrip(' ').lstrip(' ').strip('\n')
            if len(_phrase) > 0:
                #splitOut = _phrase.rsplit(' ', 1)
                pos_tags_train[re.sub(" \?", "?", re.sub(" 'll ", "'ll ", re.sub(" , ", ", ",re.sub(" ' ", "' ",re.sub("can not" , "cannot", re.sub(" n't ", "n't ", re.sub(" 't ", "'t ",re.sub(" 'm ", "'m ",re.sub(" 's", "'s"," ".join([token.split("_")[0] for token in _phrase.split()]))))))))))] =\
                    " ".join([token.split("_")[1] for token in _phrase.split()])

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

hitcount = {}
hitcountExact = {}
with open("features_exact.txt", 'r') as f:
    for line in f:
        _phrase = line.rstrip(' ').lstrip(' ').strip('\n')
        if len(_phrase) > 0:
            splitOut = _phrase.rsplit(' ', 1)
            hitcountExact[splitOut[0].rstrip(' ').lstrip(' ').strip('\n')] = int(splitOut[1].strip(' '))

with open("features.txt", 'r') as f:
    for line in f:
        _phrase = line.rstrip(' ').lstrip(' ').strip('\n')
        if len(_phrase) > 0:
            splitOut = _phrase.rsplit(' ', 3)
            hitcount[splitOut[0].rstrip(' ').lstrip(' ').strip('\n')] = int(float(splitOut[-3].strip(' ').split('.')[0]))
with open("hitcount_both.txt", 'r') as f:
    for line in f:
        _phrase = line.rstrip(' ').lstrip(' ').strip('\n')
        if len(_phrase) > 0:
            splitOut = _phrase.split('\t')
            hitcountExact[splitOut[0].rstrip(' ').lstrip(' ').strip('\n')] = int(splitOut[2].strip(' '))
            hitcount[splitOut[0].rstrip(' ').lstrip(' ').strip('\n')] = int(splitOut[1].strip(' '))

gold = set()
nonGold = set()
gold_list = []
nonGold_list = []

#output = open("clean_N_grams_3.txt", 'w')
#for text in nonGold:
#    print(text, file=output)
#output.close()
#exit()

with open("nonGoldBigrams.txt", 'r') as f:
    for line in f:
        cleanedPhrase = line.rstrip(' ').lstrip(' ').strip('\n')
        nonGold.add(cleanedPhrase)
        nonGold_list.append(cleanedPhrase)

if not use_datase3_only:
    with open("Dataset1.txt", 'r') as f:
        for line in f:
            cleanedPhrase = line.rstrip(' ').lstrip(' ').strip('\n').strip()
            gold.add(cleanedPhrase)
            gold_list.append(cleanedPhrase)

if not use_datase3_only:
    with open("Dataset2.txt", 'r') as f:
        for line in f:
            cleanedPhrase = line.rstrip(' ').lstrip(' ').strip('\n').strip()
            gold.add(cleanedPhrase)
            gold_list.append(cleanedPhrase)

with open("GoldBigrams_withNE.txt", 'r') as f:
#with open("GoldBigrams_noNE.txt", 'r') as f:
    for line in f:
        cleanedPhrase = line.rstrip(' ').lstrip(' ').strip('\n')
        gold.add(cleanedPhrase)
        gold_list.append(cleanedPhrase)

if use_3gram_dataset:
    with open("GoldTrigrams_withNE.txt", 'r') as f:
        for line in f:
            cleanedPhrase = line.rstrip(' ').lstrip(' ').strip('\n')
            gold.add(cleanedPhrase)
            gold_list.append(cleanedPhrase)

    with open("N_grams_3.txt", 'r') as f:
        for line in f:
            cleanedPhrase = line.rstrip(' ').lstrip(' ').strip('\n')
            nonGold.add(cleanedPhrase)
            nonGold_list.append(cleanedPhrase)

if use_4gram_dataset:
    with open("N_grams_4.txt", 'r') as f:
        for line in f:
            cleanedPhrase = line.rstrip(' ').lstrip(' ').strip('\n')
            nonGold.add(cleanedPhrase)
            nonGold_list.append(cleanedPhrase)
    with open("Gold4grams_withNE.txt", 'r') as f:
        for line in f:
            cleanedPhrase = line.rstrip(' ').lstrip(' ').strip('\n')
            gold.add(cleanedPhrase)
            gold_list.append(cleanedPhrase)

if use_5gram_dataset:
    with open("N_grams_5.txt", 'r') as f:
        for line in f:
            cleanedPhrase = line.rstrip(' ').lstrip(' ').strip('\n')
            nonGold.add(cleanedPhrase)
            nonGold_list.append(cleanedPhrase)

    with open("Gold5grams_withNE.txt", 'r') as f:
        for line in f:
            cleanedPhrase = line.rstrip(' ').lstrip(' ').strip('\n')
            gold.add(cleanedPhrase)
            gold_list.append(cleanedPhrase)

if use_6gram_dataset:
    with open("N_grams_6.txt", 'r') as f:
        for line in f:
            cleanedPhrase = line.rstrip(' ').lstrip(' ').strip('\n')
            nonGold.add(cleanedPhrase)
            nonGold_list.append(cleanedPhrase)

    with open("Gold6grams_withNE.txt", 'r') as f:
        for line in f:
            cleanedPhrase = line.rstrip(' ').lstrip(' ').strip('\n')
            gold.add(cleanedPhrase)
            gold_list.append(cleanedPhrase)

if use_7gram_dataset:
    with open("N_grams_7.txt", 'r') as f:
        for line in f:
            cleanedPhrase = line.rstrip(' ').lstrip(' ').strip('\n')
            nonGold.add(cleanedPhrase)
            nonGold_list.append(cleanedPhrase)

    with open("Gold7grams_withNE.txt", 'r') as f:
        for line in f:
            cleanedPhrase = line.rstrip(' ').lstrip(' ').strip('\n')
            gold.add(cleanedPhrase)
            gold_list.append(cleanedPhrase)

if use_8gram_dataset:
    with open("N_grams_8.txt", 'r') as f:
        for line in f:
            cleanedPhrase = line.rstrip(' ').lstrip(' ').strip('\n')
            nonGold.add(cleanedPhrase)
            nonGold_list.append(cleanedPhrase)

    with open("Gold8grams_withNE.txt", 'r') as f:
        for line in f:
            cleanedPhrase = line.rstrip(' ').lstrip(' ').strip('\n')
            gold.add(cleanedPhrase)
            gold_list.append(cleanedPhrase)


print("************************")
print("True Size = %d, False Size = %d" % (len(gold_list), len(nonGold_list)))
total_Fscore = 0
total_recall = 0
#lines = open('Dataset1&2&3_3gram_4gram_5gram.txt').readlines()
#lines = open('Dataset3_cleaned_3gram_4gram_5gram_6gram_7gram_8gram_withNE.txt').readlines()
#random.shuffle(lines)
#open('temp.txt', 'w').writelines(lines)
#exit()
FScore = []
all_precision = []
all_recall = []
globalTP = 0
globalFP = 0
globalTN = 0
globalFN = 0
for fold in range(5):
    X_test = []
    classes_test = []
    X_train = []
    classes_train = []
    counter = 0
    with open("temp.txt", 'r') as f:
        for line in f:
            cleanedPhrase = line.rstrip(' ').lstrip(' ').strip('\n').strip()
            if use_pos:
                pos = [0 for _ in range(len(all_pos_tags))]

            ngram_feature_vector_copy = list(ngram_feature_vector)
            if use_n_gram:
                for i in range(n_value + 1):
                    if i == 0:
                        continue
                    for gram in word2ngrams(cleanedPhrase, n = i):
                        if gram in ngram_feature_vector_map:
                            ngram_feature_vector_copy[ngram_feature_vector_map[gram]] = ngram_feature_vector_copy[ngram_feature_vector_map[gram]] + 1

                ngram_feature_vector_copy.append(len(cleanedPhrase))
                features = str(ngram_feature_vector_copy)[1:-1]
                instance = [int(feature.strip()) for feature in features.split(',')]
            else:
                instance = []

            #instance = []
            if use_hitcount_ratio:
                if hitcount[cleanedPhrase] == 0:
                    instance.append(float(0))
                else:
                    instance.append(float(hitcountExact[cleanedPhrase])/float(hitcount[cleanedPhrase]))

            if use_ksvalue:
                instance.append(ksvalues[cleanedPhrase])
            if use_klvalue:
                instance.append(klvalues[cleanedPhrase])
            if use_partial_hitcount:
                left = left_partial_hitcount[cleanedPhrase]
                right = right_partial_hitcount[cleanedPhrase]
                #exact = hitcountExactTrain[cleanedPhrase]
                #instance.append(int(left))
                #instance.append(int(right))
                if right == 0 or left == 0:
                    #print("partial hitcount")
                    #print(cleanedPhrase)
                    left = right = 1
                instance.append(hitcountExact[cleanedPhrase]/left)

                instance.append(hitcountExact[cleanedPhrase]/right)
                instance.append(hitcountExact[cleanedPhrase]/((right+left)/2))

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

            if counter > test_size * fold and counter < test_size *fold + test_size:
                X_test.append(instance)
                if cleanedPhrase in gold:
                    classes_test.append(1)
                elif cleanedPhrase in nonGold:
                    classes_test.append(0)
                else:
                    print("Error-test")
                    print(cleanedPhrase)
                    exit(0)
            else:
                X_train.append(instance)
                if cleanedPhrase in gold:
                    classes_train.append(1)
                elif cleanedPhrase in nonGold:
                    classes_train.append(0)
                else:
                    print("Error-train")
                    print(cleanedPhrase)
                    exit(0)
            counter = counter + 1

    if classifier == 'SVM':
        #clf = svm.SVC(kernel='linear', degree=2, gamma = 0.00001)
        #clf = svm.SVC(kernel='linear', class_weight='balanced')
        if use_balanced:
            clf = svm.SVC(kernel='rbf', class_weight='balanced')
        else:
            clf = svm.SVC(kernel='rbf')
        #clf = svm.SVC(kernel='rbf', degree=2, gamma = 0.00001)
    elif classifier == 'tree':
        if use_balanced:
            clf = tree.DecisionTreeClassifier(class_weight='balanced')
        else:
            clf = tree.DecisionTreeClassifier()
    elif classifier == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=3)
    elif classifier == 'NB':
        clf = GaussianNB()
    elif classifier == 'linear regression':
        clf = linear_model.LinearRegression(normalize=True)
    elif classifier == 'logstic regression':
        #clf = linear_model.LogisticRegression(C=1e5)
        if use_balanced:
            clf = linear_model.LogisticRegression(class_weight='balanced')
        else:
            clf = linear_model.LogisticRegression()
    else:
        clf = RandomForestClassifier(n_estimators=10)

    print("Train size = %d , Test size = %d" % (len(X_train), len(X_test)))

    clf.fit(X_train, classes_train)
    #with open("colocation.dot", 'w') as f:
    #    f = tree.export_graphviz(clf, out_file=f)
    #exit(0)

    if create_weka_file:
        features_file = open("features_weka_dataset3_bigram_withNE.arff",'w')


        print("@RELATION collocation", file=features_file)
        print("", file=features_file)
        flags =[use_n_gram, use_hitcount_ratio, use_ksvalue, use_klvalue, use_partial_hitcount, use_partial_hitcount, use_partial_hitcount]
        #flags =[use_hitcount_ratio, use_ksvalue, use_klvalue, use_partial_hitcount, use_partial_hitcount, use_partial_hitcount]
        number_of_features_without_ngram = sum(flags)
        number_of_ngram_features = len(X_train[0]) - number_of_features_without_ngram
        if use_n_gram:
            for gram in range(number_of_ngram_features):
                print("@ATTRIBUTE " + str(gram) + "gram" + " REAL", file=features_file)
            print("@ATTRIBUTE len" + " NUMERIC", file=features_file)

        if use_hitcount_ratio:
            print("@ATTRIBUTE hitcount_ratio REAL", file=features_file)
        if use_ksvalue:
            print("@ATTRIBUTE ksvalue REAL", file=features_file)
        if use_klvalue:
            print("@ATTRIBUTE klvalue REAL", file=features_file)
        if use_partial_hitcount:
            print("@ATTRIBUTE left_ratio REAL", file=features_file)
            print("@ATTRIBUTE right_ratio REAL", file=features_file)
            print("@ATTRIBUTE average_ratio REAL", file=features_file)

        print("@ATTRIBUTE class {0, 1}", file=features_file)
        print("", file=features_file)
        print("@Data", file=features_file)
        for totalCounter in range(len(X_train)):
            print(",".join(str(x) for x in X_train[totalCounter]) + "," + str(classes_train[totalCounter]), file=features_file)

    y_pred_test = clf.predict(X_test)
	

    if create_weka_file:
        for totalCounter in range(len(X_test)):
            print(",".join(str(x) for x in X_test[totalCounter]) + "," + str(classes_test[totalCounter]), file=features_file)

        features_file.close()
        exit(0)
    #exit(0)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    counter = 0
    #score = log_loss(y_pred_test, classes_test)
    for i in range(len(y_pred_test)):
        predicted = y_pred_test[i]
        actual = classes_test[i]
        if actual == 1:
            if predicted == actual:
                TP = TP + 1
            else:
                FN = FN + 1
        else:
            if predicted == actual:
                TN = TN + 1
            else:
                FP  = FP + 1

    globalFN = globalFN + FN
    globalFP = globalFP + FP
    globalTN = globalTN + TN
    globalTP = globalTP + TP
    print("Fold number " + str(fold))
    print("TP = %d, FP = %d" % (TP, FP))
    print("TN = %d, FN = %d" % (TN, FN))
    precision = TP/(TP+FP)
    all_precision.append(precision)
    recall = TP/(TP+FN)
    all_recall.append(recall)
    print("Precision = %f" % (precision))
    print("recall = %f" % (recall))
    if precision + recall != 0:
        print("F-Score = %f"% (2*(precision * recall)/(precision + recall)))
        FScore.append(2*(precision * recall)/(precision + recall))
        #total_Fscore = total_Fscore + (2*(precision * recall)/(precision + recall))


print("***************")
print("Average precision = %f" % (numpy.mean(all_precision)))
print("Average recall = %f" % (numpy.mean(all_recall)))
print("Average Fscore = %f" % (numpy.mean(FScore)))
print("Fscore StdDev= %f" % (numpy.std(FScore)))

precision = globalTP/(globalTP+globalFP) * 100
recall = globalTP/(globalTP+globalFN) * 100
fscore = 2*(precision * recall)/(precision + recall)
print("Latex Table")
print("& %.2f\t& %.2f\t& %.2f \\\\ \hline" % (precision, recall, fscore ))
#os.remove("temp.txt")
