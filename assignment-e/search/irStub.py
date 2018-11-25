# Nigel Ward, UTEP, October 2018
# Speech and Language Processing
# Assignment E: Information Retrieval

# This is just a skeleton that needs to be fleshed out.
# It is not intended as an example of good Python style

import sys
reload(sys)
sys.setdefaultencoding('utf8')

def parseAlternatingLinesFile(file):     #-----------------------------
   # read a sequence of pairs of lines, e.g. text of webpage(s), name/URL
   sequenceA = []
   sequenceB = [] 
   fp = open(file, 'r')
   expectingA = True 
   for line in fp.readlines():
       if expectingA:
           sequenceA.append(line.rstrip())
           expectingA = False
       else:
           sequenceB.append(line.rstrip())
           expectingA = True
   fp.close()
   return sequenceA, sequenceB


def characterTrigrams(text):         #----------------------------
  return [text[i:i+3] for i in range(len(text)-3+1)]

# EXPERIMENT 1
def wordUnigram(text):
    return text.split()

# EXPERIMENT 2
def wordBigram(text, stop_words=None):
    from sklearn.feature_extraction.text import CountVectorizer

    bigram = CountVectorizer(ngram_range=(1,2), stop_words=stop_words).build_analyzer()

    return bigram(text)
    

def computeFeatures(text, trigramInventory):        #-----------------------------
    # catches the similarities between  "social" and "societal" etc. 
    # but really should be replaced with something better
    trigrams = characterTrigrams(text)
    counts = {}
    for trigram in trigrams:
        if trigram in trigramInventory:
            if trigram in counts:
                counts[trigram] += 1
            else:
                counts[trigram] = 1              
    return counts
   

def computeSimilarity(dict1, dict2):   #-----------------------------
    # ad hoc and inefficient
    matchCount = 0
    for tri in dict1:
        if tri in dict2:
            #print "match on " + tri
            matchCount += 1 
    similarity = matchCount / (float(len(dict2)) + 0.0000000000000000001)
    #print 'similarity %.3f' % similarity
    return similarity


def retrieve(queries, trigramInventory, archive):      #-----------------------------
    # returns an array: for each query, the top 3 results found
    top3sets = [] 
    for query in queries:
        #print 'query is ' + query
        q = computeFeatures(query, trigramInventory)
        #print 'query features are '
        #print q
        similarities = [] 
        for d in archive:
            similarities.append(computeSimilarity(q, d))
        #print similarities 
        top3indices = np.argsort(similarities)[0:3]
        #print "top three indices are "
        #print top3indices
        top3sets.append(top3indices)  
    return top3sets

def valueOfSuggestion(result, position, targets):   #-----------------------------
    weight = [1.0, .5, .25]
    if result in targets:
        return weight[max(position, targets.index(result))]
    else:
        return 0


def scoreResults(results, targets):   #-----------------------------
    merits = [valueOfSuggestion(results[i], i, targets) for i in [0,1,2]]
    return sum(merits)


def scoreAllResults(queries, results, targets, descriptor):   #-----------------------------
    print '\nScores for ' + descriptor
    scores = [] 
    for q, r, t in zip(queries, results, targets):
       print 'for query: ' + q,
       print ' results = ',
       print r,
       print ' targets = ',
       print t,
       s = scoreResults(r, t)
       print '  score = %.3f' % s
       scores.append(s)
    overallScore = np.mean(scores)
    print 'all scores',
    print scores
    print 'overall score is %.3f' % overallScore
    return overallScore


def pruneUniqueNgrams(ngrams):        # ----------------------
    twoOrMore = {} 
    print 'before pruning: %d ngrams across all documents' % len(ngrams)
    for key in ngrams:
        if ngrams[key] > 1:
            twoOrMore[key] = ngrams[key]
    print 'after pruning: %d ngrams across all documents' % len(twoOrMore)
    return twoOrMore

# def findAllNgrams(contents):          # ----------------------
#     allTrigrams = {}
#     merged = ''
#     for text in contents:
#         for tri in characterTrigrams(text):
#             if tri in allTrigrams:
#                 allTrigrams[tri] += 1
#             else:
#                 allTrigrams[tri] = 1
#     return allTrigrams

# EXPERIMENT 1
def findAllUnigrams(contents):
    allUnigrams = {}
    for text in contents:
        for unigram in wordUnigram(text):
            if unigram in allUnigrams:
                allUnigrams[unigram] += 1
            else:
                allUnigrams[unigram] = 1
    return allUnigrams

# EXPERIMENT 1
def findAllBigrams(contents):
    allBigrams = {}
    for text in contents:
        for bigram in wordBigram(text, stop_words='english'):
            if bigram in allBigrams:
                allBigrams[bigram] += 1
            else:
                allBigrams[bigram] = 1
    return allBigrams

def targetNumbers(targets, nameInventory):        # ----------------------
    # targets is a list of strings, each a sequence of names
    targetIDs = []
    for target in targets:
      threeNumbers = [] 
      for name in target.split():
          threeNumbers.append(nameInventory.index(name))
      targetIDs.append(threeNumbers)
    return targetIDs
          

def stemming(text):
    from nltk.stem.snowball import SnowballStemmer
    import nltk
    nltk.download('stopwords')
    stemmer = SnowballStemmer('english', ignore_stopwords=True)
    return ' '.join([ stemmer.stem(t) for t in text.split() ])

# main ----------------------------------------------------
import sys, numpy as np
import string

print('......... irStub .........')
contents, names =  parseAlternatingLinesFile('data/csFaculty.txt') 

# EXPERIMENT 2
contents = [ doc.translate(None, string.punctuation) for doc in contents ]

# EXPERIMENT 4
contents = [ stemming(doc) for doc in contents ]


print 'read in pages for ',
print names
trigramInventory = pruneUniqueNgrams(findAllUnigrams(contents))
archive = [computeFeatures(line, trigramInventory) for line in contents]

if len(sys.argv) >= 2 and (sys.argv[1] == 'yesThisReallyIsTheFinalRun'):
    queryFile = 'data/testQueries.txt'
else: 
    queryFile = 'data/trainingQueries.txt'

queries, targets = parseAlternatingLinesFile(queryFile)
queries = [ stemming(q) for q in queries ]

targetIDs = targetNumbers(targets, names)
results = retrieve(queries, trigramInventory, archive)
modelName = 'silly character trigram model'
scoreAllResults(queries, results, targetIDs, modelName + ' on ' + queryFile)


