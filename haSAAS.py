#
# Implementation of the algorithms presented in
#
#   A. Meisl, G. Neumann: Towards Better Support for Machine-Assisted
#   Human Grading of Short-Text Answers, Presented at HICSS 55:
#   Hawai’i International Conference on System Sciences, January 4,
#   2022
#
# Copyright (C) 2021 Alexander Meisl
# Copyright (C) 2021 Gustaf Neumann
#
# This software is covered by the MIT License
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
__author__ = "Alexander Meisl and Gustaf Neumann"

#
# Flags controlling the runs
#
verbose = True
verbose = False

import treetaggerwrapper as ttw
import nltk
import jaro
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

# Basic Python stuff
import os, copy, glob, pickle
from collections import defaultdict
import csv

# The dict lang_stopwords is used for the TfidfVectorizer. For
# identical results with earlier versions, pass the string "english"
# rather than an explicit list of stop words.
#
lang_stopwords = {
    "en":  nltk.corpus.stopwords.words('english'),
    "de":  nltk.corpus.stopwords.words('german')
    }

# Data structures:
#
# The script can processes multiple exams with submissions, which are
# read into a structure of the form of an array of assessement dicts
# with the following components:
#
#   assessments = [
#     {
#        "id": "resources/testdata/en/xxx.txt",
#        "lang": "en",
#        "question": "",
#        "submissions": [....],
#        "submissions_tagged": [....],
#        "submissions_lemmatized": [....],
#        "points": [....],
#        "percentages": [....],
#     }, ...
#   ]


def load_mohler_csv(fn, lang):
    """
    Load assessments from the dataset originally provided
    by Mohler et.al.
    https://www.aclweb.org/anthology/P11-1076.pdf

    The .csv file has to contain the columns "id", "question",
    "student_answer", "score_avg".

    The file can be loaded via:
        preprocess("resources/testdata/en/*.mohlercsv")

    The function returns an array of single-item assessments.

    """
    print("load_mohler_csv " + fn);
    path = os.path.splitext(fn)[0]
    question = {}
    submissions = defaultdict(list)
    points      = defaultdict(list)
    percentages = defaultdict(list)
    for row in csv.DictReader(open(fn)):
        id = path + "-" + row["id"]
        submissions[id].append(row["student_answer"].strip())
        question[id] = row["question"]
        pts = row["score_avg"]
        percentage = float(pts)/5
        percentages[id].append(percentage)
        points[id].append(pts)

    assessments = []
    for id in submissions.keys():
        assessments.append({
            "id": id,
            "lang": lang,
            "submissions": submissions[id],
            "question": question[id],
            "points": points[id],
            "percentages": percentages[id]
            });
    return assessments


#
# Loader functions, based on file extension
#
assessment_load_funcs = {
    ".mohlercsv": load_mohler_csv,
    ".csv": load_mohler_csv
    }

def preprocess(pattern, lang=""):
    """
    Load assessments from files and preprocess it.
    The content is parsed into single item assessments
    that contain always an id and the submissions.
    The submissions are POS tagged and lemmatized.
    """
    assessments = []

    for fn in glob.glob(pattern):
        extension = os.path.splitext(fn)[1]
        if lang == "":
            # get language from standard path convention,
            # e.g. resources/testdata/de/503185999-graded
            language = os.path.split(os.path.split(fn)[0])[1]
            print("determined lang '" + language + "' from " + fn)
        else:
            # use provided language
            language = lang
        assessments.extend(assessment_load_funcs[extension](fn, language))

    for assessment in assessments:

        assessment["submissions_tagged"] = load_persisted(
            assessment["submissions"],
            assessment["lang"],
            tagged,
            assessment["id"] + "-tagged")

        if verbose:
            print("... lemmatize " + assessment["id"])
            c = 0
            for submission_tagged in assessment["submissions_tagged"]:
                c = c + 1
                print(str(c) + ': ' + "\n".join(str(t) for t in submission_tagged) + "\n----")

        assessment["submissions_lemmatized"] = load_persisted(
            assessment["submissions_tagged"],
            assessment["lang"],
            lemmatized,
            assessment["id"] + "-lemmatized")
        print(assessment["id"] \
                + ": submissions " + str(len(assessment["submissions"])) \
                + " tagged_submissions " + str(len(assessment["submissions_tagged"])) \
                + " lemmatized_submissions " + str(len(assessment["submissions_lemmatized"])))

        assessment["similarities"] = []
        assessment["variabilities"] = []
    return assessments

#
# Base statistics
#
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def pearson(predictions, targets):
    # We could use as well scipy.stats.pearsonr(x,y) which returns
    # correlatopm and the p-value
    return np.corrcoef(predictions, targets)[0][1]

def weighted_avg(means, weights):
    weighted_sum = []
    for mean, weight in zip(means, weights):
        weighted_sum.append(mean * weight)

    return sum(weighted_sum) / sum(weights)

#
# Pretty-printing
#

def pretty_id(assessment):
    return assessment["id"].replace("resources/testdata/","").replace("en/","")


def pretty_similarity(value):
    if value > 0.35:
        rating = "high"
    elif value < 0.20:
        rating = "low"
    else:
        rating = "medium"
    return "similarity " + str(value) + " suggests rating " + rating

def pretty_variability(vector, pos):
    min_value = min(vector)
    min_index = vector.index(min_value)
    if min_value > 0.80:
        clone_msg = " (quite different from other submissions)"
    else:
        c = 0
        clones = []
        for v in vector:
            c = c + 1
            if v < 0.6 and c != pos:
                clones.append(c)
        if len(clones) > 0:
            clone_msg = " (potential clones " + str(clones) + ")"
        else:
            clone_msg = ""
    return "lowest variability " + str(min_value) + " with submission " + str(min_index+1) + clone_msg

def print_submissions(assessment):
    print("ID " + assessment["id"])
    print(assessment["question"] + "\n----")
    c = 0
    if len(assessment["similarities"]) == 0:
        for submission in assessment["submissions"]:
            c = c + 1
            print(str(c) + ': '
                      + pretty_similarity( assessment["similarity"]) + "\n"
                      +  submission + "\n----")
    else:
        for submission, similarity, variability, percentage in \
          zip(assessment["submissions"],
                  assessment["similarities"],
                  assessment["variabilities"],
                  assessment["percentages"]):
            c = c + 1
            print(str(c) + ': '
                      + pretty_similarity(similarity) + ", "
                      + pretty_variability(variability, c)
                      + " -- " + str(percentage)
                      + " -- " + assessment["id"].replace("resources/testdata/","")
                      + "\n" +  submission + "\n----")

def print_correlations(label, similarities, percentage):
    np_sim = np.array(similarities)
    np_perc = np.array(percentage)
    p = pearson(np_sim, np_perc)
    r = rmse(np_sim, np_perc)
    print("\ncorrelation between", label,"similarities and percentage:", p,"RMSE:", r,
              "(n=" + str(len(similarities)) + ")")
    return (p, r)

#
# Natural language processing
#
def tagged(submission, lang):
    tagger = ttw.TreeTagger(TAGLANG=lang)
    tags = tagger.tag_text(submission,notagurl=True,notagemail=True,notagip=True,notagdns=True,nosgmlsplit=True)
    return ttw.make_tags(tags)

def lemmatized(tagged_submission, lang):
    lemmata = [c for a, b, c in tagged_submission]
    return [''.join(w) for w in lemmata]

def filter_punctuations(tagged):
    return [(a, b, c) for a, b, c in tagged if b != "SENT"]

#
# Managing persisted function results
#
def load_persisted(items, lang, func, fullfn):
    #
    # Load tagged data if it exists.
    #
    fullfn = "cache/" + fullfn
    if not os.path.exists(os.path.dirname(fullfn)):
        os.makedirs(os.path.dirname(fullfn))

    if os.path.exists(fullfn):
        with open(fullfn, 'rb') as f:
            result = pickle.load(f)
    else:
        result = [func(item, lang) for item in items]

        with open(fullfn, 'wb') as f:
            pickle.dump(result, f)

    return result

def compute_persisted(assessment, key, func):
    #
    # Load tagged data if it exists.
    #
    fullfn = "cache/" + assessment["id"] + "-" + key
    if not os.path.exists(os.path.dirname(fullfn)):
        os.makedirs(os.path.dirname(fullfn))

    if os.path.exists(fullfn):
        with open(fullfn, 'rb') as f:
            result = pickle.load(f)
    else:
        result = func(assessment)

        with open(fullfn, 'wb') as f:
            pickle.dump(result, f)

    assessment[key] = result;
    return result

#
# Similarity measures on assessment structures
#
def compute_jarowinkler(assessment):
    nrItems = len(assessment["submissions"])

    result = []
    for s1 in assessment["submissions"]:
        sNr1 = assessment["submissions"].index(s1)

        row = []
        for s2 in assessment["submissions"]:
            sNr2 = assessment["submissions"].index(s2)
            row.append(jaro.jaro_winkler_metric(s1, s2))

        result.append(row)

    return result


def compute_completeness_similarities(assessment):
    result = []
    nrItems = len(assessment["submissions"])

    tagged     = assessment["submissions_tagged"]
    lemmatized = assessment["submissions_lemmatized"]

    # For every submission: build a corpus with two
    # data elements, where
    #  * the first one is the lemmatized submission and
    #  * the second one consists of all other submissions.

    for item in assessment["submissions"]:
        data = [None] * 2
        ref = ""
        doc = ""
        itemNr = assessment["submissions"].index(item)

        for i in range(nrItems):
            if itemNr == i:
                doc += str(lemmatized[i])
            else:
                ref += str(lemmatized[i])

        data[0] = doc
        data[1] = ref

        # Initialize the tf-idf vectorizer
        vect = TfidfVectorizer(min_df=1, stop_words=lang_stopwords[assessment["lang"]])

        # Generate the tf-idf (term frequency–inverse document
        # frequency) vectors for the corpus
        tfidf = vect.fit_transform(data)

        pairwise_sim = tfidf * tfidf.T

        arr = pairwise_sim.toarray()

        result.append(round(arr[0][1], 4))

    assessment["similarities"] = result
    return result


def compute_variabilities(assessment):
    submissions = assessment["submissions"]
    result = []
    for j in range(0, len(submissions)):
        tmp = []
        for i in range(0, len(submissions)):
            tmp.append(compute_variabilities_pair(assessment, j, i))
        result.append(tmp)
    assessment["variabilities"] = result
    return result


def compute_variabilities_pair(assessment, i, r):
    input = assessment["submissions"][i]
    ref   = assessment["submissions"][r]

    input_tagged = filter_punctuations(assessment["submissions_tagged"][i])
    ref_tagged   = filter_punctuations(assessment["submissions_tagged"][r])
    input_tagged_words = [elem[2].lower() for elem in input_tagged]
    ref_tagged_words   = [elem[2].lower() for elem in ref_tagged]

    # trigrams
    input_trigrams = [input_tagged_words[i:i+3] for i in range(len(input_tagged_words)-2)]
    ref_trigrams = [ref_tagged_words[i:i+3] for i in range(len(ref_tagged_words)-2)]
    count_occ_trigrams = 0
    tmp_ref_trigrams = copy.deepcopy(ref_trigrams)
    for i in range(len(input_trigrams)):
        for j in range(len(ref_trigrams)):
            if (input_trigrams[i][0] == tmp_ref_trigrams[j][0]
                    and input_trigrams[i][1] == tmp_ref_trigrams[j][1]
                    and input_trigrams[i][2] == tmp_ref_trigrams[j][2]):

                count_occ_trigrams += 1
                tmp_ref_trigrams[j][0] = ""
                tmp_ref_trigrams[j][1] = ""
                tmp_ref_trigrams[j][2] = ""
                break

    input_max_trigrams = len(input_trigrams)
    if input_max_trigrams == 0:
        occurrence_trigrams= 0
    else:
        occurrence_trigrams = round(float(count_occ_trigrams / input_max_trigrams), 4)

    # bigrams
    input_bigrams = [input_tagged_words[i:i+2] for i in range(len(input_tagged_words)-1)]
    ref_bigrams   = [ref_tagged_words[i:i+2]   for i in range(len(ref_tagged_words)-1)]

    count_occ_bigrams = 0
    tmp_ref_bigrams = copy.deepcopy(ref_bigrams)
    for i in range(len(input_bigrams)):
        for j in range(len(ref_bigrams)):
            if (input_bigrams[i][0] == tmp_ref_bigrams[j][0]
                    and input_bigrams[i][1] == tmp_ref_bigrams[j][1]):
                count_occ_bigrams += 1
                tmp_ref_bigrams[j][0] = ""
                tmp_ref_bigrams[j][1] = ""
                break

    input_max_bigrams = len(input_bigrams)
    if input_max_bigrams == 0:
        occurrence_bigrams= 0
    else:
        occurrence_bigrams = round(float(count_occ_bigrams / input_max_bigrams), 4)

    # unigrams
    nouns_tags = ["NN", "NNS", "NP", "NPS", "NE"]
    input_unigrams = [elem[2].lower() \
                          for elem in input_tagged \
                          if input_tagged[input_tagged.index(elem)][1] in nouns_tags]
    ref_unigrams = [elem[2].lower() \
                        for elem in ref_tagged \
                        if ref_tagged[ref_tagged.index(elem)][1] in nouns_tags]

    count_occ_unigrams = 0
    tmp_ref_unigrams = copy.deepcopy(ref_unigrams)
    for i in range(len(input_unigrams)):
        for j in range(len(ref_unigrams)):
            if input_unigrams[i] == tmp_ref_unigrams[j]:
                count_occ_unigrams += 1
                tmp_ref_unigrams[j] = ""
                break

    input_max_unigrams = len(input_unigrams)
    if input_max_unigrams == 0:
        occurrence_unigrams= 0
    else:
        occurrence_unigrams = round(float(count_occ_unigrams / input_max_unigrams), 4)

    weighted_total_occurrence = round((occurrence_trigrams * 0.6)
                                         + (occurrence_bigrams * 0.3)
                                         + (occurrence_unigrams * 0.1), 4)
    variability = round(1 - weighted_total_occurrence, 4)
    return variability


##############################################################################
#
# All functions are defined.  Provide example how the read-in data and
# to compute various similarity measures.
#

if __name__ == "__main__":
    #
    # Create empty arrays, such we can easily comment out some
    # preprocess() lines if not needed.
    #
    assessments_exams = []
    mohler_assessments = []

    #
    # Read Mohler data
    #
    mohler_assessments = preprocess("resources/testdata/en/*.mohlercsv")
    assessments_exams.extend(mohler_assessments)

    assessments = assessments_exams + mohler_assessments

    nrSubmissions = 0
    [nrSubmissions := nrSubmissions + len(a["submissions"]) for a in assessments]

    print("\n" + str(len(assessments)) + " assessments + "
              + str(nrSubmissions) + " submissions")

    correlations = defaultdict(list)
    weights = defaultdict(list)
    rmses = defaultdict(list)
    percentages = defaultdict(list)

    for assessment in assessments:
        compute_persisted(assessment, "similarities", compute_completeness_similarities)
        compute_persisted(assessment, "variabilities", compute_variabilities)
        compute_persisted(assessment, "jarowinkler", compute_jarowinkler)

        for i in range(len(assessment["variabilities"])):
            assessment["variabilities"][i][i] = 9.9999

        print("=============================")

        print("similarities " + assessment["id"] + ": "
                  + str(len(assessment["submissions"])) + " submissions")
        print(assessment["similarities"], sep="\n")
        print("\npercentages " + assessment["id"] + ": ")
        print(assessment["percentages"], sep="\n")

        (corr, r) = print_correlations(pretty_id(assessment),
                                           assessment["similarities"],
                                           assessment["percentages"])

        # When all percentages are equal (e.g. Mohler 8.2), "corr"
        # will be "nan", which can't be used for weighted avg.
        #
        if (max(assessment["percentages"]) <= 1 and not np.isnan(corr)):
            correlations[assessment["lang"]].append(corr)
            weights[assessment["lang"]].append(len(assessment["submissions"]))
            rmses[assessment["lang"]].append(r)

        if max(assessment["percentages"]) <= 1:
            percentages[assessment["lang"]].extend(assessment["percentages"])

        print("\nvariabilites " + assessment["id"] + ": "
                  + str(len(assessment["submissions"])) + " submissions")

        row_format = "{:.4f} " * (len(assessment["variabilities"]))
        for v in assessment["variabilities"]:
            print(row_format.format(*v))
        print("")

        print(*assessment["jarowinkler"], sep="\n")
        print("")

        if verbose:
            print_submissions(assessment)

    correlations["ALL"] = correlations["de"] + correlations["en"]
    weights["ALL"] = weights["de"] + weights["en"]
    rmses["ALL"] = rmses["de"] + rmses["en"]
    for key in correlations.keys():
        if len(weights[key]) == 0:
            continue
        print(key + ": weighted avg of correlations", weighted_avg(correlations[key], weights[key]),
                  "RMSE:", weighted_avg(rmses[key], weights[key]),
                  " (n=" + str(sum(weights[key])) + ")")
