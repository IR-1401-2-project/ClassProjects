# In this script we want to use files in Trec format to evaluate the performance of the model
# This program takes three arguments:
# 1. Path to TrecQrel 
# 2. Path to TrecFormat
# 3. Eval Count K (optional)
# 4. Path to output file (optional)

import sys
import os
import re

def process_args():
    if len(sys.argv) < 3:
        print("Usage: python main.py <TrecQrel> <TrecFormat> [Eval Count K] [Output File]")
        sys.exit(1)
    else:
        TrecQrel = sys.argv[1]
        TrecFormat = sys.argv[2]
        if len(sys.argv) > 3:
            K = int(sys.argv[3])
        else:
            K = 10
        if len(sys.argv) > 4:
            OutFile = sys.argv[4]
        else:
            OutFile = None
    return TrecQrel, TrecFormat, K, OutFile

trecqrel, trecformat, k, outpath = process_args()
if outpath == None:
    outpath = "output.txt"

tq = ""
with open(trecqrel, "r") as f:
    tq = f.read()

tf = ""
with open(trecformat, "r") as f:
    tf = f.read()

def parse_trecqrel(trecqrel):
    lines = trecqrel.split("\n")
    res = []
    for line in lines:
        r = {}
        if line == "":
            continue
        # we split with space
        # first token is qid
        # second token is 0
        # third token is docid
        # fourth token is relevance
        tokens = re.split("\s+", line)
        r["qid"] = tokens[0]
        r["docid"] = tokens[2]
        r["relevance"] = tokens[3]
        res.append(r)
    return res

def parse_trecformat(trecformat):
    lines = trecformat.split("\n")
    res = []
    for line in lines:
        r = {}
        if line == "":
            continue
        # we split with space
        # first token is qid
        # second token is q0
        # third token is docno
        # fourth token is rank
        # fifth token is score
        # sixth token is tag
        tokens = re.split("\s+", line)
        r["qid"] = tokens[0]
        r["docno"] = tokens[2]
        r["rank"] = tokens[3]
        r["score"] = tokens[4]
        r["tag"] = tokens[5]
        res.append(r)
    return res

trecqrel = parse_trecqrel(tq)
trecformat = parse_trecformat(tf)

print("TrecQrel: \n", trecqrel)
print("TrecFormat: \n", trecformat)

# we want to measure the performance of the model using these:
# 1. MRR
# 2. MAP
# 3. P@K
# 4. R@K
# 5. NDCG@K

