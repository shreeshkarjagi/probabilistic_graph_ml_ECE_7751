import sys
import os
import math
from collections import defaultdict

DATA_DIR = r"/Users/skarjagi6/Library/CloudStorage/Dropbox-GaTech/Shreesh Karjagi/coursework/GRAPH_ML/HWK1/Data/Prob-8-hmm-data"
TRAIN_FILE = os.path.join(DATA_DIR, "gene.train")
TEST_FILE = os.path.join(DATA_DIR, "gene.test")
KEY_FILE = os.path.join(DATA_DIR, "gene.key")
RARE_TRAIN_FILE = os.path.join(DATA_DIR, "gene_rare.train")
COUNTS_FILE = os.path.join(DATA_DIR, "gene.counts")
P1_OUT = os.path.join(DATA_DIR, "gene_test.p1.out")
P2_OUT = os.path.join(DATA_DIR, "gene_test.p2.out")

RARE_THRESHOLD = 5

def get_word_counts(train_file):
    word_counts = defaultdict(int)
    with open(train_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(" ")
                word = " ".join(parts[:-1])
                word_counts[word] += 1
    return word_counts


def replace_rare_words(train_file, output_file, threshold=5):
    word_counts = get_word_counts(train_file)
    
    with open(train_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            line_stripped = line.strip()
            if line_stripped:
                parts = line_stripped.split(" ")
                word = " ".join(parts[:-1])
                tag = parts[-1]
                if word_counts[word] < threshold:
                    fout.write(f"_RARE_ {tag}\n")
                else:
                    fout.write(line)
            else:
                fout.write("\n")

def collect_counts(train_file):
    emission_counts = defaultdict(int)  #(word, tag) -> count
    unigram_counts = defaultdict(int)   #(tag,) -> count
    bigram_counts = defaultdict(int)    #(tag1, tag2) -> count
    trigram_counts = defaultdict(int)   #(tag1, tag2, tag3) -> count
    all_tags = set()
    sentences = []
    current = []
    with open(train_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(" ")
                word = " ".join(parts[:-1])
                tag = parts[-1]
                current.append((word, tag))
                all_tags.add(tag)
            else:
                if current:
                    sentences.append(current)
                    current = []
    if current:
        sentences.append(current)
    
    for sent in sentences:
        tags = [tag for word, tag in sent]
        
        for word, tag in sent:
            emission_counts[(word, tag)] += 1
        
        #N-gram counts with boundary symbols
        padded = ["*", "*"] + tags + ["STOP"]
        
        for i in range(len(tags)):
            unigram_counts[(tags[i],)] += 1
        for i in range(len(padded) - 1):
            bigram_counts[(padded[i], padded[i+1])] += 1
        for i in range(len(padded) - 2):
            trigram_counts[(padded[i], padded[i+1], padded[i+2])] += 1
    return emission_counts, unigram_counts, bigram_counts, trigram_counts, all_tags


def compute_emission_probs(emission_counts, unigram_counts):
    #e(x|y) = Count(y -> x) / Count(y)
    emission_probs = {}
    for (word, tag), count in emission_counts.items():
        emission_probs[(word, tag)] = count / unigram_counts[(tag,)]
    return emission_probs

def compute_transition_probs(bigram_counts, trigram_counts):
    #q(yi | yi-2, yi-1) = Count(yi-2, yi-1, yi) / Count(yi-2, yi-1)
    transition_probs = {}
    for (t1, t2, t3), count in trigram_counts.items():
        transition_probs[(t1, t2, t3)] = count / bigram_counts[(t1, t2)]
    return transition_probs

def baseline_tagger(test_file, output_file, emission_probs, all_tags, word_set):
    with open(test_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            word = line.strip()
            if word:
                lookup = word if word in word_set else "_RARE_"
                
                best_tag = None
                best_prob = -1.0
                for tag in all_tags:
                    prob = emission_probs.get((lookup, tag), 0.0)
                    if prob > best_prob:
                        best_prob = prob
                        best_tag = tag
                
                fout.write(f"{word} {best_tag}\n")
            else:
                fout.write("\n")

def viterbi(sentence, emission_probs, transition_probs, all_tags, word_set):
    n = len(sentence)
    tags = list(all_tags)
    
    #pi[k][(u, v)] for k = 1..n
    pi = [defaultdict(float) for _ in range(n + 1)]
    bp = [defaultdict(str) for _ in range(n + 1)]
    
    #base case: pi[0][("*", "*")] = 1.0
    pi[0][("*", "*")] = 1.0
    
    def get_emission(word, tag):
        lookup = word if word in word_set else "_RARE_"
        return emission_probs.get((lookup, tag), 0.0)
    
    def get_transition(t1, t2, t3):
        return transition_probs.get((t1, t2, t3), 0.0)
    
    def possible_tags(k):
        if k <= 0:
            return ["*"]
        return tags
    
    for k in range(1, n + 1):
        word = sentence[k - 1]
        e = {}
        for v in possible_tags(k):
            e[v] = get_emission(word, v)
        
        for v in possible_tags(k):
            if e[v] == 0:
                continue
            for u in possible_tags(k - 1):
                for w in possible_tags(k - 2):
                    if pi[k - 1].get((w, u), 0.0) == 0:
                        continue
                    q = get_transition(w, u, v)
                    if q == 0:
                        continue
                    score = pi[k - 1][(w, u)] * q * e[v]
                    if score > pi[k].get((u, v), 0.0):
                        pi[k][(u, v)] = score
                        bp[k][(u, v)] = w
    
    best_score = 0.0
    best_u, best_v = None, None
    for u in possible_tags(n - 1):
        for v in possible_tags(n):
            if pi[n].get((u, v), 0.0) == 0:
                continue
            q_stop = get_transition(u, v, "STOP")
            score = pi[n][(u, v)] * q_stop
            if score > best_score:
                best_score = score
                best_u, best_v = u, v
    
    if best_u is None:
        return ["O"] * n
    
    tag_seq = [""] * n
    tag_seq[n - 1] = best_v
    if n >= 2:
        tag_seq[n - 2] = best_u
    
    for k in range(n - 2, 0, -1):
        tag_seq[k - 1] = bp[k + 2][(tag_seq[k], tag_seq[k + 1])]
    
    return tag_seq


def viterbi_tagger(test_file, output_file, emission_probs, transition_probs, all_tags, word_set):
    sentences = []
    current = []
    
    with open(test_file, 'r') as f:
        for line in f:
            word = line.strip()
            if word:
                current.append(word)
            else:
                if current:
                    sentences.append(current)
                    current = []
    if current:
        sentences.append(current)
    
    with open(output_file, 'w') as fout:
        for i, sent in enumerate(sentences):
            tags = viterbi(sent, emission_probs, transition_probs, all_tags, word_set)
            for word, tag in zip(sent, tags):
                fout.write(f"{word} {tag}\n")
            fout.write("\n")
            
            if (i + 1) % 100 == 0:
                print(f"processed {i+1}/{len(sentences)} sentences")
    

def evaluate(key_file, pred_file):
    def read_entities(filepath):
        entities = []
        current_start = None
        idx = 0
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(" ")
                    tag = parts[-1]
                    
                    if tag.startswith("I-GENE"):
                        if current_start is None:
                            current_start = idx
                    else:
                        if current_start is not None:
                            entities.append((current_start, idx - 1))
                            current_start = None
                    idx += 1
                else:
                    if current_start is not None:
                        entities.append((current_start, idx - 1))
                        current_start = None
        
        if current_start is not None:
            entities.append((current_start, idx - 1))
        
        return set(entities)
    
    gold = read_entities(key_file)
    pred = read_entities(pred_file)
    
    correct = len(gold & pred)
    
    precision = correct / len(pred) if pred else 0
    recall = correct / len(gold) if gold else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"  Found {len(pred)} GENEs. Expected {len(gold)} GENEs; Correct: {correct}.")
    print(f"  Precision: {precision:.6f}")
    print(f"  Recall:    {recall:.6f}")
    print(f"  F1-Score:  {f1:.6f}")
    
    return precision, recall, f1


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    
    replace_rare_words(TRAIN_FILE, RARE_TRAIN_FILE, RARE_THRESHOLD)
    
    emission_counts, unigram_counts, bigram_counts, trigram_counts, all_tags = \
        collect_counts(RARE_TRAIN_FILE)
    
    print(f"tags found: {sorted(all_tags)}")
    print(f"unigram counts: {dict(unigram_counts)}")
    print(f"emission types: {len(emission_counts)}")
    print(f"bigrams: {len(bigram_counts)}")
    print(f"trigrams: {len(trigram_counts)}")
    
    word_set = set()
    for (word, tag) in emission_counts:
        word_set.add(word)
    
    emission_probs = compute_emission_probs(emission_counts, unigram_counts)
    
    baseline_tagger(TEST_FILE, P1_OUT, emission_probs, all_tags, word_set)
    
    print("\nBaseline Results:")
    evaluate(KEY_FILE, P1_OUT)
    
    transition_probs = compute_transition_probs(bigram_counts, trigram_counts)
    viterbi_tagger(TEST_FILE, P2_OUT, emission_probs, transition_probs, all_tags, word_set)
    evaluate(KEY_FILE, P2_OUT)
