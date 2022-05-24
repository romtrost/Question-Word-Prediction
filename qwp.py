from collections import defaultdict
from email.policy import default
from importlib.machinery import all_suffixes
import pandas as pd
import stanza
from os.path import exists

from happytransformer import HappyWordPrediction
from tqdm import tqdm

from sklearn.model_selection import train_test_split

PREPROCESSED_DATAFILE = "train-v2.0.json"
DATAFILE = "squad_process.json"

Q_TAGS = ["WHADJP", "WHAVP", "WHADVP", "WHNP"]


def process_data(fname_pre, fname_post):
    df = pd.read_json(fname_pre)
    pipe = stanza.Pipeline(
        lang="en",
        processors="tokenize,pos,constituency",
        package={"constituency": "wsj_bert"},
    )

    data = defaultdict(list)
    for topic in df.data:
        for p in topic["paragraphs"]:
            for q in p["qas"]:
                data["question"].append(q["question"].lower())
                answers = []
                for a in q["answers"]:
                    answers.append(a["text"])
                for a in q.get("plausible_answers", []):
                    answers.append(a["text"])
                data["answers"].append(answers)
        break

    freqs = (
        pd.DataFrame.from_dict(data)
        .question.str.split(expand=True)
        .stack()
        .value_counts()
    ).to_dict()
    tot = sum(freqs.values())
    freq_words = set(
        [w for w, _ in filter(lambda wf: wf[1] / tot > 0.001, freqs.items())]
    )

    for q in tqdm(data["question"]):
        doc = pipe(q)
        masked_q, qp_tree = mask_question(doc.sentences[0].constituency)
        if qp_tree:
            qp_as_dict = question_phrase_to_dict(qp_tree)
            qp, pruned = pruning_mask(qp_as_dict, freq_words)
            masked_q = reconstruct_question_from_pruning(pruned, masked_q)
            data["mask"].append(qp)
        else:
            data["mask"].append([])

    pd.DataFrame(data, columns=data.keys()).to_json(rf"{fname_post}")


def reconstruct_question_from_pruning(pruned, masked_q):
    mask_index = masked_q.index("<qw>")
    for w in reversed(pruned):
        masked_q.insert(mask_index + 1, w)
    return masked_q


def pruning_mask(question_phrase, freq_words):
    pruned, qw = [], []
    for word, tag in question_phrase.items():
        if not qw and "W" in tag or len(qw) == 1 and "J" in tag and word in freq_words:
            qw.append(word)
        elif qw:
            pruned.append(word)
    return qw, pruned


def mask_question(tree, found=False):
    masked, mask = [], None
    if not found and tree.label in Q_TAGS:
        mask = tree
        masked = ["<qw>"]
    elif not tree.children:
        masked = [tree.label]
    else:
        for ch in tree.children:
            ch_masked, ch_mask = mask_question(ch, found)
            masked.extend(ch_masked)
            if ch_mask and not found:
                found = True
                mask = ch_mask

    return masked, mask


def question_phrase_to_dict(tree, parent_label="QP"):
    res = {}
    if not tree.children:
        res[tree.label] = parent_label
    for ch in tree.children:
        for w, tag in question_phrase_to_dict(ch, tree.label).items():
            res[w] = tag

    return res


def construct_question_phrase_dict(masks):
    question_phrase_dict = {}
    failed = []
    for i, question_phrase in enumerate(masks):
        if question_phrase:
            qw = question_phrase[0].lower()
            if qw not in question_phrase_dict:
                question_phrase_dict[qw] = set()
            if len(question_phrase) > 1:
                question_phrase_dict[qw].add(question_phrase[1])
        else:
            failed.append(i)
    return failed, question_phrase_dict


def main():
    stanza.download("en")
    if not exists(DATAFILE):
        process_data(PREPROCESSED_DATAFILE, DATAFILE)
    data = pd.read_json(DATAFILE)
    failed, question_phrase_dict = construct_question_phrase_dict(data.loc[:, "mask"])
    print("Don't try to predict questions at index:", failed)
    print(question_phrase_dict)
    print(data)
    # # train, test = train_test_split(data, test_size=0.1)

    # happy_wp = HappyWordPrediction("ROBERTA", "roberta-large")
    # correct = 0
    # n_sample = 100
    # for i, row in tqdm(data.iterrows()):
    #     if i == n_sample:
    #         break
    #     q = row["questions"]
    #     assume_how_x = q.copy()
    #     assume_how_x.insert(row["indices"], "how")
    #     q = " ".join(q) + " " + " ".join(row["answers"])
    #     assume_how_x = " ".join(assume_how_x) + " " + " ".join(row["answers"])
    #     pred1 = happy_wp.predict_mask(q, targets=QUESTION_WORDS)[0]
    #     pred2 = happy_wp.predict_mask(assume_how_x, targets=AUX_QUESTION_WORDS["how"])[
    #         0
    #     ]
    #     pred = pred1.token if pred1.score >= pred2.score else "how " + pred2.token
    #     if pred == row["masks"]:
    #         correct += 1

    # print(f"{correct/n_sample}% accuracy")


if __name__ == "__main__":
    main()
