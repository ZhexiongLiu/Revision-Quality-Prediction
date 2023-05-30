import pandas as pd
import os
import time
import re
import torch.nn as nn
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from collections import Counter
import re
import faiss
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import openai
from data_augmentation import find_augmented_sentences_context
STOP_WORDS = set(stopwords.words('english'))
vectorizer = SentenceTransformer('all-mpnet-base-v2')


def encode_desirable_label(label):
    if label == "desirable":
        label = 1
    elif label == "undesirable":
        label = 0
    else:
        raise "encode desirable label error"
    return label

def my_collate(batch):
    len_batch = len(batch) # original batch length
    try:
        batch = list(filter(lambda x:x[4]>=0, batch)) # filter out all the Nones for holistic classification
    except:
        pass
    if len_batch > len(batch): # source all the required samples from the original dataset at random
        diff = len_batch - len(batch)
        for _ in range(diff):
            batch.append(dataset[np.random.randint(0, len(dataset))])

    return torch.utils.data.dataloader.default_collate(batch)


def my_collator_with_param(batch, args, dataset):
    if args.batch_size > len(batch):  # source all the required samples from the original dataset at random
        diff = int(args.batch_size) - len(batch)
        for _ in range(diff):
            batch.append(dataset[np.random.randint(0, len(dataset))])

    return torch.utils.data.dataloader.default_collate(batch)


def fill_blank_sentences(df, holistic, essay_id, args):

    def fill_blank_with_constant(sentences):
        sentences = sentences.tolist()
        for i in range(len(sentences)):
            if isinstance(sentences[i], float):
                sentences[i] = ""
        return sentences

    def fill_blank_with_previous(sentences):
        sentences = sentences.tolist()
        for i in range(len(sentences)):
            if isinstance(sentences[i], float):
                prev = i-1
                while prev >= 0 and isinstance(sentences[prev], float): prev -= 1
                if prev < 0:
                    prev_sentence = ""
                else:
                    prev_sentence = sentences[prev]
                sentences[i] = prev_sentence
        return sentences

    def fill_blank_with_neighbors(sentences):
        sentences = sentences.tolist()
        for i in range(len(sentences)):
            if isinstance(sentences[i], float):
                prev = i-1
                while prev >= 0 and isinstance(sentences[prev], float): prev -= 1
                post = i+1
                while post < len(sentences) and isinstance(sentences[post], float): post += 1
                if prev < 0:
                    prev_sentence = ""
                else:
                    prev_sentence = sentences[prev]
                if post >= len(sentences):
                    post_sentence = ""
                else:
                    post_sentence = sentences[post]
                sentences[i] = prev_sentence + " " + post_sentence
        return sentences

    def fill_blank_with_context(sentences, context):
        sentences = sentences.tolist()
        context = context.tolist()
        for i in range(len(sentences)):
            if isinstance(sentences[i], float):
                sentences[i] = str(context[0])
        return sentences

    def fill_blank_with_relevant(old_sentences, new_sentences):
        old_sentences = old_sentences.tolist()
        new_sentences = new_sentences.tolist()
        context_sentences = [x for x in old_sentences if not isinstance(x, float)]
        if len(context_sentences) == 0:
            context_sentences = [""]
        fast_index = build_search_engine(context_sentences, vectorizer)
        for i in range(len(old_sentences)):
            if isinstance(old_sentences[i], float):
                query_text = new_sentences[i]
                if isinstance(query_text, float):
                    old_sentences[i] = context_sentences[0]
                else:
                    this_index = do_faiss_lookup(fast_index, query_text, 1, vectorizer)[0]
                    old_sentences[i] = context_sentences[this_index]
        return old_sentences

    old_sentences = df["old_sentences"]
    new_sentences = df["new_sentences"]

    if args.context_type in ["summary", "reasoning", "evidence", "claim", "reasoning_summary", "evidence_summary", "claim_summary"]:
        old_context = holistic[holistic["essay_id"]==essay_id][f"first_{args.context_type}"]
        new_context = holistic[holistic["essay_id"]==essay_id][f"second_{args.context_type}"]
        df["old_sentences"] = fill_blank_with_context(old_sentences, old_context)
        df["new_sentences"] = fill_blank_with_context(new_sentences, new_context)
    elif args.context_type == "constant":
        df["old_sentences"] = fill_blank_with_constant(old_sentences)
        df["new_sentences"] = fill_blank_with_constant(new_sentences)
    elif args.context_type == "previous":
        df["old_sentences"] = fill_blank_with_previous(old_sentences)
        df["new_sentences"] = fill_blank_with_previous(new_sentences)
    elif args.context_type == "neighbor":
        df["old_sentences"] = fill_blank_with_neighbors(old_sentences)
        df["new_sentences"] = fill_blank_with_neighbors(new_sentences)
    elif args.context_type == "relevant":
        df["old_sentences"] = fill_blank_with_relevant(old_sentences, new_sentences)
        df["new_sentences"] = fill_blank_with_relevant(new_sentences, old_sentences)
    else:
        raise "no implement!"
    return df


def get_argumentative_context(df, holistic, essay_id, args):

    def constant_context(old_sentences, _):
        context = ["" for _ in range(len(old_sentences))]
        return context

    def previous_context(old_sentences, new_sentences):
        length = len(old_sentences)
        context = [""] * length
        for i in range(length):
            if isinstance(new_sentences[i], float):
                candidate = old_sentences
            else:
                candidate = new_sentences

            if i == 0: continue
            index = i - 1
            if isinstance(candidate[index], float):
                context[i] = ""
            else:
                context[i] = candidate[index]
        return context

    def after_context(old_sentences, new_sentences):
        length = len(old_sentences)
        context = [""] * length
        for i in range(length):
            if isinstance(new_sentences[i], float):
                candidate = old_sentences
            else:
                candidate = new_sentences

            if i == length - 1: continue
            index = i + 1
            if isinstance(candidate[index], float):
                context[i] = ""
            else:
                context[i] = candidate[index]
        return context


    def neighbor_context(old_sentences, new_sentences, window_size=1):
        length = len(old_sentences)
        context = [""] * length
        for i in range(length):
            if isinstance(new_sentences[i], float):
                candidate = old_sentences
            else:
                candidate = new_sentences

            if window_size == "boundary":
                prev_idx = i - 1
                # while prev_idx >= 0 and (old_sentences[prev_idx] != new_sentences[prev_idx]): prev_idx -= 1
                # while prev_idx >= 0 and (isinstance(old_sentences[prev_idx], float) or isinstance(new_sentences[prev_idx], float)): prev_idx -= 1
                while prev_idx >= 0 and isinstance(candidate[prev_idx], float): prev_idx -= 1
                after_idx = i + 1
                # while after_idx < length and (old_sentences[after_idx] != new_sentences[after_idx]): after_idx += 1
                # while after_idx < length and (isinstance(old_sentences[after_idx], float) or isinstance(new_sentences[after_idx], float)): after_idx += 1
                while after_idx < length and isinstance(candidate[after_idx], float): after_idx += 1
            else:
                prev_idx = max(i-window_size, 0)
                after_idx = min(i+window_size, length-1)

            context_list = candidate[prev_idx:i] + candidate[i+1:after_idx+1]
            # context_list = candidate[prev_idx:after_idx+1]
            context_list = [x for x in context_list if not isinstance(x, float)]
            context[i] = " ".join(context_list)
        return context

    def argument_context(old_sentences, new_sentences, old_context, new_context):
        length = len(old_sentences)
        context = [""] * length
        for i in range(length):
            if isinstance(new_sentences[i], float):
                candidate = old_context
            else:
                candidate = new_context
            context[i] = candidate[0]
        return context

    def relevant_context(old_sentences, new_sentences):
        length = len(old_sentences)
        context = [""] * length

        old_context = [str(x).replace("nan", "") for x in old_sentences]
        if len(old_context) == 0: old_context = [""]

        new_context = [str(x).replace("nan", "") for x in new_sentences]
        if len(new_context) == 0: new_context = [""]

        for i in range(length):
            if isinstance(new_sentences[i], float):
                candidate = old_sentences
                fast_index = build_search_engine(old_context, vectorizer)
                query_text = old_sentences[i]
            else:
                candidate = new_sentences
                fast_index = build_search_engine(new_context, vectorizer)
                query_text = new_sentences[i]

            this_index = do_faiss_lookup(fast_index, query_text, 2, vectorizer)[1]
            context[i] = candidate[this_index]
        return context

    old_sentences = df["old_sentences"].tolist()
    new_sentences = df["new_sentences"].tolist()

    if args.context_type in ["summary", "reasoning", "evidence", "claim", "reasoning_summary", "evidence_summary", "claim_summary",
                             "reasoning_single", "evidence_single", "claim_single"]:
        old_context = holistic[holistic["essay_id"]==essay_id][f"first_{args.context_type}"].tolist()
        new_context = holistic[holistic["essay_id"]==essay_id][f"second_{args.context_type}"].tolist()

        # old_summary = holistic[holistic["essay_id"]==essay_id][f"first_{args.context_type[:-8]}_single"].tolist()
        # new_summary = holistic[holistic["essay_id"]==essay_id][f"second_{args.context_type[:-8]}_single"].tolist()
        #
        # if args.context_type in ["reasoning_summary", "evidence_summary", "claim_summary"]:
        #     old_context = [old_context[0] + old_summary[0]]
        #     new_context = [new_context[0] + new_summary[0]]

        df["context"] = argument_context(old_sentences, new_sentences, old_context, new_context)
    elif args.context_type == "constant":
        df["context"] = constant_context(old_sentences, new_sentences)
    elif args.context_type == "previous":
        df["context"] = previous_context(old_sentences, new_sentences)
    elif args.context_type == "after":
        df["context"] = after_context(old_sentences, new_sentences)
    elif args.context_type == "neighbor-short-1":
        df["context"] = neighbor_context(old_sentences, new_sentences, window_size=1)
    elif args.context_type == "neighbor-short-2":
        df["context"] = neighbor_context(old_sentences, new_sentences, window_size=2)
    elif args.context_type == "neighbor-short-3":
        df["context"] = neighbor_context(old_sentences, new_sentences, window_size=3)
    elif args.context_type == "neighbor-long":
        df["context"] = neighbor_context(old_sentences, new_sentences, window_size="boundary")
    elif args.context_type == "relevant":
        df["context"] = relevant_context(old_sentences, new_sentences)
    else:
        raise "no implement!"
    return df



def build_search_engine(text_list, vectorizer):
    if vectorizer is None: return None
    embeddings = vectorizer.encode(text_list)
    n_dimensions = embeddings.shape[1]
    fastIndex = faiss.IndexFlatL2(n_dimensions)
    fastIndex.add(embeddings.astype('float32'))

    return fastIndex


def do_faiss_lookup(fastIndex, query_text, top_k, vectorizer):
    embedding = vectorizer.encode(query_text).reshape(1,-1)
    _, matched_indexes = fastIndex.search(embedding, top_k)
    return matched_indexes[0]


def get_master_data_revision_purpose(args):
    if args.data_source == "mvp":
        file_paths = get_file_paths("./data_clean/revision_purpose/mvp")
        holistic = pd.read_excel(f"./data_clean/holistic_scoring/mvp_seed{args.random_seed}.xlsx")
    elif args.data_source == "space":
        file_paths = get_file_paths("./data_clean/revision_purpose/space")
        holistic = pd.read_excel(f"./data_clean/holistic_scoring/space_seed{args.random_seed}.xlsx")
    elif args.data_source == "mixture":
        file_paths = get_file_paths("./data_clean/revision_purpose/mvp") + get_file_paths("./data_clean/revision_purpose/space")
        holistic = pd.concat([pd.read_excel(f"./data_clean/holistic_scoring/mvp_seed{args.random_seed}.xlsx"), pd.read_excel(f"./data_clean/holistic_scoring/space_seed{args.random_seed}.xlsx")])
    elif args.data_source == "college":
        file_paths = get_file_paths("./data_clean/revision_purpose/college")
        holistic = pd.read_excel(f"./data_clean/holistic_scoring/college_seed{args.random_seed}.xlsx")
    elif args.data_source == "elementary":
        file_paths = get_file_paths("./data_clean/revision_purpose/mvp/")
        file_paths = [x for x in file_paths if x[-6] == "1"]
        holistic = pd.read_excel("./data_clean/holistic_scoring/mvp.xlsx")
    else:
        raise "wrong path!"

    # if args.data_source == "college":
    #     df_list = []
    #     for path in tqdm(file_paths):
    #         if "college_essays" in path: continue
    #         tmp_df = pd.read_excel(path)
    #         tmp_df = tmp_df.rename(columns={"ID": "essay_ids", "S1": "old_sentences", "S2": "new_sentences", "Label2": "desirable_labels", "Label1": "purpose_labels"})
    #
    #         tmp_df["old_sentences"] = tmp_df["old_sentences"].fillna("")
    #         tmp_df["new_sentences"] = tmp_df["new_sentences"].fillna("")
    #
    #         if "Evidence" in path:
    #             tmp_df["fine_labels"] = "evidence"
    #             tmp_df["essay_ids"] = tmp_df["essay_ids"] + "1111"
    #         else:
    #             tmp_df["fine_labels"] = "reasoning"
    #             tmp_df["essay_ids"] = tmp_df["essay_ids"] + "22222"
    #
    #         tmp_df["essay_ids"] = tmp_df["essay_ids"].apply(lambda x: x.replace("native", "88").replace("esl", "99"))
    #         tmp_df["desirable_labels"] = tmp_df["desirable_labels"].apply(lambda x: x.lower())
    #
    #         tmp_add_df = tmp_df[tmp_df["Add"] == 1]
    #         tmp_delete_df = tmp_df[tmp_df["Delete"] == 1]
    #         tmp_modify_df = tmp_df[tmp_df["Modify"] == 1]
    #
    #         if args.context_type == "previous":
    #             tmp_add_df["old_sentences"] = tmp_add_df["old_sentences"] + tmp_add_df["ContextD1"]
    #             tmp_delete_df["new_sentences"] = tmp_delete_df["new_sentences"] + tmp_delete_df["ContextD1"]
    #             df_list.append(tmp_add_df)
    #             df_list.append(tmp_delete_df)
    #             df_list.append(tmp_modify_df)
    #         elif args.context_type == "neighbor":
    #             tmp_add_df["old_sentences"] = tmp_add_df["old_sentences"] + tmp_add_df["ContextD2"]
    #             tmp_delete_df["new_sentences"] = tmp_delete_df["new_sentences"] + tmp_delete_df["ContextD2"]
    #             df_list.append(tmp_add_df)
    #             df_list.append(tmp_delete_df)
    #             df_list.append(tmp_modify_df)
    #         elif args.context_type in ["constant", "relevant"]:
    #             df_list.append(tmp_df)
    #         else:
    #             group_df = tmp_df.groupby("essay_ids")
    #             for group in group_df:
    #                 essay_id = group[0]
    #                 g_df = group[1]
    #                 # g_df = fill_blank_sentences(g_df, holistic, essay_id, args)
    #                 g_df = get_argumentative_context(g_df, holistic, essay_id, args)
    #                 df_list.append(g_df)
    #     master_df = pd.concat(df_list)
    #
    # else:
    df_list = []
    for path in tqdm(file_paths):
        essay_id = int(path.split("/")[-1].split(".")[0])
        df = pd.read_excel(path)
        df["essay_ids"] = essay_id
        # df = fill_blank_sentences(df, holistic, essay_id, args)
        df = get_argumentative_context(df, holistic, essay_id, args)

        if args.exp_type == "desirable":
            selected_df = df[~df["desirable_labels"].isna()]
        elif args.exp_type == "purpose":
            selected_df = df[~df["purpose_labels"].isna()]
        else:
            raise NotImplementedError
        df_list.append(selected_df)
    master_df = pd.concat(df_list)

    return master_df


def get_augment_data(df):
    df_list = []
    index = df.index.tolist()
    for i in range(len(index)):
        tmp_df = df[df.index==i]
        old_sentences = tmp_df["old_sentences"].tolist()[0]
        new_sentences = tmp_df["new_sentences"].tolist()[0]
        if len(old_sentences) < len(new_sentences):
            augmented_sentences = find_augmented_sentences_context(old_sentences, new_sentences)
            aug_old_sentences = [x[0] for x in augmented_sentences]
            aug_new_sentences = [x[1] for x in augmented_sentences]
        else:
            augmented_sentences = find_augmented_sentences_context(new_sentences, old_sentences)
            aug_old_sentences = [x[1] for x in augmented_sentences]
            aug_new_sentences = [x[0] for x in augmented_sentences]

        if len(augmented_sentences) == 0:
            aug_tmp_df = tmp_df
        else:
            aug_tmp_df = pd.concat([tmp_df]*len(augmented_sentences))
            aug_tmp_df["essay_ids_postfix"] = list(range(len(augmented_sentences)))
            aug_tmp_df["essay_ids_postfix"] = aug_tmp_df["essay_ids_postfix"].apply(lambda x: "{:05d}".format(x))
            aug_tmp_df["old_sentences"] = aug_old_sentences
            aug_tmp_df["new_sentences"] = aug_new_sentences
            aug_tmp_df["essay_ids"] = aug_tmp_df["essay_ids"] + aug_tmp_df["essay_ids_postfix"]
        df_list.append(aug_tmp_df)
    aug_df = pd.concat(df_list)
    aug_df = aug_df.reset_index(drop=True)
    return aug_df


def get_master_data_essay_scoring(args):
    space_raw_data_path = os.path.join(args.raw_data_dir, "Random Compiled 300 Space with Maya Scores.csv")
    mvp_raw_data_path = os.path.join(args.raw_data_dir, "Random Compiled 155 MVP with Maya Scores.csv")
    feedback_text_path = os.path.join(args.raw_data_dir, "feedback_levels.csv")
    space_processed_data_dir = os.path.join(args.processed_data_dir, "SPACE")
    mvp_processed_data_dir = os.path.join(args.processed_data_dir, "MVP")

    feedback_text = pd.read_csv(feedback_text_path)
    space_data = pd.read_csv(space_raw_data_path)[["essay_id", "first_draft", "second_draft", "revision_score", "feedback_level"]]
    mvp_data = pd.read_csv(mvp_raw_data_path)[["essay_id", "first_draft", "second_draft", "revision_score", "feedback_level"]]

    # space processed data
    first_draft_processed = []
    second_draft_processed = []
    for i in tqdm(range(len(space_data))):
        id = space_data["essay_id"].tolist()[i]
        file_path = os.path.join(space_processed_data_dir, f"{id}.xlsx")
        single_data = pd.read_excel(file_path)
        select_data = single_data[single_data["coarse_labels"].isnull() != True]
        if len(select_data) == 0:
            first_draft_processed.append("null")
            second_draft_processed.append("null")
        else:
            first_draft = select_data["old_sentences"]
            second_draft = select_data["new_sentences"]

            first_draft_str = " ".join([tmp for tmp in first_draft if type(tmp) == str])
            second_draft_str = " ".join([tmp for tmp in second_draft if type(tmp) == str])
            if len(first_draft_str) == 0:
                first_draft_str = "null"
            if len(second_draft_str) == 0:
                second_draft_str = "null"
            first_draft_processed.append(first_draft_str)
            second_draft_processed.append(second_draft_str)
    space_data["first_draft_processed"] = first_draft_processed
    space_data["second_draft_processed"] = second_draft_processed

    # mvp processed data
    first_draft_processed = []
    second_draft_processed = []
    for i in tqdm(range(len(mvp_data))):
        id = mvp_data["essay_id"].tolist()[i]
        file_path = os.path.join(mvp_processed_data_dir, f"{id}.xlsx")
        single_data = pd.read_excel(file_path)
        select_data = single_data[single_data["coarse_labels"].isnull() != True]
        if len(select_data) == 0:
            first_draft_processed.append("null")
            second_draft_processed.append("null")
        else:
            first_draft = select_data["old_sentences"]
            second_draft = select_data["new_sentences"]

            first_draft_str = " ".join([tmp for tmp in first_draft if type(tmp) == str])
            second_draft_str = " ".join([tmp for tmp in second_draft if type(tmp) == str])
            if len(first_draft_str) == 0:
                first_draft_str = "null"
            if len(second_draft_str) == 0:
                second_draft_str = "null"
            first_draft_processed.append(first_draft_str)
            second_draft_processed.append(second_draft_str)
    mvp_data["first_draft_processed"] = first_draft_processed
    mvp_data["second_draft_processed"] = second_draft_processed

    if args.data_source == "mixture":
        data = pd.concat([space_data, mvp_data], ignore_index=True)
    elif args.data_source == "mvp":
        data = mvp_data
    elif args.data_source == "space":
        data = space_data
    else:
        raise "wrong data source"

    data = pd.merge(data, feedback_text,  how='left', on=["feedback_level", "revision_score"])
    # data = data.sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
    data = data.sample(frac=1, random_state=22).reset_index(drop=True)

    return data


def get_master_data_with_mvp143(args):
    space_raw_data_path = os.path.join(args.raw_data_dir, "Random Compiled 300 Space with Maya Scores.csv")
    mvp155_raw_data_path = os.path.join(args.raw_data_dir, "Random Compiled 155 MVP with Maya Scores.csv")
    mvp143_raw_data_path = os.path.join(args.raw_data_dir, "N143 MVP Essays_Original_and_Draft.csv")
    feedback_text_path = os.path.join(args.raw_data_dir, "feedback_levels.csv")
    # space_processed_data_dir = os.path.join(args.processed_data_dir, "SPACE")
    space_processed_data_dir = os.path.join(args.raw_data_dir, "Space300_Maya")
    mvp143_processed_data_dir = os.path.join(args.processed_data_dir, "MVP143")
    mvp153_processed_data_dir = os.path.join(args.raw_data_dir, "MVP153_Maya")

    feedback_text = pd.read_csv(feedback_text_path)
    space_data = pd.read_csv(space_raw_data_path)[["essay_id", "first_draft", "second_draft", "revision_score", "feedback_level"]]
    mvp153_data = pd.read_csv(mvp155_raw_data_path)[["essay_id", "first_draft", "second_draft", "revision_score", "feedback_level"]]
    mvp143_data = pd.read_csv(mvp143_raw_data_path)[["essay_id", "first_draft", "second_draft", "revision_score", "feedback_level"]]

    mvp143_data = mvp143_data[~mvp143_data["essay_id"].isin([23, 127])] # skip cruppted data


    # space processed data
    first_draft_processed = []
    second_draft_processed = []
    for i in tqdm(range(len(space_data))):
        id = space_data["essay_id"].tolist()[i]
        file_path = os.path.join(space_processed_data_dir, f"{id}.xlsx")
        single_data = pd.read_excel(file_path)
        select_data = single_data[single_data["coarse_labels"].isnull() != True]
        if len(select_data) == 0:
            first_draft_processed.append("null")
            second_draft_processed.append("null")
        else:

            if args.use_purpose_label:

                first_draft = select_data["old_sentences"].apply(lambda x: "" if type(x) == float and np.isnan(x) else x).tolist()
                second_draft = select_data["new_sentences"].apply(lambda x: "" if type(x) == float and np.isnan(x) else x).tolist()

                coarse_labels = select_data["coarse_labels"].apply(lambda x: "" if type(x) == float and np.isnan(x) else x).tolist()
                if "Unnamed: 7" in select_data.columns:
                    evidence_reason = select_data["Unnamed: 7"].apply(lambda x: "" if type(x) == float and np.isnan(x) else x).tolist()
                else:
                    evidence_reason = [""]*len(select_data)

                if "Unnamed: 8" in select_data.columns:
                    purpose_labels = select_data["Unnamed: 8"].apply(lambda x: "" if type(x) == float and np.isnan(x) else x).tolist()
                else:
                    purpose_labels = [""]*len(select_data)

                first_tmp_list = []
                second_tmp_list = []
                for j in range(len(first_draft)):
                    if len(first_draft) > 0:
                        if coarse_labels[j].strip() == "surface":
                            template = f"[{coarse_labels[j].strip()}].".replace("  ","").replace(" ]", "]")
                        else:
                            template = f"[{coarse_labels[j].strip()} {evidence_reason[j].strip()} {purpose_labels[j].strip()}] {first_draft[j].strip()}".replace("  ", "").replace(" ]", "]")
                        first_tmp_list.append(template)
                    else:
                        first_tmp_list.append("")

                    if len(second_draft) > 0:
                        if coarse_labels[j].strip() == "surface":
                            template = f"[{coarse_labels[j].strip()}].".replace("  ","").replace(" ]", "]")
                        else:
                            template = f"[{coarse_labels[j].strip()} {evidence_reason[j].strip()} {purpose_labels[j].strip()}] {second_draft[j].strip()}".replace("  ","").replace(" ]", "]")
                        second_tmp_list.append(template)
                    else:
                        second_tmp_list.append("")

                first_draft_str = " ".join([x for x in first_tmp_list if not (x[0] == "[" and x[-2:]=="] ")])
                second_draft_str = " ".join([x for x in second_tmp_list if not (x[0] == "[" and x[-2:]=="] ")])

            else:
                first_draft = select_data["old_sentences"].apply(lambda x: "" if type(x) == float and np.isnan(x) else x).tolist()
                second_draft = select_data["new_sentences"].apply(lambda x: "" if type(x) == float and np.isnan(x) else x).tolist()

                first_draft_str = " ".join([tmp for tmp in first_draft if type(tmp) == str])
                second_draft_str = " ".join([tmp for tmp in second_draft if type(tmp) == str])

            if len(first_draft_str) == 0:
                first_draft_str = "null"
            if len(second_draft_str) == 0:
                second_draft_str = "null"

            print(first_draft_str)
            print(second_draft_str)

            first_draft_processed.append(first_draft_str)
            second_draft_processed.append(second_draft_str)

    space_data["first_draft_processed"] = first_draft_processed
    space_data["second_draft_processed"] = second_draft_processed

    # mvp153 processed data
    first_draft_processed = []
    second_draft_processed = []
    for i in tqdm(range(len(mvp153_data))):
        id = mvp153_data["essay_id"].tolist()[i]
        file_path = os.path.join(mvp153_processed_data_dir, f"{id}.xlsx")
        single_data = pd.read_excel(file_path)
        select_data = single_data[single_data["coarse_labels"].isnull() != True]
        if len(select_data) == 0:
            first_draft_processed.append("null")
            second_draft_processed.append("null")
        else:
            if args.use_purpose_label:

                first_draft = select_data["old_sentences"].apply(
                    lambda x: "" if type(x) == float and np.isnan(x) else x).tolist()
                second_draft = select_data["new_sentences"].apply(
                    lambda x: "" if type(x) == float and np.isnan(x) else x).tolist()

                coarse_labels = select_data["coarse_labels"].apply(
                    lambda x: "" if type(x) == float and np.isnan(x) else x).tolist()
                if "Unnamed: 7" in select_data.columns:
                    evidence_reason = select_data["Unnamed: 7"].apply(
                        lambda x: "" if type(x) == float and np.isnan(x) else x).tolist()
                else:
                    evidence_reason = [""] * len(select_data)

                if "Unnamed: 8" in select_data.columns:
                    purpose_labels = select_data["Unnamed: 8"].apply(
                        lambda x: "" if type(x) == float and np.isnan(x) else x).tolist()
                else:
                    purpose_labels = [""] * len(select_data)

                first_tmp_list = []
                second_tmp_list = []
                for j in range(len(first_draft)):
                    if len(first_draft) > 0:
                        if coarse_labels[j].strip() == "surface":
                            template = f"[{coarse_labels[j].strip()}].".replace("  ", "").replace(" ]", "]")
                        else:
                            template = f"[{coarse_labels[j].strip()} {evidence_reason[j].strip()} {purpose_labels[j].strip()}] {first_draft[j].strip()}".replace(
                                "  ", "").replace(" ]", "]")
                        first_tmp_list.append(template)
                    else:
                        first_tmp_list.append("")

                    if len(second_draft) > 0:
                        if coarse_labels[j].strip() == "surface":
                            template = f"[{coarse_labels[j].strip()}].".replace("  ", "").replace(" ]", "]")
                        else:
                            template = f"[{coarse_labels[j].strip()} {evidence_reason[j].strip()} {purpose_labels[j].strip()}] {second_draft[j].strip()}".replace(
                                "  ", "").replace(" ]", "]")
                        second_tmp_list.append(template)
                    else:
                        second_tmp_list.append("")

                first_draft_str = " ".join([x for x in first_tmp_list if not (x[0] == "[" and x[-2:] == "] ")])
                second_draft_str = " ".join([x for x in second_tmp_list if not (x[0] == "[" and x[-2:] == "] ")])

            else:
                first_draft = select_data["old_sentences"].apply(
                    lambda x: "" if type(x) == float and np.isnan(x) else x).tolist()
                second_draft = select_data["new_sentences"].apply(
                    lambda x: "" if type(x) == float and np.isnan(x) else x).tolist()

                first_draft_str = " ".join([tmp for tmp in first_draft if type(tmp) == str])
                second_draft_str = " ".join([tmp for tmp in second_draft if type(tmp) == str])

            if len(first_draft_str) == 0:
                first_draft_str = "null"
            if len(second_draft_str) == 0:
                second_draft_str = "null"
            first_draft_processed.append(first_draft_str)
            second_draft_processed.append(second_draft_str)
    mvp153_data["first_draft_processed"] = first_draft_processed
    mvp153_data["second_draft_processed"] = second_draft_processed

    # mvp143 processed data
    first_draft_processed = []
    second_draft_processed = []
    for i in tqdm(range(len(mvp143_data))):
        id = mvp143_data["essay_id"].tolist()[i]
        file_path = os.path.join(mvp143_processed_data_dir, f"{id}.xlsx")
        single_data = pd.read_excel(file_path)
        select_data = single_data[single_data["coarse_labels"].isnull() != True]
        if len(select_data) == 0:
            first_draft_processed.append("null")
            second_draft_processed.append("null")
        else:
            if args.use_purpose_label:

                first_draft = select_data["old_sentences"].apply(
                    lambda x: "" if type(x) == float and np.isnan(x) else x).tolist()
                second_draft = select_data["new_sentences"].apply(
                    lambda x: "" if type(x) == float and np.isnan(x) else x).tolist()

                coarse_labels = select_data["coarse_labels"].apply(
                    lambda x: "" if type(x) == float and np.isnan(x) else x).tolist()
                if "Unnamed: 7" in select_data.columns:
                    evidence_reason = select_data["Unnamed: 7"].apply(
                        lambda x: "" if type(x) == float and np.isnan(x) else x).tolist()
                else:
                    evidence_reason = [""] * len(select_data)

                if "Unnamed: 8" in select_data.columns:
                    purpose_labels = select_data["Unnamed: 8"].apply(
                        lambda x: "" if type(x) == float and np.isnan(x) else x).tolist()
                else:
                    purpose_labels = [""] * len(select_data)

                first_tmp_list = []
                second_tmp_list = []
                for j in range(len(first_draft)):
                    if len(first_draft) > 0:
                        if coarse_labels[j].strip() == "surface":
                            template = f"[{coarse_labels[j].strip()}].".replace("  ", "").replace(" ]", "]")
                        else:
                            template = f"[{coarse_labels[j].strip()} {evidence_reason[j].strip()} {purpose_labels[j].strip()}] {first_draft[j].strip()}".replace(
                                "  ", "").replace(" ]", "]")
                        first_tmp_list.append(template)
                    else:
                        first_tmp_list.append("")

                    if len(second_draft) > 0:
                        if coarse_labels[j].strip() == "surface":
                            template = f"[{coarse_labels[j].strip()}].".replace("  ", "").replace(" ]", "]")
                        else:
                            template = f"[{coarse_labels[j].strip()} {evidence_reason[j].strip()} {purpose_labels[j].strip()}] {second_draft[j].strip()}".replace(
                                "  ", "").replace(" ]", "]")
                        second_tmp_list.append(template)
                    else:
                        second_tmp_list.append("")

                first_draft_str = " ".join([x for x in first_tmp_list if not (x[0] == "[" and x[-2:] == "] ")])
                second_draft_str = " ".join([x for x in second_tmp_list if not (x[0] == "[" and x[-2:] == "] ")])

            else:
                first_draft = select_data["old_sentences"].apply(
                    lambda x: "" if type(x) == float and np.isnan(x) else x).tolist()
                second_draft = select_data["new_sentences"].apply(
                    lambda x: "" if type(x) == float and np.isnan(x) else x).tolist()

                first_draft_str = " ".join([tmp for tmp in first_draft if type(tmp) == str])
                second_draft_str = " ".join([tmp for tmp in second_draft if type(tmp) == str])

            if len(first_draft_str) == 0:
                first_draft_str = "null"
            if len(second_draft_str) == 0:
                second_draft_str = "null"
            first_draft_processed.append(first_draft_str)
            second_draft_processed.append(second_draft_str)
    mvp143_data["first_draft_processed"] = first_draft_processed
    mvp143_data["second_draft_processed"] = second_draft_processed

    if args.data_source == "mixture":
        data = pd.concat([space_data, mvp143_data, mvp153_data], ignore_index=True)
    elif args.data_source == "mvp":
        data = pd.concat([mvp143_data, mvp153_data], ignore_index=True)
    elif args.data_source == "space":
        data = space_data
    else:
        raise "wrong data source"

    data = pd.merge(data, feedback_text,  how='left', on=["feedback_level", "revision_score"])
    # data = data.sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
    data = data.sample(frac=1, random_state=22).reset_index(drop=True)

    return data


def clean_text(text):

    if not isinstance(text, str):
        text = ""

    text = re.sub(r'[\n|\r]', ' ', text)
    # remove space between ending word and punctuations
    text = re.sub(r'[ ]+([\.\?\!\,]{1,})', r'\1 ', text)
    # remove duplicated spaces
    text = re.sub(r' +', ' ', text)
    # add space if no between punctuation and words
    text = re.sub(r'([a-z|A-Z]{2,})([\.\?\!]{1,})([a-z|A-Z]{1,})', r'\1\2\n\3', text)
    # handle case "...\" that" that the sentence spliter cannot do
    text = re.sub(r'([\?\!\.]+)(\")([\s]+)([a-z|A-Z]{1,})', r'\1\2\n\3\4', text)
    # remove space between letter and punctuation
    text = re.sub(r'([a-z|A-Z]{2,})([ ]+)([\.\?\!])', r'\1\3', text)
    # handle case '\".word' that needs space after '.'
    text = re.sub(r'([\"\']+\.)([a-z|A-Z]{1,})', r'\1\n\2', text)
    # handle case '.\"word' that needs space after '\"'
    text = re.sub(r'(\.[\"\']+)([a-z|A-Z]{1,})', r'\1\n\2', text)
    # text = re.sub('\n', ' ', text)
    text = text.strip()
    text = text.lower()

    if len(text) > 0 and text[-1].isalpha():
        text += "."
    return text



def get_file_paths(path):
    paths = glob(os.path.join(path, "*.xlsx"))
    return paths

def get_error_marker_stats(file_paths):
    error_num = 0
    total_num = 0
    surface_num = 0
    content_num = 0
    tmp = []
    corrected_purpose_label_list = []
    purpose_category_label_list = []
    purpose_secondary_category_label_list = []
    for path in file_paths:
        data = pd.read_excel(path)
        data = clean_df_data(data)
        # data["coarse_labels"] = data["coarse_labels"].apply(lambda x: x.strip() if isinstance(x, str) else x)
        # error marker
        if "Unnamed: 9" in data.columns:
            # data["Unnamed: 9"] = data["Unnamed: 9"].apply(lambda x: x.strip() if isinstance(x, str) else x)
            select_data = data[data["coarse_labels"].notnull() & data["Unnamed: 9"].notnull()]
            corrected_label = select_data["coarse_labels"].tolist()
            purpose_category_label = select_data["Unnamed: 7"].tolist()
            purpose_secondary_category_label = select_data["Unnamed: 8"].tolist()
            error_col = select_data["Unnamed: 9"].tolist()
            corrected_purpose_label_list += corrected_label
            purpose_category_label_list += purpose_category_label
            purpose_secondary_category_label_list += purpose_secondary_category_label
            error_num += error_col.count("X")

        coarse_label_col = data["coarse_labels"]
        total_num += coarse_label_col.count()

        if "surface" in coarse_label_col.value_counts():
            surface_value = coarse_label_col.value_counts()["surface"]
        else:
            surface_value = 0
        if "content" in coarse_label_col.value_counts():
            content_value = coarse_label_col.value_counts()["content"]
        else:
            content_value = 0

        if "claim" in coarse_label_col.value_counts().keys().tolist() or "evidence" in coarse_label_col.value_counts().keys().tolist() or "reasoning" in coarse_label_col.value_counts().keys().tolist() or "surace" in coarse_label_col.value_counts().keys().tolist():
            print(path)

        surface_num += surface_value
        content_num += content_value
    print(Counter(tmp))
    counter_dict = Counter(corrected_purpose_label_list)
    counter_dict_purpose = Counter(purpose_category_label_list)
    counter_dict_secondary_purpose = Counter(purpose_secondary_category_label_list)
    for key in counter_dict:
        if key == "surface":
            print(f"{counter_dict[key]} out of {content_num} content labels had errors ({int(counter_dict[key])/content_num:4f})")
            print(counter_dict_purpose)
            print(counter_dict_secondary_purpose)
        elif key == "content":
            print(f"{counter_dict[key]} out of {surface_num} surface labels had errors ({int(counter_dict[key])/surface_num:4f})")
        else:
            print(counter_dict)
            # raise "wrong key!"
    print(f"{error_num} out of {total_num} errors ({error_num/total_num:4f})")

def clean_df_data(df):
    # remove space before or after strings
    for col_name in df.columns:
        df[col_name] = df[col_name].apply(lambda x: x.strip() if isinstance(x, str) else x)
    # replace typos
    if "Unnamed: 7" in df.columns:
        df["Unnamed: 7"] = df["Unnamed: 7"].apply(lambda x: x if isinstance(x, str) else x)
    if "Unnamed: 8" in df.columns:
        df["Unnamed: 8"] = df["Unnamed: 8"].apply(lambda x: x.replace("releveant", "relevant").replace("not relevant", "irrelevant").replace("CLE", "LCE") if isinstance(x, str) else x)
    if "coarse_labels" in df.columns:
        df["coarse_labels"] = df["coarse_labels"].apply(lambda x: x.replace("surace", "surface") if isinstance(x, str) else x)
    return df

def get_desirable_label_stats(file_paths):
    purpose_label_list = []
    purpose_category_list = []

    for path in file_paths:
        if "IGNORE" in path: continue # skip cracked data
        data = pd.read_excel(path)
        data = clean_df_data(data)
        if "Unnamed: 8" in data.columns:
            select_data = data[data["Unnamed: 7"].notnull() & data["Unnamed: 8"].notnull()]
            purpose_category_list += select_data["Unnamed: 7"].tolist()
            purpose_label_list += select_data["Unnamed: 8"].tolist()

    print(Counter(purpose_category_list))
    print(Counter(purpose_label_list))
    # for key in Counter(purpose_label_list).keys():
    #     print(key, Counter(purpose_label_list)[key])

    res_dict = {"evidence_desirable": 0, "evidence_undesirable": 0, "reasoning_desirable": 0, "reasoning_undesirable": 0}
    for i in range(len(purpose_category_list)):
        category = purpose_category_list[i]
        label = purpose_label_list[i]

        if category == "reasoning" and label in ['LCE', 'paraphrase']:
            res_dict["reasoning_desirable"] += 1
        elif category == "reasoning" and label in ['not LCE', 'commentary', 'minimal', 'generic']:
            res_dict["reasoning_undesirable"] += 1
        elif category == "evidence" and label in ['relevant']:
            res_dict["evidence_desirable"] += 1
        elif category == "evidence" and label in ['irrelevant', 'already exists', 'not text based', 'minimal']:
            res_dict["evidence_undesirable"] += 1
        else:
            print(category, label)

    print(f"revision purpose")
    for key in res_dict:
        print(key, res_dict[key])



def get_revision_purpose_data(file_paths):
    purpose_label_list = []
    purpose_category_list = []
    old_sentence_list = []
    new_sentence_list = []
    essay_id_list = []
    for path in file_paths:
        if "IGNORE" in path: continue # skip cracked data
        essay_id = path.split("/")[-1].split(".")[0]
        data = pd.read_excel(path)
        data = clean_df_data(data)
        if "Unnamed: 8" in data.columns:
            select_data = data[data["Unnamed: 7"].notnull() & data["Unnamed: 8"].notnull()]
            purpose_category_list += select_data["Unnamed: 7"].tolist()
            purpose_label_list += select_data["Unnamed: 8"].tolist()
            old_sentence_list += select_data["old_sentences"].tolist()
            new_sentence_list += select_data["new_sentences"].tolist()
            essay_id_list += [essay_id] * len(select_data)

    res_dict = {"evidence_desirable": 0, "evidence_undesirable": 0, "reasoning_desirable": 0, "reasoning_undesirable": 0}
    res_old_sent_list = []
    res_new_sent_list = []
    res_purpose_list = []
    res_purpose_label_list = []
    res_desirable_list = []
    res_essay_id_list = []
    for i in range(len(purpose_category_list)):
        category = purpose_category_list[i]
        label = purpose_label_list[i]
        old_sent = old_sentence_list[i]
        new_sent = new_sentence_list[i]
        desirable_essay_id = essay_id_list[i]

        if isinstance(old_sent, float):
            old_sent = "delete bad sentence"
        if isinstance(new_sent, float):
            new_sent = "insert good sentence"

        if category == "reasoning" and label in ['LCE', 'paraphrase']:
            res_dict["reasoning_desirable"] += 1
            desirable_label = 1
            purpose_label = "reasoning"
        elif category == "reasoning" and label in ['not LCE', 'commentary', 'minimal', 'generic']:
            res_dict["reasoning_undesirable"] += 1
            desirable_label = 0
            purpose_label = "reasoning"
        elif category == "evidence" and label in ['relevant']:
            res_dict["evidence_desirable"] += 1
            desirable_label = 1
            purpose_label = "evidence"
        elif category == "evidence" and label in ['irrelevant', 'already exists', 'not text based', 'minimal']:
            res_dict["evidence_undesirable"] += 1
            desirable_label = 0
            purpose_label = "evidence"
        else:
            print(category, label)
            desirable_label = None
            purpose_label = None

        if desirable_label is not None:
            res_old_sent_list.append(old_sent)
            res_new_sent_list.append(new_sent)
            res_purpose_list.append(purpose_label)
            res_purpose_label_list.append(label)
            res_desirable_list.append(desirable_label)
            res_essay_id_list.append(desirable_essay_id)

    df = pd.DataFrame({"essay_ids": res_essay_id_list, "old_sentences":res_old_sent_list, "new_sentences":res_new_sent_list, "revision_purpose": res_purpose_list, "desirable_labels": res_desirable_list, "revision_purpose_label":res_purpose_label_list})

    print(f"revision purpose")
    for key in res_dict:
        print(key, res_dict[key])

    return df


def add_disirable_label(data, is_college=False):
    fine_labels_list = data["fine_labels"].tolist()
    purpose_labels_list = data["purpose_labels"].tolist()
    old_sentence_list = data["old_sentences"].tolist()
    new_sentence_list = data["new_sentences"].tolist()
    desirable_labels_list = [np.nan]*len(data)

    for i in range(len(fine_labels_list)):
        if isinstance(fine_labels_list[i], float) or isinstance(purpose_labels_list[i], float):
            continue
        fine_label = fine_labels_list[i]
        purpose_label = purpose_labels_list[i]
        old_sent = old_sentence_list[i]
        new_sent = new_sentence_list[i]

        # delete old sentence:
        if isinstance(new_sent, float):
            if is_college:
                if fine_label == "reasoning" and purpose_label in ['LCE']:
                    desirable_label = "desirable"
                elif fine_label == "reasoning" and purpose_label in ['paraphrase', 'not LCE', 'commentary', 'minimal', 'generic']:
                    desirable_label = "undesirable"
                elif fine_label == "evidence" and purpose_label in ['relevant']:
                    desirable_label = "desirable"
                elif fine_label == "evidence" and purpose_label in ['irrelevant', 'already exists', 'not text based', 'minimal']:
                    desirable_label = "undesirable"
                else:
                    # print(path)
                    print(fine_label, purpose_label)
                    desirable_label = None
            else:
                if fine_label == "reasoning" and purpose_label in ['LCE', 'paraphrase']:
                    desirable_label = "desirable"
                elif fine_label == "reasoning" and purpose_label in ['not LCE', 'commentary', 'minimal', 'generic']:
                    desirable_label = "undesirable"
                elif fine_label == "evidence" and purpose_label in ['relevant']:
                    desirable_label = "desirable"
                elif fine_label == "evidence" and purpose_label in ['irrelevant', 'already exists', 'not text based', 'minimal']:
                    desirable_label = "undesirable"
                else:
                    # print(path)
                    print(fine_label, purpose_label)
                    desirable_label = None
        # add new sentences or modify
        else:
            if is_college:
                if fine_label == "reasoning" and purpose_label in ['LCE']:
                    desirable_label = "desirable"
                elif fine_label == "reasoning" and purpose_label in ['paraphrase', 'not LCE', 'commentary', 'minimal', 'generic']:
                    desirable_label = "undesirable"
                elif fine_label == "evidence" and purpose_label in ['relevant']:
                    desirable_label = "desirable"
                elif fine_label == "evidence" and purpose_label in ['irrelevant', 'already exists', 'not text based', 'minimal']:
                    desirable_label = "undesirable"
                else:
                    # print(path)
                    print(fine_label, purpose_label)
                    desirable_label = None

            else:
                if fine_label == "reasoning" and purpose_label in ['LCE', 'paraphrase']:
                    desirable_label = "desirable"
                elif fine_label == "reasoning" and purpose_label in ['not LCE', 'commentary', 'minimal', 'generic']:
                    desirable_label = "undesirable"
                elif fine_label == "evidence" and purpose_label in ['relevant']:
                    desirable_label = "desirable"
                elif fine_label == "evidence" and purpose_label in ['irrelevant', 'already exists', 'not text based', 'minimal']:
                    desirable_label = "undesirable"
                else:
                    # print(path)
                    print(fine_label, purpose_label)
                    desirable_label = None

        if desirable_label is not None:
            desirable_labels_list[i] = desirable_label

    data["desirable_labels"] = desirable_labels_list

    return data


class UncertaintyLoss(nn.Module):

    def __init__(self, v_num):
        super(UncertaintyLoss, self).__init__()
        sigma = torch.randn(v_num)
        self.sigma = nn.Parameter(sigma)
        self.v_num = v_num

    def forward(self, *input):
        loss = 0
        for i in range(self.v_num):
            loss += input[i] / (2 * self.sigma[i] ** 2)
        loss += torch.log(self.sigma.pow(2).prod())
        return loss


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
            # loss_sum += torch.sum(torch.exp(-self.params[i]**2)*loss + torch.log(1 + self.params[i] ** 2), -1)
            # print(self.params[i].item())
        return loss_sum

def extract_sentence(text):
    text = text + "\n"
    re_template = r'[^a-zA-Z0-9]*(.*?\n)'
    matches = re.findall(re_template, text)
    matches = [clean_text(x) for x in matches]
    match_str = "\n".join(matches)
    # match_str = text.replace("\n", " ")
    if len(match_str) == 0:
        match_str = clean_text(text)
    return match_str

def get_essay_sentences(df, type="evidence"):
    # openai.api_key = "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"
    openai.api_key = "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"

    # df = df.loc[20,:]

    first_list = []
    second_list = []
    for i in tqdm(range(len(df))):
        tmp_first_name = "first_"+type
        tmp_second_name = "second_"+type
        tmp_first = df[tmp_first_name].tolist()[i] if tmp_first_name in df.columns and str(df[tmp_first_name].tolist()[i])!="nan" else ""
        tmp_second = df[tmp_second_name].tolist()[i] if tmp_second_name in df.columns and str(df[tmp_second_name].tolist()[i])!="nan" else ""

        # if exist not call chatgpt
        if len(tmp_first)>0 and len(tmp_second)>0:
            tmp_first = extract_sentence(tmp_first)
            tmp_second = extract_sentence(tmp_second)
            first_list.append(tmp_first)
            second_list.append(tmp_second)
        else:

            first_draft = df["first_draft"].tolist()[i]
            second_draft = df["second_draft"].tolist()[i]
            id = df["essay_id"].tolist()[i]

            first_draft = clean_text(first_draft)
            second_draft = clean_text(second_draft)

            if type == "summary":
                context_first = f"summarize in two sentence \n\n{first_draft}"
                context_second = f"summarize in two sentence \n\n{second_draft}"
            elif type == "claim_single":
                context_first = f"summarize claim sentence in two sentences \n\n{first_draft}."
                context_second = f"summarize claim sentence in two sentences \n\n{second_draft}."
            elif type == "evidence_single":
                context_first = f"summarize evidence sentence in two sentences \n\n{first_draft}."
                context_second = f"summarize evidence sentence in two sentences \n\n{second_draft}."
            elif type == "reasoning_single":
                context_first = f"summarize reasoning sentence in two sentences \n\n{first_draft}."
                context_second = f"summarize reasoning sentence in two sentences \n\n{second_draft}."
            elif type == "claim":
                context_first = f"list claim sentence in bullets \n\n{first_draft}."
                context_second = f"list claim sentence in bullets \n\n{second_draft}."
            elif type == "evidence":
                context_first = f"list evidence sentence in bullets \n\n{first_draft}."
                context_second = f"list evidence sentence in bullets \n\n{second_draft}."
            elif type == "reasoning":
                context_first = f"list reasoning sentence in bullets \n\n{first_draft}."
                context_second = f"list reasoning sentence in bullets \n\n{second_draft}."
            elif type == "claim_summary":
                first_draft = df["first_claim"].tolist()[i]
                second_draft = df["second_claim"].tolist()[i]
                context_first = f"summarize in two sentences \n\n{first_draft}."
                context_second = f"summarize in two sentences \n\n{second_draft}."
            elif type == "evidence_summary":
                first_draft = df["first_evidence"].tolist()[i]
                second_draft = df["second_evidence"].tolist()[i]
                context_first = f"summarize in two sentences \n\n{first_draft}."
                context_second = f"summarize in two sentences \n\n{second_draft}."
            elif type == "reasoning_summary":
                first_draft = df["first_reasoning"].tolist()[i]
                second_draft = df["second_reasoning"].tolist()[i]
                context_first = f"summarize in two sentences \n\n{first_draft}."
                context_second = f"summarize in two sentences \n\n{second_draft}."
            else:
                raise "wrong context!"


            try:
                completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0301", messages=[
                    {"role": "user", "content": context_first}])
                completion_first = completion.choices[0].message.content
                completion_first_str = extract_sentence(completion_first)

                if len(completion_first_str) == 0:
                    completion_first_str = completion_first

                completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0301", messages=[
                    {"role": "user", "content": context_second}])
                completion_second = completion.choices[0].message.content
                completion_second_str = extract_sentence(completion_second)

                if len(completion_second_str) == 0:
                    completion_second_str = completion_second

            except Exception as err:
                print(f"error essay: {id}" , err)
                completion_first_str = ""
                completion_second_str = ""
                time.sleep(10)  # limited 20 runs per minute

            first_list.append(completion_first_str)
            second_list.append(completion_second_str)


    df[f"first_{type}"] = first_list
    df[f"second_{type}"] = second_list

    return df

def get_essay_sentences_new(df, type="evidence"):
    # openai.api_key = "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"
    # openai.api_key = "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"

    openai.api_type = "azure"
    openai.api_base = "https://erevise.openai.azure.com/"
    openai.api_version = "2023-03-15-preview"
    openai.api_key = "7367ecb09bf84e5fb4a52374a958d99e"

    # df = df.loc[20,:]

    first_list = []
    second_list = []
    for i in tqdm(range(len(df))):
        tmp_first_name = "first_"+type
        tmp_second_name = "second_"+type
        tmp_first = df[tmp_first_name].tolist()[i] if tmp_first_name in df.columns and str(df[tmp_first_name].tolist()[i])!="nan" else ""
        tmp_second = df[tmp_second_name].tolist()[i] if tmp_second_name in df.columns and str(df[tmp_second_name].tolist()[i])!="nan" else ""

        # if exist not call chatgpt
        if len(tmp_first)>0 and len(tmp_second)>0:
            tmp_first = extract_sentence(tmp_first)
            tmp_second = extract_sentence(tmp_second)
            first_list.append(tmp_first)
            second_list.append(tmp_second)
        else:

            first_draft = df["first_draft"].tolist()[i]
            second_draft = df["second_draft"].tolist()[i]
            id = df["essay_id"].tolist()[i]

            first_draft = clean_text(first_draft)
            second_draft = clean_text(second_draft)

            if type == "summary":
                context_first = f"summarize the essay in two sentence"
                context_second = f"summarize the essay in two sentence"
            elif type == "claim_single":
                context_first = f"summarize the claim in two sentences."
                context_second = f"summarize the claim in two sentences."
            elif type == "evidence_single":
                context_first = f"summarize the evidence in two sentences."
                context_second = f"summarize the evidence in two sentences."
            elif type == "reasoning_single":
                context_first = f"summarize the reasoning in two sentences."
                context_second = f"summarize the reasoning in two sentences."
            elif type == "claim":
                context_first = f"list claim sentence in bullets."
                context_second = f"list claim sentence in bullets."
            elif type == "evidence":
                context_first = f"list evidence sentence in bullets."
                context_second = f"list evidence sentence in bullets."
            elif type == "reasoning":
                context_first = f"list reasoning sentence in bullets."
                context_second = f"list reasoning sentence in bullets."
            elif type == "claim_summary":
                first_draft = df["first_claim"].tolist()[i]
                second_draft = df["second_claim"].tolist()[i]
                context_first = f"summarize in two sentences."
                context_second = f"summarize in two sentences."
            elif type == "evidence_summary":
                first_draft = df["first_evidence"].tolist()[i]
                second_draft = df["second_evidence"].tolist()[i]
                context_first = f"summarize in two sentences."
                context_second = f"summarize in two sentences."
            elif type == "reasoning_summary":
                first_draft = df["first_reasoning"].tolist()[i]
                second_draft = df["second_reasoning"].tolist()[i]
                context_first = f"summarize in two sentences."
                context_second = f"summarize in two sentences."
            else:
                raise "wrong context!"


            try:
                completion = openai.ChatCompletion.create(
                    engine="gpt-35-turbo",  # engine = "deployment_name".
                    temperature=0,
                    messages=[{"role": "system", "content": first_draft},
                              {"role": "user", "content": context_first}])

                # completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0301", messages=[
                #     {"role": "user", "content": context_first}])
                completion_first = completion.choices[0].message.content
                completion_first_str = extract_sentence(completion_first)

                if len(completion_first_str) == 0:
                    completion_first_str = completion_first

                completion = openai.ChatCompletion.create(
                    engine="gpt-35-turbo",  # engine = "deployment_name".
                    temperature=0,
                    messages=[{"role": "system", "content": second_draft},
                              {"role": "user", "content": context_second}])
                # completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0301", messages=[
                #     {"role": "user", "content": context_second}])
                completion_second = completion.choices[0].message.content
                completion_second_str = extract_sentence(completion_second)

                if len(completion_second_str) == 0:
                    completion_second_str = completion_second

            except Exception as err:
                print(f"error essay: {id}" , err)
                completion_first_str = ""
                completion_second_str = ""
                time.sleep(10)  # limited 20 runs per minute

            first_list.append(completion_first_str)
            second_list.append(completion_second_str)


    df[f"first_{type}"] = first_list
    df[f"second_{type}"] = second_list

    return df
