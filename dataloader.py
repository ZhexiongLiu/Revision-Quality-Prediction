from transformers import AutoTokenizer #, AutoModelForMaskedLM
from torch.utils.data import Dataset
from utilities import *

class TextDataset(Dataset):
    def __init__(self, args, data):
        self.essay_id = data["essay_id"]
        self.label = data["revision_score"].tolist()
        self.feedback_level = data["feedback_level"].tolist()
        self.feedback_text = data["feedback_text"].tolist()
        self.revision_score = data["revision_score"].tolist()
        self.raw_first_draft = data["first_draft"].tolist()
        self.raw_second_draft = data["second_draft"].tolist()
        self.processed_first_draft = data["first_draft_processed"].tolist()
        self.processed_second_draft = data["second_draft_processed"].tolist()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        if args.merge_label:
            self.label = [x if x!=3 else 2 for x in self.label]

        if args.data_type == "raw":
            fst_draft = self.raw_first_draft
            snd_draft = self.raw_second_draft
        else:
            fst_draft = self.processed_first_draft
            snd_draft = self.processed_second_draft

        self.fst_tokens = [self.tokenizer(clean_text(text),
                                padding='max_length', truncation=True,
                                return_tensors="pt") for text in fst_draft]

        self.snd_tokens = [self.tokenizer(clean_text(text),
                                 padding='max_length', truncation=True,
                                 return_tensors="pt") for text in snd_draft]

        self.fb_tokens = [self.tokenizer(clean_text(text),
                                 padding='max_length', truncation=True,
                                 return_tensors="pt") for text in self.feedback_text]


    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        essay_id = self.essay_id[idx]
        fst_token = self.fst_tokens[idx]
        snd_token = self.snd_tokens[idx]
        fb_token = self.fb_tokens[idx]
        fb_level = self.feedback_level[idx]
        label = self.label[idx]
        return essay_id, fst_token, snd_token, fb_token, fb_level, label


class DesirableDataset(Dataset):
    def __init__(self, args, data):
        self.essay_ids = data["essay_ids"].apply(lambda x: int(x))
        self.context = data["context"].apply(lambda x: clean_text(x)).tolist()
        self.old_sentences = data["old_sentences"].apply(lambda x: clean_text(x)).tolist()
        self.new_sentences = data["new_sentences"].apply(lambda x: clean_text(x)).tolist()
        self.desirable_labels = data["desirable_labels"].apply(lambda x: encode_desirable_label(x)).tolist()
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-distilroberta-v1")

        self.tokens = []
        for i in range(len(self.old_sentences)):
            if len(self.old_sentences[i]) > 0 and len(self.new_sentences[i]) > 0:
                first_sent = self.context[i]
                # first_sent = self.old_sentences[i]
                second_sent = self.new_sentences[i]
            elif len(self.old_sentences[i]) > 0:
                first_sent = self.old_sentences[i]
                second_sent = self.context[i]
            else:
                first_sent = self.context[i]
                second_sent = self.new_sentences[i]

            tk = self.tokenizer(first_sent, second_sent, padding="max_length", truncation=True, return_tensors="pt")
            # tk = self.tokenizer(first_sent, second_sent, max_length=512, truncation=True, padding='max_length', return_tensors="pt")

            # if "this is a comparing sentence" in self.old_sentences[i]:
            #     tk = self.tokenizer(self.old_sentences[i], self.new_sentences[i],
            #                         padding="max_length", truncation=True, return_tensors="pt")
            # else:
            #     tk = self.tokenizer(self.new_sentences[i], self.old_sentences[i],
            #                         padding="max_length", truncation=True, return_tensors="pt")
            self.tokens.append(tk)


    def __len__(self):
        return len(self.desirable_labels)

    def __getitem__(self, idx):
        essay_id = self.essay_ids[idx]
        token = self.tokens[idx]
        label = self.desirable_labels[idx]
        return essay_id, token, label
