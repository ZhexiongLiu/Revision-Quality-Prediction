import transformers
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification

class DesirableModel(torch.nn.Module):
    def __init__(self, args):
        super(DesirableModel, self).__init__()
        self.args = args
        out_dim = 1 # nn.BCEWithLogitsLoss()

        if args.model_name in ["xlm-roberta-base", "distilbert-base-uncased", "distilroberta-base", "gpt2"]:
            embedding_dim = 768
        elif args.model_name in ["roberta-large", "bert-large", "gpt2-large"]:
            embedding_dim = 1024
        elif "longformer" in args.model_name:
            embedding_dim = 2048
        else:
            raise "wrong model name"

        self.model = AutoModelForSequenceClassification.from_pretrained(args.model_name)

        for param in self.model.parameters():
            if args.freeze_model:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.fc = torch.nn.Sequential(
            torch.nn.BatchNorm1d(embedding_dim, eps=1e-05, momentum=0.1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5, inplace=False),
            torch.nn.Linear(embedding_dim, out_dim))


    def forward(self, input_ids, attention_mask):
        hidden_out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
        out = torch.mean(hidden_out, dim=1)
        out = F.normalize(out, dim=-1)
        out = self.fc(out)
        return out
