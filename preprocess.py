import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
from utilities import  add_disirable_label

evidence_data = pd.read_excel("./data/college/EvidenceSentencePairData.xlsx")
reason_data = pd.read_excel("./data/college/ReasoningSentencePairData.xlsx")
doc_paths = glob("./data/college/revisions/*/*.xlsx")

eval_gold = []
eval_predict = []
purpose_dict = set()
for path in tqdm(doc_paths):
    print("Data file: ", path)
    if "esl" in path:
        essay_name = str(777777) + path.split("/")[-1].split(".")[0].split("_")[-1].replace("esl", "")
        essay_id = path.split("/")[-1].split(".")[0].split("_")[-1]
    else:
        essay_name = str(666666) + path.split("/")[-1].split(".")[0].split("_")[-1].replace("native", "")
        essay_id = path.split("/")[-1].split(".")[0].split("_")[-1]


    # if "e23." in path or "e127." in path:
    #     print()
    #     continue

    # if "e10." in path:
    #     print()

    # if "e153." in path or "e152." in path or "e109." in path or "e23." in path or "e52." in path or "e124." in path or "e50." in path or "e127." in path or "e55." in path:
    #     # continue
    #     print("skip")
    #     continue
    old_draft = pd.read_excel(path, sheet_name="Old Draft")
    new_draft = pd.read_excel(path, sheet_name="New Draft")

    purpose_dict.update(old_draft["Revision Purpose Level 0"].tolist())
    purpose_dict.update(new_draft["Revision Purpose Level 0"].tolist())

    if "General Content Development,Conventions/Grammar/Spelling" in old_draft["Revision Purpose Level 0"].tolist():
        print("----------")

    old_draft_list = old_draft["Sentence Content"].tolist()
    new_draft_list = new_draft["Sentence Content"].tolist()

    tmp_old_gold_align = old_draft["Aligned Index"].apply(lambda x: x-1 if type(x)!=str else x).apply(lambda x: str(int(x)-1) if type(x)==str and x not in ["ADD", "DELETE"] else x).tolist()
    tmp_new_gold_align = new_draft["Aligned Index"].apply(lambda x: x-1 if type(x)!=str else x).apply(lambda x: str(int(x)-1) if type(x)==str and x not in ["ADD", "DELETE"] else x).tolist()

    old_sent_index = old_draft["Sentence Index"].apply(lambda x: x-1 if type(x)!=str else x).apply(lambda x: str(int(x)-1) if type(x)==str and x not in ["ADD", "DELETE"] else x).tolist()
    new_sent_index = new_draft["Sentence Index"].apply(lambda x: x-1 if type(x)!=str else x).apply(lambda x: str(int(x)-1) if type(x)==str and x not in ["ADD", "DELETE"] else x).tolist()

    old_gold_align = old_draft["Aligned Index"].apply(lambda x: x-1 if type(x)!=str else x).apply(lambda x: str(int(x)-1) if type(x)==str and x not in ["ADD", "DELETE"] else x).tolist()
    new_gold_align = new_draft["Aligned Index"].apply(lambda x: x-1 if type(x)!=str else x).apply(lambda x: str(int(x)-1) if type(x)==str and x not in ["ADD", "DELETE"] else x).tolist()

    old_purpose = old_draft["Revision Purpose Level 0"].tolist()
    new_purpose = new_draft["Revision Purpose Level 0"].tolist()

    # new draft
    latest_align_num = 0
    inserted_num = 0
    for i in range(len(tmp_new_gold_align)):
        tmp_align = tmp_new_gold_align[i]
        if str(tmp_align) != "ADD":
            latest_align_num = int(tmp_align)
        else:
            j = latest_align_num + inserted_num + 1
            # if delete and insert are conjuncted, do delete first then insert
            while j < len(old_gold_align) and old_gold_align[j] == "DELETE":
                j += 1
            old_draft_list.insert(j, np.NaN)
            old_sent_index.insert(j, np.NaN)
            old_gold_align.insert(j, np.NaN)
            old_purpose.insert(j, np.NaN)
            inserted_num += 1

    # old draft
    latest_align_num = 0
    inserted_num = 0
    for i in range(len(tmp_old_gold_align)):
        tmp_align = tmp_old_gold_align[i]
        if str(tmp_align) != "DELETE":
            latest_align_num = int(tmp_align)
        else:
            j = latest_align_num + inserted_num + 1
            new_draft_list.insert(j, np.NaN)
            new_sent_index.insert(j, np.NaN)
            new_gold_align.insert(j, np.NaN)
            new_purpose.insert(j, np.NaN)
            inserted_num += 1

    purpose_mapper = {'Evidence,Warrant/Reasoning/Backing': 'content',
                      'Warrant/Reasoning/Backing': 'content',
                      'Conventions/Grammar/Spelling': 'surface',
                      'Warrant/Reasoning/Backing,General Content Development': 'content',
                      'Word-Usage/Clarity': 'surface',
                      'Precision': 'content',
                      'General Content Development,Conventions/Grammar/Spelling': 'content',
                      'Organization': 'surface',
                      'Evidence': 'content',
                      'Rebuttal/Reservation': 'content',
                      'Claims/Ideas': 'content',
                      'General Content Development': 'content'}

    merged_purpose = []
    for i in range(len(new_purpose)):
        old_p = old_purpose[i]
        new_p = new_purpose[i]
        if type(old_p)==float and np.isnan(old_p):
            if type(new_p)==float and np.isnan(new_p):
                merged_purpose.append(new_p)
            else:
                merged_purpose.append(purpose_mapper[new_p])
        else:
            merged_purpose.append(purpose_mapper[old_p])

    reason_evidence_list = [np.NaN]*len(merged_purpose)
    fine_grain_label_list = [np.NaN]*len(merged_purpose)

    # reason data
    selected_reason_data = reason_data[reason_data["ID"]==essay_id]
    selected_reason_data = selected_reason_data.to_dict('records')
    for i in range(len(selected_reason_data)):
        tmp_idx1 = selected_reason_data[i]["index1"]
        tmp_idx2 = selected_reason_data[i]["index2"]
        if tmp_idx1 == -1:
            tmp_idx = new_sent_index.index(tmp_idx2-1)
        else:
            tmp_idx = old_sent_index.index(tmp_idx1-1)

        fine_label = selected_reason_data[i]["Label1"].replace("-","").replace("Repeat","already exists").replace("EVIDENCE","").replace("REASONING","").replace("NonTextBased","not text based").lower().replace("lce", "LCE").strip()
        this_label = "reasoning"

        reason_evidence_list[tmp_idx] = this_label
        fine_grain_label_list[tmp_idx] = fine_label

    # evidence data
    selected_evidence_data = evidence_data[evidence_data["ID"] == essay_id]
    selected_evidence_data = selected_evidence_data.to_dict('records')
    for i in range(len(selected_evidence_data)):
        tmp_idx1 = selected_evidence_data[i]["index1"]
        tmp_idx2 = selected_evidence_data[i]["index2"]
        if tmp_idx1 == -1:
            tmp_idx = new_sent_index.index(tmp_idx2 - 1)
        else:
            tmp_idx = old_sent_index.index(tmp_idx1 - 1)

        fine_label = selected_evidence_data[i]["Label1"].replace("-","").replace("Repeat","already exists").replace("EVIDENCE","").replace("REASONING","").replace("NonTextBased","not text based").lower().replace("lce", "LCE").strip()
        this_label = "evidence"
        if "nontextbased" in fine_label:
            print()

        reason_evidence_list[tmp_idx] = this_label
        fine_grain_label_list[tmp_idx] = fine_label


    df = pd.DataFrame({"old_sentence_ids": old_sent_index,
                       "old_sentence_aligned_ids": old_gold_align,
                       "old_sentences": old_draft_list,
                       "new_sentences": new_draft_list,
                       "new_sentence_ids": new_sent_index,
                       "new_sentence_aligned_ids": new_gold_align,
                       "coarse_labels": merged_purpose,
                       "fine_labels": reason_evidence_list,
                       "purpose_labels": fine_grain_label_list,
                       })
    if "Annotation_eagerstudy_esl40" in path:
        print()
    df = add_disirable_label(df, is_college=True)
    print(df)


    df.to_excel(f"./data_clean/revision_purpose/college/{essay_name}.xlsx", index=False)