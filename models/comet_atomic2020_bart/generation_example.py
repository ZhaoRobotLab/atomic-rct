import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import calculate_rouge, use_task_specific_params, calculate_bleu_score, trim_batch
import pandas as pd
from openie import StanfordOpenIE
import os

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class Comet:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarization"
        use_task_specific_params(self.model, task)
        self.batch_size = 1
        self.decoder_start_token_id = None

    def generate(
            self, 
            queries,
            decode_method="beam", 
            num_generate=5, 
            ):

        with torch.no_grad():
            examples = queries

            decs = []
            for batch in list(chunks(examples, self.batch_size)):

                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)

                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=num_generate,
                    num_return_sequences=num_generate,
                    )

                dec = self.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                decs.append(dec)

            return decs


all_relations = [
    "AtLocation",
    "CapableOf",
    "Causes",
    "CausesDesire",
    "CreatedBy",
    "DefinedAs",
    "DesireOf",
    "Desires",
    "HasA",
    "HasFirstSubevent",
    "HasLastSubevent",
    "HasPainCharacter",
    "HasPainIntensity",
    "HasPrerequisite",
    "HasProperty",
    "HasSubEvent",
    "HasSubevent",
    "HinderedBy",
    "InheritsFrom",
    "InstanceOf",
    "IsA",
    "LocatedNear",
    "LocationOfAction",
    "MadeOf",
    "MadeUpOf",
    "MotivatedByGoal",
    "NotCapableOf",
    "NotDesires",
    "NotHasA",
    "NotHasProperty",
    "NotIsA",
    "NotMadeOf",
    "ObjectUse",
    "PartOf",
    "ReceivesAction",
    "RelatedTo",
    "SymbolOf",
    "UsedFor",
    "isAfter",
    "isBefore",
    "isFilledBy",
    "oEffect",
    "oReact",
    "oWant",
    "xAttr",
    "xEffect",
    "xIntent",
    "xNeed",
    "xReact",
    "xReason",
    "xWant",
    ]

def join_sent(s1, s2):
    if s1 == None or s2 == None:
        return ''
    return s1 + " " + s2

if __name__ == "__main__":
    # set environment var for java
    java_path = 'C:\\Program Files (x86)\\Java\\jdk1.8.0_301\\bin\\java.exe'
    os.environ['JAVAHOME'] = java_path
    # os.environ["CORENLP_HOME"] = 'C:\\Users\\rbfre\\.stanfordnlp_resources\\stanford-corenlp-4.1.0\\*'
    # get relations csv, loop on the directory
    # data\Manual_Trans\C1007_VC_3_FullCon_Wk01_Day3_100318.csv
    datadir = os.path.join('data', 'Manual_Trans')
    files = os.listdir(datadir)
    for file in files:
        if not file.endswith('.csv'):
            continue
        csv_name = os.path.join(os.path.join(datadir, file))
        df = pd.read_csv(csv_name)

        # get rid of weird floats
        df.dropna(inplace=True)

        # group by role
        grouped = df.groupby(df.role.ne(df.role.shift()).cumsum(), as_index=False).agg({'role': 'first', 'text': ' '.join})
        grouped.pair = list(map(join_sent, grouped.text.shift(), grouped.text))
        print(grouped.pair)

        # make rounds
        rounds = []
        for indx in range(0, len(grouped)-1, 2):
            rounds.append(join_sent(grouped.iloc[indx].text, grouped.iloc[indx+1].text))

        # construct tuples
        print("model loading ...")
        comet = Comet(os.path.join('models','comet_atomic2020_bart','comet-atomic_2020_BART_aaai'))
        comet.model.zero_grad()
        print("model loaded")
        properties = {
            'openie.affinity_probability_cap': 2 / 3,
            # 'memory': '1G',
            # 'timeout': 1_000_000
        }

        with StanfordOpenIE(properties=properties) as client:
            new_df = pd.DataFrame(columns=['queries', 'results'])

            save_res_df = pd.DataFrame(columns=['text', 'round', 'res_dict_raw', 'res_dict_fixed'])
            num_rounds = 5
            queries = []
            queries_fixed = []
            save_res_dict_raw = {}
            save_res_dict_fixed = {}
            for i in range(len(rounds)):
                # reset queries & dictionaries
                queries = []
                queries_fixed = []
                save_res_dict_raw = {}
                save_res_dict_fixed = {}
                # for row in rows:
                text = rounds[i]
                print(text)
                # print('Text: %s.' % text)
                try:
                    for triple in client.annotate(text, properties=properties):
                        print(type(triple))
                        print(triple)
                        queries.append('{} {}'.format(triple['subject'], triple['relation']))
                        queries_fixed.append('{} {}'.format('PersonX', triple['relation']))

                    results_raw = comet.generate(queries, decode_method="greedy", num_generate=1)
                    results_fixed = comet.generate(queries_fixed, decode_method="greedy", num_generate=1)
                    
                    # save queries/results
                    #make dictionary for storing results
                    for indx in range(len(results_raw)):
                        save_res_dict_raw[queries[indx]] = results_raw[indx]
                        save_res_dict_fixed[queries_fixed[indx]] = results_fixed[indx]

                    # concatenate results to df and save to csv every time j
                    new_df = pd.DataFrame.from_dict({
                        'text': text, 
                        'round': i,
                        'res_dict_raw': [save_res_dict_raw], 
                        'res_dict_fixed': [save_res_dict_fixed]}).astype('object')
                    save_res_df = pd.concat([save_res_df, new_df], ignore_index=True)
                    save_res_df.to_csv(os.path.join('results', file))
                    print(save_res_df)
                except AttributeError as e:
                    print(e)  
        
    # sample usage (reproducing AAAI)
    
    # queries = []
    # head = "PersonX pleases ___ to make"
    # rel = "xWant"
    # head = "PersonX"
    # rel = "was"
    # query = "{} {} [GEN]".format(head, rel)
    # queries.append(query)
    
    


    # sample usage (reproducing demo)
    # print("model loading ...")
    # comet = Comet("./comet-atomic_2020_BART_aaai")
    # comet.model.zero_grad()
    # print("model loaded")
    # queries = []
    # head = "PersonX pleases ___ to make"
    # rel = "xWant"
    # query = "{} {} [GEN]".format(head, rel)
    # queries.append(query)
    # print(queries)
    # results = comet.generate(queries, decode_method="beam", num_generate=5)
    # print(results)



