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
    # get relations csv
    csv_name = os.path.join('Manual_Trans','Manual_Trans','C1007_VC_3_FullCon_Wk01_Day3_100318.csv')
    df = pd.read_csv(csv_name)

    # get rid of weird floats
    df.dropna(inplace=True)

    # group by role
    grouped = df.groupby(df.role.ne(df.role.shift()).cumsum(), as_index=False).agg({'role': 'first', 'text': ' '.join})
    grouped.pair = list(map(join_sent, grouped.text, grouped.text.shift()))
    print(grouped.pair)

    # construct tuples
    print("model loading ...")
    comet = Comet(os.path.join('.', 'comet-atomic_2020_BART_aaai'))
    comet.model.zero_grad()
    print("model loaded")
    properties = {
        'openie.affinity_probability_cap': 2 / 3,
    }

    with StanfordOpenIE(properties=properties) as client:
        new_df = pd.DataFrame(columns=['queries', 'results'])

        save_res_df = pd.DataFrame(columns=['text', 'queries', 'results'])
        with open('results.txt', 'w+') as f:
            for i in range(len(df)):
                offset = 5
                if len(df) - i < 5:
                    offset = len(df)-i
                rows = df.iloc[i : i+offset]
                queries = []
                queries_fixed = []
                # for row in rows:
                text = ' '.join(rows.text)
                print(text)
                # print('Text: %s.' % text)
                try:
                    for triple in client.annotate(text):
                        print(type(triple))
                        print(triple)
                        queries.append('{} {}'.format(triple['subject'], triple['relation']))
                        queries_fixed.append('{} {}'.format('PersonX', triple['relation']))

                    f.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nQUERIES\n~~~~~~~~~~~~~~~~~~\n{}\n'.format(queries))
                    results = comet.generate(queries, decode_method="greedy", num_generate=2)
                    f.write('{}\n'.format(results))
                    f.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nQUERIES FIXED\n~~~~~~~~~~~~~~~~~~\n{}\n'.format(queries_fixed))
                    results = comet.generate(queries_fixed, decode_method="greedy", num_generate=2)
                    f.write('{}\n'.format(results))

                    # save queries/results
                    save_res_df = pd.concat([save_res_df, {'text': text, 'queries': queries, 'results': results}])
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



