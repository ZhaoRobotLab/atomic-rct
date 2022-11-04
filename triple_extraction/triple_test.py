from openie import StanfordOpenIE
import pandas as pd

# https://stanfordnlp.github.io/CoreNLP/openie.html#api
# Default value of openie.affinity_probability_cap was 1/3.
properties = {
    'openie.affinity_probability_cap': 2 / 3,
}

with StanfordOpenIE(properties=properties) as client:
    df = pd.read_csv('../data/C1007_VC_3_FullCon_Wk01_Day1_100118.ft_DS2.csv')
    new_df = pd.DataFrame(columns=['subject', 'relation', 'object', 'text'])

    for indx, row in df.iterrows():
        text = row.asr
        # print('Text: %s.' % text)
        try:
            for triple in client.annotate(text):
                print(type(triple))
                print(triple)
                triple['text'] = text
                new_df = new_df.append(triple, ignore_index=True) 
        except AttributeError as e:
            print(e)  

    new_df.to_csv('relations.csv') 
    # with open('corpus/pg6130.txt', encoding='utf8') as r:
    #     corpus = r.read().replace('\n', ' ').replace('\r', '')

    # triples_corpus = client.annotate(corpus[0:5000])
    # print('Corpus: %s [...].' % corpus[0:80])
    # print('Found %s triples in the corpus.' % len(triples_corpus))
    # for triple in triples_corpus[:3]:
    #     print('|-', triple)
    # print('[...]')