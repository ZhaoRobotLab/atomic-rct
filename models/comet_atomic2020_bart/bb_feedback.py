from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration
import os 
import pandas as pd

datadir = os.path.join('Manual_Trans', 'Manual_Trans')
testdir = os.path.join('testdir')
target_dir = os.path.join('blenderbot_feedbacks')
files = os.listdir(datadir)
after_participant = "</s>"
before_bot = "<s>"

# model
mname = "facebook/blenderbot_small-90M"
model = BlenderbotSmallForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)

for file in files:
    if not file.endswith('.csv'):
        continue
    csv_name = os.path.join(os.path.join(datadir, file))
    df = pd.read_csv(csv_name)

    # get rid of weird floats
    df.dropna(inplace=True)

    # group by role
    grouped = df.groupby(df.role.ne(df.role.shift()).cumsum(), as_index=False).agg({'role': 'first', 'text': ' '.join})
    print(grouped)

    # create conv history
    conv_hist = ""
    new_df = pd.DataFrame(columns=["participant", "moderator", "generated"])
    for indx in range(0, len(grouped)-1, 2):
        conv_hist += grouped.iloc[indx].text + tokenizer.eos_token

        # generate responses and store
        # help from this https://huggingface.co/blog/how-to-generate
        inputs = tokenizer([grouped.iloc[indx].text + tokenizer.eos_token], return_tensors="pt")
        reply_ids = model.generate(**inputs, 
                do_sample=True,
                top_k=50,
                top_p=0.95,
                num_return_sequences=3)
        print(grouped.iloc[indx].text)
        responses = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)
        print(responses[0])
        
        conv_hist += "{}\n".format(grouped.iloc[indx+1].text)
        print(conv_hist)

        # make new row in df
        new_df.loc[len(new_df.index)] = [grouped.iloc[indx].text, grouped.iloc[indx+1].text, responses]

    # save
    new_df.to_csv(os.path.join(target_dir, file))


# UTTERANCE = "My friends are cool but they eat too many carbs."
# print("Participant: ", UTTERANCE)


# print("Moderator: ", tokenizer.batch_decode(reply_ids, skip_special_tokens=True))

# >>> REPLY = "I'm not sure"
# >>> print("Human: ", REPLY)
# Human: I'm not sure

# >>> NEXT_UTTERANCE = (
# ...     "My friends are cool but they eat too many carbs.</s> <s>what kind of carbs do they eat? "
# ...     "i don't know much about carbs</s> "
# ...     "<s> I'm not sure."
# ... )
# >>> inputs = tokenizer([NEXT_UTTERANCE], return_tensors="pt")
# >>> next_reply_ids = model.generate(**inputs)
# >>> print("Bot: ", tokenizer.batch_decode(next_reply_ids, skip_special_tokens=True)[0])
# Bot:  they eat a lot of carbs. carbs are high in fat, protein, and carbohydrates.