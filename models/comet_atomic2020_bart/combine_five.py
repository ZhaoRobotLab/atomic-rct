import pandas as pd
import os

src_dir = 'results'
target_dir = os.path.join('results', 'combined')

files = os.listdir(src_dir)
for file in files:
  if not file.endswith('.csv'):
    continue

  df = pd.read_csv(os.path.join(src_dir, file))

  c = 0
  curr_text = ''
  start_round = 0
  end_round = 0
  curr_res_dict_raw = {}
  curr_res_dict_fixed = {}
  new_df = pd.DataFrame(columns=['text', 'start_round', 'end_round', 'res_dict_raw', 'res_dict_fixed'])
  for indx, row in df.iterrows():
    if c % 5 == 0 and c != 0:
      new_df = pd.concat([new_df, pd.DataFrame.from_dict({
        'text': curr_text,
        'start_round': start_round,
        'end_round': c-1,
        'res_dict_raw': [curr_res_dict_raw],
        'res_dict_fixed': [curr_res_dict_fixed],  
        })], ignore_index=True)

      curr_text = ''
      start_round = c
      end_round = c+4
      curr_res_dict_raw = {}
      curr_res_dict_fixed = {}

    curr_text += row.text + ' '
    curr_res_dict_raw = {**curr_res_dict_raw, **eval(row.res_dict_raw)}
    curr_res_dict_fixed = {**curr_res_dict_fixed, **eval(row.res_dict_fixed)}

    c += 1

  print(new_df)
  new_df.to_csv(os.path.join(target_dir, file))