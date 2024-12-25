import pandas as pd
import json

train_df = pd.read_csv('faq_first_dog.csv')
#converting dataset to JSONL format

output_file = 'training.jsonl'
with open(output_file, 'w') as f:
    for _, row in train_df.iterrows():
        #create json lines for chat model fine tuning
        json_line = json.dumps({
            "messages": [
                {"role": "system", "content": "You are a helpful assistant which acts as FAQ Support Assistant for getting your first dog and answer to user queries."},
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["completion"]}
            ]
        })
        f.write(json_line + '\n')

# output_file = 'validation.jsonl'
# with open(output_file, 'w') as f:
#     for _, row in valid_df.iterrows():
#         # Create JSON lines for chat model fine tuning
#         json_line = json.dumps({
#             "messages": [
#                 {"role": "system", "content": "You are a helpful assistant which acts as FAQ Support Assistant for getting your first dog and answer to user queries."},
#                 {"role": "user", "content": row['prompt']},
#                 {"role": "assistant", "content": row['completion']}
#             ]
#         })
#         f.write(json_line + '\n')

print(f"Dataset converted and saved to {output_file}")