import pandas as pd
import json

#read the parquet file
df = pd.read_parquet('/misc/student/sharmaa/coco/mscoco.parquet')
#print top 10 rows
print(df.head(10))
json_data = []
for i in range(len(df)):
    filename = df.iloc[i]['URL'].split('/')[-1]
    caption = df.iloc[i]['TEXT']
    json_data.append({"filename": filename, "caption": caption})

with open('coco_meta_new.json', 'w') as f:
    for data in json_data:
        print(data)
        json.dump(data, f)
        f.write('\n')
    f.close()