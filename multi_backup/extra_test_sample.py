from datasets import load_dataset
import json

def load_json(path):
    f = open(path, 'r', encoding = 'utf-8')
    data = json.load(f)
    f.close()

    return data

def json_save(data, path):
    f = open(path, 'w', encoding = 'utf-8')
    json.dump(data, f)
    f.close()


ds = load_dataset("/map-vepfs/siwei/coig/hf/BioLaySumm2025-eLife")['train']

# print(ds)


# ds = load_dataset("/map-vepfs/siwei/coig/hf/BioLaySumm2025-PLOS")

data = []
for item in ds.select(range(100)):
    # print(item.keys())
    article = item['article']
    reference = item['summary']
    generated_caption = item['summary']

    # print('test')
    # print(reference)
    # input()

    new_item = {
        'reference': reference,
        'generated_caption': generated_caption,
        'document': article,
    }

    data.append(new_item)

    

json_save(data, './BioLaySumm2025-eLife_result.json')
