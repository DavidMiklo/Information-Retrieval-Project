import os 
 
def preprocess_abstracts(lines):
    abstracts = []
    current_id = None
    current_abstract = []

    for line in lines:
        if line.isspace():
            continue
        if line.startswith('###'):
            if current_id is not None:
                abstracts.append({'id': current_id, 'abstract': ' '.join(current_abstract)})
            current_id = line[3:].strip()
            current_abstract = []
        else:
            line_content = line.split('\t')[1].strip() if '\t' in line else line.strip()
            current_abstract.append(line_content)
    
    if current_id is not None:
        abstracts.append({'id': current_id, 'abstract': ' '.join(current_abstract)})

    return abstracts

def read_lines(filename):
    with open(os.path.join(path, filename), 'r') as f:
        return f.readlines()

path = 'data/'

train = read_lines('train.txt')
abstracts = preprocess_abstracts(train)

abstract_texts = [item['abstract'] for item in abstracts]
abstract_ids = [item['id'] for item in abstracts]

ground_truth = {
    14519753, 18272913, 18757324, 19731015,
    21336679, 21481449, 21439044, 22112544,
    22907422, 23790994, 24136693, 24361787,
    24609919, 24804802, 25139726, 24934783, 26077235
}

query = "trial and obesity and cancer and not menopausal and not postmenopausal and not men"
query2 = "Trials linking obesity to cancer in not menopausal and not postmenopausal women"

    
def evaluate(results, ground_truth):
    retrieved_ids = {int(result['id']) for result in results}  
    
    ground_truth = set(ground_truth)
    
    true_positives = len(retrieved_ids & ground_truth)
    false_positives = len(retrieved_ids - ground_truth)
    false_negatives = len(ground_truth - retrieved_ids)

    precision = true_positives / (true_positives + false_positives) if \
                                 (true_positives + false_positives) > 0 else 0.0

    recall = true_positives / (true_positives + false_negatives) if \
                              (true_positives + false_negatives) > 0 else 0.0

    return precision, recall