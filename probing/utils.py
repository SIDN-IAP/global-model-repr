import torch
import os
import numpy as np
from transformers import XLNetTokenizer, XLNetModel, GPT2Tokenizer, GPT2Model, XLMTokenizer, XLMModel, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, DistilBertTokenizer, DistilBertModel

DATA_DIR = "data"
UD_EN_PREF = "en-ud-"

def get_model_and_tokenizer(model_name, device, random_weights=False):

    model_name = model_name

    if model_name.startswith('xlnet'):
        model = XLNetModel.from_pretrained(model_name, output_hidden_states=True).to(device)
        tokenizer = XLNetTokenizer.from_pretrained(model_name)
        sep = u'▁'
        emb_dim = 1024 if "large" in model_name else 768        
    elif model_name.startswith('gpt2'):
        model = GPT2Model.from_pretrained(model_name, output_hidden_states=True).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        sep = 'Ġ'
        sizes = {"gpt2": 768, "gpt2-medium": 1024, "gpt2-large": 1280, "gpt2-xl": 1600}
        emb_dim = sizes[model_name]
    elif model_name.startswith('xlm'):
        model = XLMModel.from_pretrained(model_name, output_hidden_states=True).to(device)
        tokenizer = XLMTokenizer.from_pretrained(model_name)
        sep = '</w>'
    elif model_name.startswith('bert'):
        model = BertModel.from_pretrained(model_name, output_hidden_states=True).to(device)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        sep = '##'
        emb_dim = 1024 if "large" in model_name else 768
    elif model_name.startswith('distilbert'):
        model = DistilBertModel.from_pretrained(model_name, output_hidden_states=True).to(device)
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        sep = '##'    
        emb_dim = 768
    elif model_name.startswith('roberta'):
        model = RobertaModel.from_pretrained(model_name, output_hidden_states=True).to(device)
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        sep = 'Ġ'        
        emb_dim = 1024 if "large" in model_name else 768
    else:
        print('Unrecognized model name:', model_name)
        sys.exit()

    if random_weights:
        print('Randomizing weights')
        model.init_weights()

    return model, tokenizer, sep, emb_dim

# this follows the HuggingFace API for pytorch-transformers
def get_sentence_repr(sentence, model, tokenizer, sep, model_name, device):
    """
    Get representations for one sentence
    """

    with torch.no_grad():
        ids = tokenizer.encode(sentence)
        input_ids = torch.tensor([ids]).to(device)
        # Hugging Face format: list of torch.FloatTensor of shape (batch_size, sequence_length, hidden_size) (hidden_states at output of each layer plus initial embedding outputs)
        all_hidden_states = model(input_ids)[-1]
        # convert to format required for contexteval: numpy array of shape (num_layers, sequence_length, representation_dim)
        all_hidden_states = [hidden_states[0].cpu().numpy() for hidden_states in all_hidden_states]
        all_hidden_states = np.array(all_hidden_states)

    #For each word, take the representation of its last sub-word
    segmented_tokens = tokenizer.convert_ids_to_tokens(ids)
    assert len(segmented_tokens) == all_hidden_states.shape[1], 'incompatible tokens and states'
    mask = np.full(len(segmented_tokens), False)

    if model_name.startswith('gpt2') or model_name.startswith('xlnet') or model_name.startswith('roberta'):
        # if next token is a new word, take current token's representation
        #print(segmented_tokens)
        for i in range(len(segmented_tokens)-1):
            if segmented_tokens[i+1].startswith(sep):
                #print(i)
                mask[i] = True
        # always take the last token representation for the last word
        mask[-1] = True
    # example: ['jim</w>', 'henson</w>', 'was</w>', 'a</w>', 'pup', 'pe', 'teer</w>']
    elif model_name.startswith('xlm'):
        # if current token is a new word, take it
        for i in range(len(segmented_tokens)):
            if segmented_tokens[i].endswith(sep):
                mask[i] = True
        mask[-1] = True
    elif model_name.startswith('bert') or model_name.startswith('distilbert'):
        # if next token is not a continuation, take current token's representation
        for i in range(len(segmented_tokens)-1):
            if not segmented_tokens[i+1].startswith(sep):
                mask[i] = True
        mask[-1] = True
    else:
        print('Unrecognized model name:', model_name)
        sys.exit()

    all_hidden_states = all_hidden_states[:, mask]
    # all_hidden_states = torch.tensor(all_hidden_states).to(device)

    return all_hidden_states


def get_pos_data(probing_dir, frac=1.0, device='cpu'):

    return get_data("pos", probing_dir=probing_dir, frac=frac, device=device)


def get_data(data_type, probing_dir, data_pref=UD_EN_PREF, frac=1.0, device='cpu'):

    with open(os.path.join(probing_dir, DATA_DIR, data_pref + "train.txt")) as f:
        train_sentences = [line.strip().split() for line in f.readlines()]
    with open(os.path.join(probing_dir, DATA_DIR, data_pref + "test.txt")) as f:
        test_sentences = [line.strip().split() for line in f.readlines()]
    with open(os.path.join(probing_dir, DATA_DIR, data_pref + "dev.txt")) as f:
        dev_sentences = [line.strip().split() for line in f.readlines()]

    with open(os.path.join(probing_dir, DATA_DIR, data_pref + "train." + data_type)) as f:
        train_labels = [line.strip().split() for line in f.readlines()]
    with open(os.path.join(probing_dir, DATA_DIR, data_pref + "test." + data_type)) as f:
        test_labels = [line.strip().split() for line in f.readlines()]
    with open(os.path.join(probing_dir, DATA_DIR, data_pref + "dev." + data_type)) as f:
        dev_labels = [line.strip().split() for line in f.readlines()]

    # take a fraction of the data
    train_sentences = train_sentences[:round(len(train_sentences)*frac)]
    test_sentences = train_sentences[:round(len(test_sentences)*frac)]
    dev_sentences = train_sentences[:round(len(dev_sentences)*frac)]
    train_labels = train_labels[:round(len(train_labels)*frac)]
    test_labels = train_labels[:round(len(test_labels)*frac)]
    dev_labels = train_labels[:round(len(dev_labels)*frac)]

    unique_labels = list(set.union(*[set(l) for l in train_labels + test_labels + dev_labels]))
    label2index = dict()
    for label in unique_labels:
        label2index[label] = label2index.get(label, len(label2index))

    train_labels = [[label2index[l] for l in labels] for labels in train_labels]
    test_labels = [[label2index[l] for l in labels] for labels in test_labels]
    dev_labels = [[label2index[l] for l in labels] for labels in dev_labels]


    return train_sentences, train_labels, test_sentences, test_labels, dev_sentences, dev_labels, label2index



