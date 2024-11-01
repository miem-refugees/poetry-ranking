import torch
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer

print("Loading models...", end="")
model_name_bert = "DeepPavlov/rubert-base-cased"
tokenizer_bert = BertTokenizer.from_pretrained(model_name_bert)
model_bert = BertModel.from_pretrained(model_name_bert)
model_bert.eval()

model_name_sbert = "ai-forever/sbert_large_nlu_ru"
tokenizer_sbert = AutoTokenizer.from_pretrained(model_name_sbert)
model_sbert = AutoModel.from_pretrained(model_name_sbert)
model_sbert.eval()
print("OK")


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_sentence_embedding_bert(sentence: str) -> torch.Tensor:
    model_bert.cuda()
    inputs = tokenizer_bert(
        sentence, return_tensors="pt", truncation=True, padding=True, max_length=128
    ).to("cuda")
    with torch.no_grad():

        outputs = model_bert(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    return embedding.cpu().numpy()


def get_sentence_embedding_sbert(sentence: str) -> torch.Tensor:
    model_sbert.cuda()
    encoded_input = tokenizer_sbert(
        sentence, padding=True, truncation=True, max_length=24, return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        model_output = model_sbert(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

    return sentence_embeddings.cpu().numpy()
