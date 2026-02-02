import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import os

from common.focal_loss import FocalLoss
from common.dataset import *
from common.utils import *
from common.load_transformers import LoadTransformers
from common.preprocessor import Preprocessor
from common.torch_trainer import TorchTrainer
from models_architecture.TransformerWithExtraFeatures import TransformerWithExtraFeatures

# ===================== LOGGING =====================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
file_handler = logging.FileHandler('xlmr_model.log')
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def print_log(*args):
    for arg in args:
        print(arg)
        logger.info(arg)

# ===================== DEVICE =====================
print_log("-" * 20)
if torch.cuda.is_available():
    device = torch.device('cuda')
    print_log("Device is cuda.\n")
else:
    device = torch.device('cpu')
    print_log("Device is cpu.\n")

# ===================== STRATIFIED TRAIN SUBSET =====================
FULL_TRAIN_PATH = "data/b2c/train_set_final.xlsx"
SUBSET_TRAIN_PATH = "data/b2c/train_set_subset_100.xlsx"

if not os.path.exists(SUBSET_TRAIN_PATH):
    print_log("Creating stratified 100-sample training subset...")

    df = pd.read_excel(FULL_TRAIN_PATH)
    assert "stance" in df.columns, "Column 'stance' not found"

    subset_df, _ = train_test_split(
        df,
        train_size=100,
        stratify=df["stance"],
        random_state=42
    )

    subset_df.to_excel(SUBSET_TRAIN_PATH, index=False)

    print_log("Subset saved:", SUBSET_TRAIN_PATH)
    print_log("Subset label distribution:")
    print_log(subset_df["stance"].value_counts())
else:
    print_log("Using existing subset:", SUBSET_TRAIN_PATH)

# ===================== DATASET =====================
ner_model_id = 'HooshvareLab/bert-base-parsbert-ner-uncased'

dataset_labels = {
    'claim_name': "claim",
    'text_name': "content",
    'label_name': "stance"
}

dataset_params = {
    'padding': True,
    'truncation': True,
    'max_length': 256,
    'claim_max_length': 38,
    'device': device,
    'similarity_features_no_for_start': 5,
    'similarity_features_no_for_end': 2,
    'ner_features_no': 8,
    'remove_dissimilar_sentences': True,
    'similar_sentence_no': 8,
    'ner_model_path': ner_model_id,
    'ner_tokenizer_path': ner_model_id,
    'ner_none_token_id': 0,
    'sentence_similarity_model_path': "sentence-transformers/all-MiniLM-L12-v2",
    'log_function': print_log,
    'save_load_features_path': 'data/features',
    'remove_news_agency_name': False,
    'news_agency_name_path': 'data/news_agency_names.xlsx',
    'news_agency_coloumn_name': 'name'
}

# ===================== TRANSFORMER =====================
transformer_params = {
    'model_loader': LoadTransformers.xlm_roberta,
    'transformer_model_path': "FacebookAI/xlm-roberta-base"
}

vectorizer = AutoTokenizer.from_pretrained(transformer_params["transformer_model_path"])
transformer_params["vectorizer"] = vectorizer

# ===================== TRAINING PARAMS (UNCHANGED) =====================
batch_size = 16
accumulation_steps = 1
num_epochs = 5
patience = 3

# ===================== TRAIN SET (SUBSET) =====================
train_set_path = SUBSET_TRAIN_PATH

train_init = DatasetInitModel(
    data_set_path=train_set_path,
    preprocessor=Preprocessor(),
    **dataset_labels,
    **dataset_params,
    **transformer_params
)

train_dataset = StanceDataset(train_init)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print_log("train_dataloader length =", len(train_dataset))

# ===================== VALIDATION SET (UNCHANGED) =====================
val_set_path = "data/b2c/dev_set_final.xlsx"

val_init = DatasetInitModel(
    data_set_path=val_set_path,
    preprocessor=Preprocessor(),
    **dataset_labels,
    **dataset_params,
    **transformer_params
)

val_dataset = StanceDataset(val_init)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print_log("val_dataloader length =", len(val_dataset))

# ===================== MODEL =====================
extra_features_no = (
    dataset_params["similarity_features_no_for_start"]
    + dataset_params["similarity_features_no_for_end"]
    + dataset_params["ner_features_no"]
)

first_dense_input_size = 768 + extra_features_no

model = TransformerWithExtraFeatures(
    transformer_params['model_loader'],
    transformer_params['transformer_model_path'],
    dens_input_size=[first_dense_input_size, 128, 32],
    extra_features_no= extra_features_no,
    num_classes=4,
    dens_dropout=[0.4, 0.3]
)

model.to(device)

# ===================== OPTIMIZER =====================
loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(
#     model.parameters(),
#     lr=3e-5,
#     weight_decay=0.8e-5
# )

### improve
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=3e-3,
    momentum=0.9,
    weight_decay=0.8e-5
)

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[2, 4],
    gamma=0.1
)

trainer = TorchTrainer(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    loss_fn,
    device,
    accumulation_steps,
    print_log,
    scheduler=scheduler
)
###

# ===================== TRAIN =====================
model_save_directory = "models_sgd/"
loss_log = trainer.fit(num_epochs, patience, model_save_directory)
