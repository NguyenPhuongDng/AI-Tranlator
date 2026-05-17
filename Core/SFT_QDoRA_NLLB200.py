#!/usr/bin/env python
# coding: utf-8

# ============================================ CẤU HÌNH ĐƯỜNG DẪN SERVER =======================================
import os
import tempfile

# Cấu hình cache
BASE_PATH = "/workspace/dongg"
os.environ["HF_HOME"] = f"{BASE_PATH}/hf_cache"
os.environ["HF_DATASETS_CACHE"] = f"{BASE_PATH}/hf_cache/datasets"
os.environ["TRANSFORMERS_CACHE"] = f"{BASE_PATH}/hf_cache/models"
os.environ["HF_HUB_CACHE"] = f"{BASE_PATH}/hf_cache/hub"
os.environ["TMPDIR"] = f"{BASE_PATH}/tmp"

os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

print("TMPDIR:", tempfile.gettempdir())

CONTINUE_TRAIN = True
VI_TO_OTHER = True
OTHER_TO_VI = True

# ================================================== IMPORT ================================================
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup
)
from datasets import load_dataset, concatenate_datasets, load_from_disk
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from peft import PeftModel, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training, PeftConfig
from peft.optimizers import create_loraplus_optimizer
from torch.utils.data import DataLoader
import bitsandbytes as bnb
import numpy as np
import gc
import random
import unicodedata

def normalize_text(text):
    return unicodedata.normalize("NFC", text)

# Ngay sau phần IMPORT, trước khi load model
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True    # Tự chọn CUDA kernel nhanh nhất
torch.set_float32_matmul_precision("high")


# Đặt đường dẫn gốc
BASE_PATH = "/workspace/dongg/NLLB_3.3B_ALT_org"
MODEL_OUTPUT_PATH = f"{BASE_PATH}/models_output" 
ADAPTER_PATH = f"{BASE_PATH}/model_lora" 
LOG_PATH = f"{BASE_PATH}/logs"
TOKENIZED_CACHE = f"{BASE_PATH}/tokenized_cache"

# Tạo thư mục
os.makedirs(BASE_PATH, exist_ok=True)
os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(ADAPTER_PATH, exist_ok=True)


# ================================================= TẢI MÔ HÌNH ==========================================
MODEL_NAME = "facebook/nllb-200-3.3B"

CHECKPOINT_PATH = "NLLB_3.3B_ALT_org/models_output/checkpoint-3000" if CONTINUE_TRAIN else "NLLB_3.3B_ALT_org/models_output/checkpoint-3000"
#CHECKPOINT_PATH = None

print(f"Loading tokenizer: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=False
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"Loading model base: {MODEL_NAME}...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    #tie_word_embeddings=True
)

model = prepare_model_for_kbit_training(model)
model.enable_input_require_grads()

# model.save_pretrained("nllb_safetensors", safe_serialization=True)
# =======================================Load Adapter cũ (nếu có) và set trainable==================================
if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
    print(f"Loading PEFT adapter from: {CHECKPOINT_PATH}...")
    model = PeftModel.from_pretrained(
        model,
        CHECKPOINT_PATH,
        is_trainable=True
    )

    for name, param in model.named_parameters():
        if "lora" in name.lower() or "modules_to_save" in name.lower():
            param.requires_grad = True
    
else:
    print("Configuring DoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        init_lora_weights="olora",
        # use_rslora=True,
        inference_mode=False,
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "out_proj",
            "fc1", "fc2"
        ],
        lora_dropout=0.05,
        use_dora=True, # <--- BẬT THUẬT TOÁN DORA TẠI ĐÂY
    )

    model = get_peft_model(model, lora_config)
    

# replace_lora_weights_loftq(model)
model.print_trainable_parameters()


# ========================================== XỬ LÝ DỮ LIỆU ==========================================

# mã ngôn ngữ
LANG_CODE_MAP = {
    'en': 'eng_Latn',
    'id': 'ind_Latn',
    'ms': 'zsm_Latn', 
    'fil': 'tgl_Latn', 
    'khm': 'khm_Khmr',
    'lo': 'lao_Laoo',
    'th': 'tha_Thai',
    'my': 'mya_Mymr',
    'vi': 'vie_Latn',
}
#
# langs = ["vi", "fil", "id", "ms", "en"]
# langs = ["vi", 'lo', "khm", "my", "th"]
langs = ["vi", 'lo', "khm", "my", "th", "fil", "id", "ms", "en"]
langs_augrument = [LANG_CODE_MAP[lang] for lang in langs]

# ================================1. Xử lý tập Augmented (Dữ liệu tăng cường)==============================
ds_augrument = load_dataset("json", data_files={
    "train": "DATA/fineTranlation/clustered_results1/news.jsonl",
    "test": "DATA/fineTranlation/test.jsonl"
})

def make_all_vi_pairs_batched_augrument(batch):
    new_src_texts = []
    new_tgt_texts = []
    src_langs = [] 
    tgt_langs = [] 

    batch_size = len(batch['source_lang'])
    vi_code = LANG_CODE_MAP['vi']
    
    for i in range(batch_size):
        # Format mới: 'source' là tiếng nước ngoài, 'target' là tiếng Việt
        other_text = batch['source'][i].strip() if isinstance(batch['source'][i], str) else ""
        vi_text = batch['target'][i].strip() if isinstance(batch['target'][i], str) else ""
        
        if not vi_text or not other_text:
            continue
            
        # Map mã ngôn ngữ từ file json (ví dụ 'id') sang mã NLLB ('ind_Latn')
        raw_source_lang = batch['source_lang'][i].strip()
        if raw_source_lang not in LANG_CODE_MAP or raw_source_lang == "vi":
            continue
            
        other_code = LANG_CODE_MAP[raw_source_lang]
        if other_code not in langs_augrument:
            continue

        # Thêm dữ liệu (Other -> Vi)
        if OTHER_TO_VI:
            new_src_texts.append(normalize_text(other_text))
            new_tgt_texts.append(normalize_text(vi_text))
            src_langs.append(other_code)
            tgt_langs.append(vi_code)

        # Thêm dữ liệu ngược lại (Vi -> Other) nếu bạn muốn train song ngữ 2 chiều
        if VI_TO_OTHER:
            new_src_texts.append(normalize_text(vi_text))
            new_tgt_texts.append(normalize_text(other_text))
            src_langs.append(vi_code)
            tgt_langs.append(other_code)

    return {
        "src_text": new_src_texts,
        "tgt_text": new_tgt_texts,
        "src_lang": src_langs, 
        "tgt_lang": tgt_langs  
    }


ds_augrument = ds_augrument.map(
    make_all_vi_pairs_batched_augrument,
    batched=True,            
    batch_size=1000,
    remove_columns=ds_augrument["train"].column_names, 
    num_proc=8,
    desc="🔄 Processing Augmented pairs"
)




# =======================================2. Xử lý tập Train chính (ALT Dataset)========================================
ds = load_dataset("json", data_files={
    "train": "DATA/ALT/alt_multilingual_dataset/train.jsonl",
    "test": "DATA/ALT/alt_multilingual_dataset/test.jsonl"
})

def make_all_vi_pairs_batched(batch):
    new_src_texts, new_tgt_texts = [], []
    src_langs, tgt_langs = [], [] # Thêm 2 mảng lưu mã ngôn ngữ

    vi_code = LANG_CODE_MAP['vi']
    bsz = len(batch['vi'])

    for i in range(bsz):
        vi_text = batch['vi'][i].strip() if isinstance(batch['vi'][i], str) else ""
        if not vi_text: continue

        for lang in langs:
            if lang == 'vi': continue
            other_text = batch[lang][i].strip() if isinstance(batch[lang][i], str) else ""
            if not other_text: continue
            
            other_code = LANG_CODE_MAP[lang]

            # vi -> other
            if VI_TO_OTHER:
                new_src_texts.append(normalize_text(vi_text))
                new_tgt_texts.append(normalize_text(other_text))
                src_langs.append(vi_code)
                tgt_langs.append(other_code)

            # other -> vi
            if OTHER_TO_VI:
                new_src_texts.append(normalize_text(other_text))
                new_tgt_texts.append(normalize_text(vi_text))
                src_langs.append(other_code)
                tgt_langs.append(vi_code)

    return {
        "src_text": new_src_texts,
        "tgt_text": new_tgt_texts,
        "src_lang": src_langs, # Cột mới
        "tgt_lang": tgt_langs  # Cột mới
    }

ds = ds.map(
    make_all_vi_pairs_batched,
    batched=True,             
    batch_size=1000,
    remove_columns=ds["train"].column_names, 
    num_proc=8,
    desc="🔄 Creating ALL vi <-> others pairs"
)

# ============================================= MAXLENGTH ==================================================

MAX_LENGTH = 256

# ============================================= TOKENIZE ==================================================

def preprocess_function(examples):
    # Khai báo không giới hạn độ dài ở bước này để đếm độ dài thực tế
    src_encoded = tokenizer(
        examples["src_text"],
        padding=False,
        add_special_tokens=False
    )

    tgt_encoded = tokenizer(
        text_target=examples["tgt_text"],
        padding=False,
        add_special_tokens=False
    )

    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    
    eos_id = tokenizer.eos_token_id 

    for i in range(len(examples["src_text"])):
        src_tokens = src_encoded["input_ids"][i]
        tgt_tokens = tgt_encoded["input_ids"][i]

        # NẾU CÂU QUÁ DÀI -> BỎ QUA (Không cắt ngang để tránh hỏng logic [EOS])
        # Trừ 2 vì cần chừa chỗ cho [LANG_ID] và [EOS]
        if len(src_tokens) > MAX_LENGTH - 2 or len(tgt_tokens) > MAX_LENGTH - 2:
            # Gán rỗng để lát nữa dùng .filter() lọc đi
            all_input_ids.append([])
            all_attention_mask.append([])
            all_labels.append([])
            continue

        src_lang_id = tokenizer.convert_tokens_to_ids(examples["src_lang"][i])
        tgt_lang_id = tokenizer.convert_tokens_to_ids(examples["tgt_lang"][i])

        input_ids = [src_lang_id] + src_tokens + [eos_id]
        attention_mask = [1] * len(input_ids)
        
        labels = [tgt_lang_id] + tgt_tokens + [eos_id]

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }


import random

def debug_nllb_sample(tokenized_dataset, raw_dataset, tokenizer, split="train"):
    print("\n🔍 DEBUG NLLB SAMPLE\n")

    idx = random.randint(0, len(tokenized_dataset[split]) - 1)

    tok = tokenized_dataset[split][idx]
    raw = raw_dataset[split][idx]

    print(f"📌 Index: {idx}")
    print("-" * 50)

    print("📝 SRC:", raw["src_text"])
    print("📝 TGT:", raw["tgt_text"])
    print("🌐 SRC_LANG:", raw["src_lang"])
    print("🌐 TGT_LANG:", raw["tgt_lang"])
    print("-" * 50)

    # ⚠️ QUAN TRỌNG: set lại lang trước khi decode
    tokenizer.src_lang = raw["src_lang"]

    decoded_src = tokenizer.decode(
        tok["input_ids"],
        skip_special_tokens=False
    )

    # Với target → phải set tgt_lang
    tokenizer.tgt_lang = raw["tgt_lang"]

    decoded_tgt = tokenizer.decode(
        tok["labels"],
        skip_special_tokens=False
    )

    print("🔁 Decoded SRC:", decoded_src)
    print("🔁 Decoded TGT:", decoded_tgt)
    print("-" * 50)

    print("⚖️ SRC:", "OK" if decoded_src.strip() == raw["src_text"].strip() else "⚠️ MISMATCH")
    print("⚖️ TGT:", "OK" if decoded_tgt.strip() == raw["tgt_text"].strip() else "⚠️ MISMATCH")

    print("-" * 50)
    print("🔢 First tokens SRC:", tok["input_ids"])
    print("🔢 First tokens TGT:", tok["labels"])


print("🔡 Tokenizing datasets...")

if os.path.exists(TOKENIZED_CACHE):
    tokenized_datasets = load_from_disk(f"{TOKENIZED_CACHE}/main")
    tokenized_datasets_augrument = load_from_disk(f"{TOKENIZED_CACHE}/augment")
else:
    tokenized_datasets = ds.map(
        preprocess_function,
        batched=True,
        batch_size=2000,
        remove_columns=ds["train"].column_names,
        num_proc=8,
        desc="Tokenizing Main"
    ).filter(lambda x: len(x["input_ids"]) > 0)

    tokenized_datasets_augrument = ds_augrument.map(
        preprocess_function,
        batched=True,
        batch_size=2000,
        remove_columns=ds_augrument["train"].column_names,
        num_proc=8,
        desc="Tokenizing Augment"
    ).filter(lambda x: len(x["input_ids"]) > 0)

    tokenized_datasets.save_to_disk(f"{TOKENIZED_CACHE}/main")
    tokenized_datasets_augrument.save_to_disk(f"{TOKENIZED_CACHE}/augment")



#CHECK
debug_nllb_sample(tokenized_datasets, ds, tokenizer)
debug_nllb_sample(tokenized_datasets_augrument, ds_augrument, tokenizer)

# ==============================================Gộp dataset====================================
combined_train_dataset = concatenate_datasets([
    tokenized_datasets["train"],
    tokenized_datasets_augrument["train"]
    #tokenized_datasets["test"]
])
combined_eval_dataset = concatenate_datasets([
    tokenized_datasets["test"],
    #tokenized_datasets_augrument["test"]
])


# trộn data
combined_train_dataset = combined_train_dataset.shuffle(seed=42)
print(f"Tổng mẫu huấn luyện: {len(combined_train_dataset)}")
print(f"Tổng mẫu kiểm thử: {len(combined_eval_dataset)}")


# =================================================== TRAINING SETUP ==================================================
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8,
    return_tensors="pt",
    label_pad_token_id=-100
)

#===================================================TEST DATALOALA====================================================
# # Lấy 2 mẫu đầu tiên từ tập dataset để test
# samples = [combined_train_dataset[0], combined_train_dataset[1]]

# # Đưa qua data_collator (thêm .to('cpu') nếu cần)
# batch = data_collator(samples)

# print("="*60)
# print("🔍 KIỂM TRA KÍCH THƯỚC (SHAPE) CỦA BATCH")
# print("="*60)
# for k, v in batch.items():
#     print(f"{k:20}: {v.shape}")

# print("\n" + "="*60)True
# print("🧐 KIỂM TRA CHI TIẾT TỪNG TOKEN (MẪU SỐ 1)")
# print("="*60)

# idx = 0 # Kiểm tra mẫu đầu tiên trong batch

# # --- 1. KIỂM TRA ENCODER INPUT ---
# input_ids = batch["input_ids"][idx]
# print("\n🟢 1. INPUT_IDS (Đầu vào của Encoder):")
# print(f"   - Tensor nguyên bản : {input_ids.tolist()}")
# print(f"   - Token dạng chữ    : {tokenizer.convert_ids_to_tokens(input_ids)}")
# print(f"   - Dịch ra văn bản   : {tokenizer.decode(input_ids, skip_special_tokens=False)}")

# # --- 2. KIỂM TRA LABELS ---
# labels = batch["labels"][idx]
# # Lọc bỏ các token -100 (đây là token yêu cầu hàm loss bỏ qua khi padding)
# valid_labels = labels[labels != -100]

# print("\n🔴 2. LABELS (Đầu ra mục tiêu để tính Loss):")
# print(f"   - Tensor nguyên bản : {valid_labels.tolist()} (đã bỏ qua các số -100)")
# print(f"   - Token dạng chữ    : {tokenizer.convert_ids_to_tokens(valid_labels)}")
# print(f"   - Dịch ra văn bản   : {tokenizer.decode(valid_labels, skip_special_tokens=False)}")
# print(f"   - Có chứa -100 ở cuối không?: {'Có' if -100 in labels else 'Không'}")

# # --- 3. KIỂM TRA DECODER INPUT ---
# if "decoder_input_ids" in batch:
#     dec_input_ids = batch["decoder_input_ids"][idx]
#     # Lọc bỏ token padding (thường NLLB dùng token id 1 cho padding)
#     valid_dec_inputs = dec_input_ids[dec_input_ids != tokenizer.pad_token_id]
    
#     print("\n🔵 3. DECODER_INPUT_IDS (Đầu vào của Decoder do HF tự sinh ra):")
#     print(f"   - Tensor nguyên bản : {valid_dec_inputs.tolist()}")
#     print(f"   - Token dạng chữ    : {tokenizer.convert_ids_to_tokens(valid_dec_inputs)}")
#     print(f"   - Dịch ra văn bản   : {tokenizer.decode(valid_dec_inputs, skip_special_tokens=False)}")
# else:
#     print("\n🔵 3. DECODER_INPUT_IDS:")
#     print("   ⚠️ Không tìm thấy khóa này trong batch! Models của Hugging Face thường sẽ tự động 'shift_right' từ `labels` bên trong hàm forward() nếu bạn không truyền `decoder_input_ids`.")



#=======================================================training config=============================================

BS = 16
ACCUM = 4
EPOCHS = 5
LR = 1e-4
STEP = 1000
LOG = 100

training_args = Seq2SeqTrainingArguments(
    output_dir=MODEL_OUTPUT_PATH, 
    per_device_train_batch_size=BS,
    per_device_eval_batch_size=8,  
    gradient_accumulation_steps=ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    
    # Optimization
    lr_scheduler_type="linear",
    #warmup_ratio=0.1,
    warmup_steps=500,
    weight_decay=0.01,
    adam_epsilon=1e-8,
    
    # Evaluation & Saving
    eval_strategy="steps",  
    eval_steps=STEP,
    logging_strategy="steps",
    logging_steps=LOG,
    logging_dir=LOG_PATH,  
    
    save_strategy="steps",
    save_steps=STEP,
    save_total_limit=3,  
    
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    #label_smoothing_factor=0.1,
    
    # Mixed Precision 
    bf16=True, 
    fp16=False,
    tf32=True,
    
    # Performance optimizations cho server
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_pin_memory=True,
    dataloader_num_workers=8,  
    dataloader_drop_last=True,
    dataloader_prefetch_factor=4,
    full_determinism=False,
    
    # Reporting
    report_to="none",
    predict_with_generate=False,
    generation_max_length=MAX_LENGTH,
    
    # Important configs
    remove_unused_columns=True,
    label_names=["labels"],
    max_grad_norm=1.0
)

# === THÊM ĐOẠN NÀY ĐỂ TẠO LORA+ OPTIMIZER VÀ SCHEDULER ===
# optimizer = create_loraplus_optimizer(
#     model=model,
#     optimizer_cls=torch.optim.AdamW,
#     lr=LR,
#     loraplus_lr_ratio=16,
#     weight_decay=0.01,
#     # Các tham số khác của AdamW
#     eps=1e-8,
#     betas=(0.9, 0.999)
# )


# scheduler = ReduceLROnPlateau(
#     optimizer,
#     mode="min",
#     factor=0.5,
#     patience=2
# )
# =================================================== TRAINER & EXECUTION =============================================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=combined_train_dataset,
    eval_dataset=combined_eval_dataset,
    data_collator=data_collator,
    processing_class=tokenizer, 
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    # optimizers=(optimizer, None)
)

print("=== System Information ===")
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

print("🚀 Bắt đầu training...")
torch.cuda.empty_cache()
model.enable_input_require_grads()

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
# model = torch.compile(model)
if CONTINUE_TRAIN:
    trainer.train(resume_from_checkpoint=CHECKPOINT_PATH)
else:
    trainer.train()

# ============================================== Đánh giá ===================================================
import torch
import evaluate

bleu = evaluate.load("sacrebleu")

def test_overfit_nllb_prefix(model, tokenizer, dataset, max_length=256, batch_size=16):
    model.eval()
    preds, refs = [], []

    # Xử lý theo batch thay vì từng sample
    for start in range(0, len(dataset), batch_size):
        batch_samples = dataset[start : start + batch_size]
        
        input_ids_list = batch_samples["input_ids"]
        labels_list    = batch_samples["labels"]

        # Pad thủ công để stack
        max_src_len = max(len(x) for x in input_ids_list)
        padded = torch.zeros(len(input_ids_list), max_src_len, dtype=torch.long)
        attn   = torch.zeros_like(padded)
        for i, ids in enumerate(input_ids_list):
            padded[i, :len(ids)] = torch.tensor(ids)
            attn[i, :len(ids)]   = 1

        target_lang_ids = [labels[0] for labels in labels_list]
        
        # SỬA Ở ĐÂY: Thêm decoder_start_token_id (thường là 2) vào trước LANG_ID
        decoder_start_id = model.config.decoder_start_token_id # hoặc tokenizer.eos_token_id
        
        decoder_input_ids = torch.tensor(
            [[decoder_start_id, lang_id] for lang_id in target_lang_ids], 
            dtype=torch.long
        ).to(model.device)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.generate(
                input_ids=padded.to(model.device),
                attention_mask=attn.to(model.device),
                max_new_tokens=max_length,
                decoder_input_ids=decoder_input_ids, # Mồi đúng chuẩn của NLLB
            )

        for i, (out, labels) in enumerate(zip(outputs, labels_list)):
            label_ids = [x for x in labels if x != -100]
            
            # Decode và dọn dẹp chuỗi
            pred_text = tokenizer.decode(out, skip_special_tokens=True).strip()
            ref_text = tokenizer.decode(label_ids, skip_special_tokens=True).strip()
            
            preds.append(pred_text)
            refs.append([ref_text])
            
            # In thử 1 sample đầu tiên của batch đầu tiên để kiểm chứng
            if start == 0 and i == 0:
                print("\n" + "="*50)
                print(f"🎯 Reference : {ref_text}")
                print(f"🤖 Prediction: {pred_text}")
                print("="*50 + "\n")

    bleu_score = bleu.compute(predictions=preds, references=refs)
    print("\nBLEU SCORE:", bleu_score["score"])


#Lấy vài mẫu để test
indices = random.sample(range(len(tokenized_datasets["test"])), 1000)
combined_eval_dataset_test = tokenized_datasets["test"].select(indices)

test_overfit_nllb_prefix(model, tokenizer, combined_eval_dataset_test)

# =================================================== SAVE ===================================================
print(f"💾 Saving model to {ADAPTER_PATH}...")
model.save_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(ADAPTER_PATH)

print("✅ Hoàn tất!")