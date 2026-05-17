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


# ================================================== IMPORT ================================================
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)
from datasets import load_dataset, concatenate_datasets, load_from_disk
import torch
import torch.nn.functional as F
from peft import PeftModel, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import numpy as np
import gc
import random
import unicodedata
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import evaluate

def normalize_text(text):
    return unicodedata.normalize("NFC", text)


# Ngay sau phần IMPORT, trước khi load model
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True    # Tự chọn CUDA kernel nhanh nhất
torch.set_float32_matmul_precision("high")


# Đặt đường dẫn gốc
BASE_PATH = "/workspace/dongg/NLLB_3.3B_ALT_SimPO_CPO_v2"
MODEL_OUTPUT_PATH = f"{BASE_PATH}/models_output"
ADAPTER_PATH = f"{BASE_PATH}/model_lora"
LOG_PATH = f"{BASE_PATH}/logs"
TOKENIZED_CACHE = f"{BASE_PATH}/tokenized_cache"
CONTINUE_TRAIN = True

os.makedirs(BASE_PATH, exist_ok=True)
os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(ADAPTER_PATH, exist_ok=True)


# ================================================= TẢI MÔ HÌNH ==========================================
MODEL_NAME = "facebook/nllb-200-3.3B"

CHECKPOINT_PATH = "NLLB_3.3B_ALT_SimPO_CPO_v2/models_output/checkpoint-3000" if CONTINUE_TRAIN else "NLLB_3.3B_ALT_SimPO_CPO_v2/models_output/checkpoint-3000"

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
)

model = prepare_model_for_kbit_training(model)
model.enable_input_require_grads()

if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
    print(f"Loading PEFT adapter from: {CHECKPOINT_PATH}...")
    model = PeftModel.from_pretrained(model, CHECKPOINT_PATH, is_trainable=True)
    for name, param in model.named_parameters():
        if "lora" in name.lower() or "modules_to_save" in name.lower():
            param.requires_grad = True
else:
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        init_lora_weights="olora",
        use_rslora=True,
        inference_mode=False,
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "out_proj",
            "fc1", "fc2",
        ],
        lora_dropout=0.05,
        use_dora=False,
    )
    model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

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
langs = ["vi", 'lo', "khm", "my", "th", "fil", "id", "ms", "en"]
langs_augrument = [LANG_CODE_MAP[lang] for lang in langs]

MAX_LENGTH = 64


# ============================= 1. Dataset processing (chosen + rejected) ================================
# Raw dataset schema: { source, source_lang, chosen, rejected }
# chosen and rejected are both target-language (vi) translations of source.
# We create bidirectional pairs: (other→vi) and (vi→other).
# For the vi→other direction, we use chosen/rejected as the source, and the
# original source text as the single reference — so we only add that as a
# chosen=rejected pair (SFT-style) to avoid fabricating a bad translation.

def make_all_vi_pairs_batched_augrument(batch):
    new_src_texts = []
    new_chosen_texts = []
    new_rejected_texts = []
    src_langs = []
    tgt_langs = []

    # Lấy kích thước batch dựa trên cột 'source'
    batch_size = len(batch['source'])

    for i in range(batch_size):
        # Lấy văn bản và làm sạch
        src_text  = batch['source'][i].strip()   if isinstance(batch['source'][i], str)   else ""
        chosen_text = batch['chosen'][i].strip()   if isinstance(batch['chosen'][i], str)   else ""
        rejected_text = batch['rejected'][i].strip() if isinstance(batch['rejected'][i], str) else ""

        # Bỏ qua các dòng bị thiếu dữ liệu text
        if not src_text or not chosen_text or not rejected_text:
            continue

        # Lấy mã ngôn ngữ
        raw_src_lang = batch['src_lang'][i].strip()
        raw_tgt_lang = batch['tgt_lang'][i].strip()

        # Map sang mã chuẩn (nếu trong dữ liệu mới src_lang/tgt_lang đã chuẩn hóa rồi thì có thể bỏ LANG_CODE_MAP)
        # Sử dụng .get() để an toàn trả về chính nó nếu không có trong map
        src_code = LANG_CODE_MAP.get(raw_src_lang, raw_src_lang)
        tgt_code = LANG_CODE_MAP.get(raw_tgt_lang, raw_tgt_lang)

        # Lọc ngôn ngữ (Tùy chọn: Nếu bạn vẫn muốn kiểm tra xem có thuộc danh sách cho phép không)
        # if src_code not in langs_augrument and tgt_code not in langs_augrument:
        #     continue

        # Thêm trực tiếp dữ liệu một chiều duy nhất như trong dataset
        new_src_texts.append(normalize_text(src_text))
        new_chosen_texts.append(normalize_text(chosen_text))
        new_rejected_texts.append(normalize_text(rejected_text))
        src_langs.append(src_code)
        tgt_langs.append(tgt_code)

    # Nếu bạn muốn giữ lại điểm (score) cho các bước pipeline sau, có thể thêm khóa "chosen_score" và "rejected_score"
    return {
        "src_text":      new_src_texts,
        "chosen_text":   new_chosen_texts,
        "rejected_text": new_rejected_texts,
        "src_lang":      src_langs,
        "tgt_lang":      tgt_langs,
    }


# ============================================= TOKENIZE ==================================================
# Output columns: input_ids, attention_mask, labels (chosen), rejected_labels

def preprocess_function(examples):
    src_encoded = tokenizer(
        examples["src_text"],
        padding=False,
        add_special_tokens=False,
    )
    chosen_encoded = tokenizer(
        text_target=examples["chosen_text"],
        padding=False,
        add_special_tokens=False,
    )
    rejected_encoded = tokenizer(
        text_target=examples["rejected_text"],
        padding=False,
        add_special_tokens=False,
    )

    all_input_ids       = []
    all_attention_mask  = []
    all_labels          = []
    all_rejected_labels = []

    eos_id = tokenizer.eos_token_id

    for i in range(len(examples["src_text"])):
        src_tokens = src_encoded["input_ids"][i]
        chosen_tokens = chosen_encoded["input_ids"][i]
        rejected_tokens = rejected_encoded["input_ids"][i]

        # NẾU CÂU QUÁ DÀI -> BỎ QUA (Không cắt ngang để tránh hỏng logic [EOS])
        # Trừ 2 vì cần chừa chỗ cho [LANG_ID] và [EOS]
        if (len(src_tokens) > MAX_LENGTH - 2 or 
            len(chosen_tokens) > MAX_LENGTH - 2 or 
            len(rejected_tokens) > MAX_LENGTH - 2):
            
            # Gán rỗng để lát nữa dùng tập dataset .filter(lambda x: len(x["input_ids"]) > 0) lọc đi
            all_input_ids.append([])
            all_attention_mask.append([])
            all_labels.append([])
            all_rejected_labels.append([])
            continue

        src_lang_id = tokenizer.convert_tokens_to_ids(examples["src_lang"][i])
        tgt_lang_id = tokenizer.convert_tokens_to_ids(examples["tgt_lang"][i])

        # Encoder input: [LANG_ID] + tokens + [EOS]
        input_ids = [src_lang_id] + src_tokens + [eos_id]
        attention_mask = [1] * len(input_ids)

        # Decoder targets: [LANG_ID] + tokens + [EOS]
        labels = [tgt_lang_id] + chosen_tokens + [eos_id]
        rejected_labels = [tgt_lang_id] + rejected_tokens + [eos_id]

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(labels)
        all_rejected_labels.append(rejected_labels)

    return {
        "input_ids":        all_input_ids,
        "attention_mask":   all_attention_mask,
        "labels":           all_labels,
        "rejected_labels":  all_rejected_labels,
    }

# ======================================= CUSTOM DATA COLLATOR ============================================
# Extends DataCollatorForSeq2Seq to also pad rejected_labels with -100.

@dataclass
class SimPODataCollator(DataCollatorForSeq2Seq):
    """Pads `rejected_labels` exactly like standard `labels` (pad value = -100)."""

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        # Temporarily pop rejected_labels so the parent collator ignores them
        rejected_labels = [f.pop("rejected_labels") for f in features]

        batch = super().__call__(features, return_tensors=return_tensors)

        # Pad rejected_labels to the longest sequence in the batch
        max_len = max(len(seq) for seq in rejected_labels)
        padded_rejected = [
            seq + [-100] * (max_len - len(seq))
            for seq in rejected_labels
        ]

        # Round up to multiple of 8 if the parent did the same for labels
        labels_len = batch["labels"].shape[1]
        if labels_len > max_len:
            # Extend to match labels padding width for consistent tensor shapes
            padded_rejected = [
                seq + [-100] * (labels_len - len(seq))
                for seq in padded_rejected
            ]

        batch["rejected_labels"] = torch.tensor(padded_rejected, dtype=torch.long)

        # Restore the feature dicts in case the caller reuses them
        for f, rl in zip(features, rejected_labels):
            f["rejected_labels"] = rl

        return batch


# ======================================= SimPO-CPO TRAINER ===============================================

def _seq_log_probs(
    logits: torch.Tensor,   # (B, T, V)
    labels: torch.Tensor,   # (B, T)
):
    """
    Computes per-sequence log-probabilities from decoder logits.

    Returns:
        log_prob_sum  (B,) — Σ log π(yₜ|x, y<t)  for non-padding tokens
        log_prob_norm (B,) — log_prob_sum / |y|    (length-normalised, used in SimPO margin)
        seq_len       (B,) — number of non-padding tokens |y|
    """
    log_probs = F.log_softmax(logits, dim=-1)                       # (B, T, V)
    token_log_probs = torch.gather(
        log_probs, dim=2, index=labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)                                                    # (B, T)

    mask          = (labels != -100).float()                         # (B, T)
    log_prob_sum  = (token_log_probs * mask).sum(dim=1)              # (B,)
    seq_len       = mask.sum(dim=1).clamp(min=1.0)                   # (B,)
    log_prob_norm = log_prob_sum / seq_len                           # (B,)
    return log_prob_sum, log_prob_norm, seq_len


class SimPOSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Seq2SeqTrainer implementing the CPO-SimPO hybrid loss:

        L = -E[ log σ( β/|y_w|·log π(y_w|x) - β/|y_l|·log π(y_l|x) - γ )
                + α·log π(y_w|x) ]

    Where:
        β/|y_w|·log π(y_w|x)  — length-normalised log-prob of chosen   (SimPO term)
        β/|y_l|·log π(y_l|x)  — length-normalised log-prob of rejected  (SimPO term)
        α·log π(y_w|x)         — full (unnormalised) NLL regularizer      (CPO term)

    Args:
        beta  (float): temperature for the SimPO margin, default 2.0
        gamma (float): target reward gap γ, default 0.5
        alpha (float): weight of the CPO NLL regularizer, default 0.5; set 0.0 to disable
    """

    def __init__(self, *args, beta: float = 2.0, gamma: float = 0.5, alpha: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta  = beta
        self.gamma = gamma
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # ── 1. Split labels ─────────────────────────────────────────────────
        rejected_labels = inputs.pop("rejected_labels")     # (B, T_rej)
        chosen_labels   = inputs["labels"]                  # (B, T_cho)

        # ── 2. Forward: chosen ──────────────────────────────────────────────
        chosen_outputs = model(**inputs)
        chosen_logits  = chosen_outputs.logits              # (B, T_cho, V)

        # ── 3. Forward: rejected (same encoder output, different decoder) ───
        rejected_inputs          = {k: v for k, v in inputs.items() if k != "labels"}
        rejected_inputs["labels"] = rejected_labels
        rejected_outputs = model(**rejected_inputs)
        rejected_logits  = rejected_outputs.logits          # (B, T_rej, V)

        # ── 4. Log-probabilities ─────────────────────────────────────────────
        # log_prob_sum  = Σ log π(yₜ|x)        ← used in CPO NLL term (α·log π(y_w|x))
        # log_prob_norm = Σ log π(yₜ|x) / |y|  ← used in SimPO margin  (β/|y|·log π(y|x))
        chosen_sum,   chosen_norm,   _ = _seq_log_probs(chosen_logits,   chosen_labels)
        rejected_sum, rejected_norm, _ = _seq_log_probs(rejected_logits, rejected_labels)

        # ── 5. SimPO margin loss ─────────────────────────────────────────────
        # -log σ( β·(log π_norm(y_w|x) - log π_norm(y_l|x)) - γ )
        margin     = self.beta * (chosen_norm - rejected_norm) - self.gamma
        simpo_loss = -F.logsigmoid(margin)                  # (B,)

        # ── 6. CPO NLL term ──────────────────────────────────────────────────
        # -α·log π(y_w|x)  — unnormalised, matches the formula exactly
        cpo_nll = -self.alpha * chosen_sum                  # (B,)

        # ── 7. Combined loss (expectation over batch) ────────────────────────
        loss = (simpo_loss + cpo_nll).mean()

        return (loss, chosen_outputs) if return_outputs else loss


# ======================================== DATASET LOADING ================================================

ds_augrument = load_dataset("json", data_files={
    "train": [
        "nllb_comet_rl_dataset_other->vi.jsonl", 
        "nllb_comet_rl_dataset_vi_other.jsonl"
    ],
    "test":  "nllb_comet_rl_dataset_test.jsonl"
})

ds_augrument = ds_augrument.map(
    make_all_vi_pairs_batched_augrument,
    batched=True,
    batch_size=1000,
    remove_columns=ds_augrument["train"].column_names,
    num_proc=8,
    desc="🔄 Processing preference pairs",
)

print("🔡 Tokenizing datasets...")
if os.path.exists(f"{TOKENIZED_CACHE}/augment_pref"):
    tokenized_datasets_augrument = load_from_disk(f"{TOKENIZED_CACHE}/augment_pref")
else:
    tokenized_datasets_augrument = ds_augrument.map(
        preprocess_function,
        batched=True,
        batch_size=2000,
        remove_columns=ds_augrument["train"].column_names,
        num_proc=8,
        desc="Tokenizing preference pairs",
    ).filter(lambda x: len(x["input_ids"]) > 0)
    tokenized_datasets_augrument.save_to_disk(f"{TOKENIZED_CACHE}/augment_pref")

combined_train_dataset = tokenized_datasets_augrument["train"].shuffle(seed=42)
combined_eval_dataset  = tokenized_datasets_augrument["test"]

print(f"Tổng mẫu huấn luyện: {len(combined_train_dataset)}")
print(f"Tổng mẫu kiểm thử:   {len(combined_eval_dataset)}")


# =================================================== TRAINING SETUP ======================================

data_collator = SimPODataCollator(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8,
    return_tensors="pt",
    label_pad_token_id=-100,
)

BS     = 32
ACCUM  = 2
EPOCHS = 5
LR     = 3e-6
STEP   = 1000
LOG    = 100

training_args = Seq2SeqTrainingArguments(
    output_dir=MODEL_OUTPUT_PATH,
    per_device_train_batch_size=BS,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,

    lr_scheduler_type="linear",
    #warmup_ratio=0.1,
    warmup_steps=500,
    weight_decay=0.01,
    adam_epsilon=1e-8,

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

    bf16=True,
    fp16=False,
    tf32=True,

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_pin_memory=True,
    dataloader_num_workers=8,
    dataloader_drop_last=True,
    dataloader_prefetch_factor=4,
    full_determinism=False,

    report_to="none",
    predict_with_generate=False,
    generation_max_length=MAX_LENGTH,

    remove_unused_columns=False,  # must be False — collator adds rejected_labels
    label_names=["labels"],
    max_grad_norm=1.0,
)

trainer = SimPOSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=combined_train_dataset,
    eval_dataset=combined_eval_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    # SimPO-CPO hyperparameters (β, γ, α từ công thức)
    beta=1.0,
    gamma=0.08,
    alpha=0.1,
)

print("=== System Information ===")
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

print("🚀 Bắt đầu training...")
torch.cuda.empty_cache()
model.enable_input_require_grads()

if CONTINUE_TRAIN:
    trainer.train(resume_from_checkpoint=CHECKPOINT_PATH)
else:
    trainer.train()


# ============================================== Đánh giá ===================================================
# Evaluation uses only the `chosen` side (labels), which is the standard BLEU target.

bleu = evaluate.load("sacrebleu")

def test_overfit_nllb_prefix(model, tokenizer, dataset, max_length=256, batch_size=16):
    model.eval()
    preds, refs = [], []

    for start in range(0, len(dataset), batch_size):
        batch_samples = dataset[start : start + batch_size]

        input_ids_list = batch_samples["input_ids"]
        labels_list    = batch_samples["labels"]   # chosen labels used as reference

        max_src_len = max(len(x) for x in input_ids_list)
        padded = torch.zeros(len(input_ids_list), max_src_len, dtype=torch.long)
        attn   = torch.zeros_like(padded)
        for i, ids in enumerate(input_ids_list):
            padded[i, :len(ids)] = torch.tensor(ids)
            attn[i, :len(ids)]   = 1

        target_lang_ids  = [labels[0] for labels in labels_list]
        decoder_start_id = model.config.decoder_start_token_id

        decoder_input_ids = torch.tensor(
            [[decoder_start_id, lang_id] for lang_id in target_lang_ids],
            dtype=torch.long,
        ).to(model.device)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.generate(
                input_ids=padded.to(model.device),
                attention_mask=attn.to(model.device),
                max_length=max_length,
                decoder_input_ids=decoder_input_ids,
            )

        for i, (out, labels) in enumerate(zip(outputs, labels_list)):
            label_ids = [x for x in labels if x != -100]
            pred_text = tokenizer.decode(out, skip_special_tokens=True).strip()
            ref_text  = tokenizer.decode(label_ids, skip_special_tokens=True).strip()
            preds.append(pred_text)
            refs.append([ref_text])

            if start == 0 and i == 0:
                print("\n" + "=" * 50)
                print(f"🎯 Reference : {ref_text}")
                print(f"🤖 Prediction: {pred_text}")
                print("=" * 50 + "\n")

    bleu_score = bleu.compute(predictions=preds, references=refs)
    print("\nBLEU SCORE:", bleu_score["score"])


indices = random.sample(range(len(tokenized_datasets_augrument["test"])), 1000)
eval_sample = tokenized_datasets_augrument["test"].select(indices)
test_overfit_nllb_prefix(model, tokenizer, eval_sample)


# =================================================== SAVE ===================================================
print(f"💾 Saving model to {ADAPTER_PATH}...")
model.save_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(ADAPTER_PATH)
print("✅ Hoàn tất!")