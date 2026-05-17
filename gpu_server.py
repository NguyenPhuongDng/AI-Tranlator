import os
import re
from flask import Flask, request, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import random
import tempfile
from peft import PeftModel

# Cấu hình cache
BASE_PATH = "/workspace/dongg"

os.environ["HF_HOME"] = f"{BASE_PATH}/hf_cache"
os.environ["HF_DATASETS_CACHE"] = f"{BASE_PATH}/hf_cache/datasets"
os.environ["TRANSFORMERS_CACHE"] = f"{BASE_PATH}/hf_cache/models"
os.environ["HF_HUB_CACHE"] = f"{BASE_PATH}/hf_cache/hub"
os.environ["TMPDIR"] = f"{BASE_PATH}/tmp"

# Tạo thư mục nếu chưa tồn tại
for path in ["datasets", "models", "hub"]:
    os.makedirs(f"{BASE_PATH}/hf_cache/{path}", exist_ok=True)
os.makedirs(f"{BASE_PATH}/tmp", exist_ok=True)

app = Flask(__name__)

print("Đang tải mô hình NLLB-200. Quá trình này có thể mất vài phút...")

class LocalTranslator:
    def __init__(self, model_name="facebook/nllb-200-3.3B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        use_PEFT = True
        if use_PEFT:
            # Sửa lại đường dẫn PEFT nếu cần
            self.model = PeftModel.from_pretrained(self.model, "NLLB_3.3B_ALT_SimPO_CPO_v2/models_output/checkpoint-7000")
            self.model.merge_and_unload()
        self.model.eval()
        
    def chunk_text_smartly(self, text, max_tokens):
        """
        Chia văn bản thành các cửa sổ trượt (chunk) dựa trên dấu câu và token.
        LƯU Ý: Không xử lý \n ở đây nữa để tránh lỗi dồn đoạn văn.
        """
        # Đã bỏ \n ra khỏi regex
        pattern = r'(?<=[.!?。។៕ฯ၏၊။])\s*'
        splits = re.split(pattern, text)
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for segment in splits:
            if not segment.strip():
                continue
                
            encoded_segment = self.tokenizer.encode(segment, add_special_tokens=False)
            segment_tokens_count = len(encoded_segment)
            
            if current_length + segment_tokens_count <= max_tokens:
                spacer = " " if current_chunk else ""
                current_chunk += spacer + segment
                current_length += segment_tokens_count
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                if segment_tokens_count > max_tokens:
                    for i in range(0, segment_tokens_count, max_tokens):
                        sub_encoded = encoded_segment[i:i+max_tokens]
                        sub_text = self.tokenizer.decode(sub_encoded, skip_special_tokens=True)
                        chunks.append(sub_text.strip())
                    current_chunk = ""
                    current_length = 0
                else:
                    current_chunk = segment
                    current_length = segment_tokens_count
                    
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def _translate_single(self, text, tgt_lang, **kwargs):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        
        # Lấy độ dài đầu vào để tính toán max_new_tokens an toàn hơn (ít nhất x2 độ dài đầu vào hoặc tối thiểu 256)
        input_length = inputs["input_ids"].shape[1]
        safe_max_tokens = max(256, input_length * 2)
        
        gen_kwargs = {
            "forced_bos_token_id": forced_bos_token_id,
            "max_new_tokens": int(kwargs.get("max_length", safe_max_tokens)),
        }
        
        method = kwargs.get("method", "Greedy Search")
        if method == "beam":
            gen_kwargs["num_beams"] = int(kwargs.get("num_beams", 5))
            gen_kwargs["early_stopping"] = True
        elif method == "sampling":
            gen_kwargs["do_sample"] = True
            gen_kwargs["top_k"] = int(kwargs.get("top_k", 50))
            gen_kwargs["top_p"] = float(kwargs.get("top_p", 0.95))
            gen_kwargs["temperature"] = float(kwargs.get("temperature", 1.0))
            
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    def translate(self, text, src_lang, tgt_lang, **kwargs):
        self.tokenizer.src_lang = src_lang
        
        use_sliding_window = kwargs.get("use_sliding_window", True)
        window_threshold = int(kwargs.get("window_threshold", 128))
        
        spaceless_langs = ["tha_Thai", "lao_Lao", "khm_Khmr", "mya_Mymr", "zho_Hans", "zho_Hant", "jpn_Jpan"]
        separator = "" if tgt_lang in spaceless_langs else " "

        # CHÌA KHÓA: Tách văn bản theo đoạn văn (\n) trước để giữ nguyên cấu trúc
        # và đảm bảo NLLB không bao giờ bị dính \n vào giữa input.
        paragraphs = text.split('\n')
        translated_paragraphs = []
        
        for para in paragraphs:
            if not para.strip():
                translated_paragraphs.append("")
                continue
                
            total_tokens = len(self.tokenizer.encode(para, add_special_tokens=False))
            
            if not use_sliding_window or total_tokens <= window_threshold:
                translated_paragraphs.append(self._translate_single(para, tgt_lang, **kwargs))
            else:
                chunks = self.chunk_text_smartly(para, window_threshold)
                translated_chunks = []
                for chunk in chunks:
                    if chunk.strip():
                        translated_chunks.append(self._translate_single(chunk, tgt_lang, **kwargs))
                
                translated_paragraphs.append(separator.join(translated_chunks))
                
        # Nối lại bằng dấu xuống dòng để trả về format y như bản gốc
        return "\n".join(translated_paragraphs)
        
    def generate_rlhf_candidates(self, text, src_lang, tgt_lang, **kwargs):
        ans1 = self.translate(text, src_lang, tgt_lang, **kwargs)
        
        kwargs2 = kwargs.copy()
        if kwargs2.get("method", "Greedy Search") in ["Greedy Search", "Beam Search"]:
            kwargs2["method"] = "sampling"
            kwargs2["temperature"] = 0.8
            kwargs2["top_p"] = 0.9
        else:
            current_temp = float(kwargs.get("temperature", 1.0))
            kwargs2["temperature"] = max(0.1, current_temp - 0.3)
            
        ans2 = self.translate(text, src_lang, tgt_lang, **kwargs2)
        
        options = [ans1, ans2]
        random.shuffle(options)
        return options[0], options[1]


translator = LocalTranslator()
print("Tải mô hình thành công! Server API đã sẵn sàng.")

@app.route('/api/translate', methods=['POST'])
def translate_api():
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No JSON payload provided"}), 400
            
        text = data.get("text", "")
        src_lang = data.get("src_lang", "eng_Latn")
        tgt_lang = data.get("tgt_lang", "vie_Latn")
        params = data.get("params", {})
        is_rlhf = data.get("rlhf", False)
        
        # Mặc định bật Sliding Window với ngưỡng 400 token (nếu user không tự truyền vào params)
        params.setdefault("use_sliding_window", True)
        params.setdefault("window_threshold", 128)
        
        if is_rlhf:
            opt_a, opt_b = translator.generate_rlhf_candidates(text, src_lang, tgt_lang, **params)
            return jsonify({
                "status": "success",
                "option_a": opt_a,
                "option_b": opt_b
            })
        else:
            translated_text = translator.translate(text, src_lang, tgt_lang, **params)
            return jsonify({
                "status": "success",
                "translated_text": translated_text
            })
            
    except Exception as e:
        print(f"Lỗi khi dịch: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 50001))
    app.run(host='0.0.0.0', port=port, debug=False)