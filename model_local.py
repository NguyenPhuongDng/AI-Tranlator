from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import random

class LocalTranslator:
    def __init__(self, model_name="facebook/nllb-200-3.3b"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
    def translate(self, text, src_lang, tgt_lang, **kwargs):
        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        forced_bos_token_id = self.tokenizer.lang_code_to_id[tgt_lang]
        
        # Merge kwargs with defaults
        gen_kwargs = {
            "forced_bos_token_id": forced_bos_token_id,
            "max_length": int(kwargs.get("max_length", 200)),
        }
        
        method = kwargs.get("method", "Greedy Search")
        if method == "Beam Search":
            gen_kwargs["num_beams"] = int(kwargs.get("num_beams", 5))
            gen_kwargs["early_stopping"] = True
        elif method == "Sampling":
            gen_kwargs["do_sample"] = True
            gen_kwargs["top_k"] = int(kwargs.get("top_k", 50))
            gen_kwargs["top_p"] = float(kwargs.get("top_p", 0.95))
            gen_kwargs["temperature"] = float(kwargs.get("temperature", 1.0))
            
        outputs = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
    def generate_rlhf_candidates(self, text, src_lang, tgt_lang, **kwargs):
        # Option A: using user-provided kwargs
        ans1 = self.translate(text, src_lang, tgt_lang, **kwargs)
        
        # Option B: variation based on the chosen method
        kwargs2 = kwargs.copy()
        if kwargs2.get("method", "Greedy Search") in ["Greedy Search", "Beam Search"]:
            # Introduce randomness by switching to sampling
            kwargs2["method"] = "Sampling"
            kwargs2["temperature"] = 0.8
            kwargs2["top_p"] = 0.9
        else:
            # Change temperature slightly if already sampling
            current_temp = float(kwargs.get("temperature", 1.0))
            kwargs2["temperature"] = max(0.1, current_temp - 0.3)
            
        ans2 = self.translate(text, src_lang, tgt_lang, **kwargs2)
        
        # Randomize order to avoid bias
        options = [ans1, ans2]
        random.shuffle(options)
        return options[0], options[1]
