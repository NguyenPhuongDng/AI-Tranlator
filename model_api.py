import requests

class APITranslator:
    def __init__(self, api_url="http://localhost:50001/api/translate"):
        self.api_url = api_url
        
    def translate(self, text, src_lang, tgt_lang, **kwargs):
        payload = {
            "text": text,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "params": kwargs,
            "rlhf": False
        }
        response = requests.post(self.api_url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("translated_text", "")
        
    def generate_rlhf_candidates(self, text, src_lang, tgt_lang, **kwargs):
        payload = {
            "text": text,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "params": kwargs,
            "rlhf": True
        }
        response = requests.post(self.api_url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("option_a", ""), data.get("option_b", "")
