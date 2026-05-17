import os
import json
import random
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Configurable: switch between local and API via env var
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "True").lower() == "False"

if USE_LOCAL_MODEL:
    from model_local import LocalTranslator
    # Initialize the translator on startup
    translator = LocalTranslator()
else:
    from model_api import APITranslator
    API_URL = os.getenv("TRANSLATION_API_URL", "http://127.0.0.1:50001/api/translate")
    translator = APITranslator(api_url=API_URL)

lang_map = {
    'fil': 'tgl_Latn', 'khm': 'khm_Khmr', 'lo': 'lao_Laoo',
    'th': 'tha_Thai', 'my': 'mya_Mymr', 'vi': 'vie_Latn',
    'en': 'eng_Latn', 'id': 'ind_Latn', 'ms': 'zsm_Latn',
}

@app.route('/')
def index():
    return render_template('index.html', languages=lang_map)

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get('text', '')
    src = data.get('src', 'en')
    tgt = data.get('tgt', 'vi')
    use_pivot = data.get('use_pivot', False)
    pivot = data.get('pivot', 'en')
    params = data.get('params', {})
    
    src_code = lang_map.get(src, 'eng_Latn')
    tgt_code = lang_map.get(tgt, 'vie_Latn')
    pivot_code = lang_map.get(pivot, 'eng_Latn')
    
    # RLHF logic: 30% chance
    is_rlhf = random.random() < 0.1
    
    try:
        if is_rlhf:
            if use_pivot:
                # Pivot translation logic for RLHF
                pivot_text = translator.translate(text, src_code, pivot_code, **params)
                opt_a, opt_b = translator.generate_rlhf_candidates(pivot_text, pivot_code, tgt_code, **params)
            else:
                opt_a, opt_b = translator.generate_rlhf_candidates(text, src_code, tgt_code, **params)
                
            return jsonify({
                "status": "success",
                "is_rlhf": True,
                "options": [opt_a, opt_b],
                "original_text": text
            })
        else:
            if use_pivot:
                pivot_text = translator.translate(text, src_code, pivot_code, **params)
                translated_text = translator.translate(pivot_text, pivot_code, tgt_code, **params)
            else:
                translated_text = translator.translate(text, src_code, tgt_code, **params)
                
            return jsonify({
                "status": "success",
                "is_rlhf": False,
                "translated_text": translated_text
            })
            
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/rlhf_feedback', methods=['POST'])
def rlhf_feedback():
    data = request.json
    source = data.get('source', '')
    chosen = data.get('chosen', '')
    rejected = data.get('rejected', '')
    
    # Append log to rlhf_data.jsonl
    with open('rlhf_data.jsonl', 'a', encoding='utf-8') as f:
        json.dump({"source": source, "chosen": chosen, "rejected": rejected}, f, ensure_ascii=False)
        f.write('\n')
        
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
