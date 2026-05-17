import argparse
from model_api import APITranslator

def main():
    parser = argparse.ArgumentParser(description="Test API translation on server")
    parser.add_argument("--url", type=str, default="http://localhost:5001/api/translate", help="API URL of the GPU server")
    parser.add_argument("--text", type=str, default="Hello, this is a test from the client.", help="Text to translate")
    parser.add_argument("--src", type=str, default="eng_Latn", help="Source language")
    parser.add_argument("--tgt", type=str, default="vie_Latn", help="Target language")
    parser.add_argument("--rlhf", action="store_true", help="Test RLHF dual-option generation")
    args = parser.parse_args()

    print(f"Connecting to API Server at: {args.url}")
    translator = APITranslator(api_url=args.url)

    if args.rlhf:
        print(f"Generating RLHF options for: '{args.text}'")
        opt_a, opt_b = translator.generate_rlhf_candidates(args.text, args.src, args.tgt)
        print("\n--- Results ---")
        print(f"Option A: {opt_a}")
        print(f"Option B: {opt_b}")
    else:
        print(f"Translating: '{args.text}'")
        result = translator.translate(args.text, args.src, args.tgt)
        print("\n--- Result ---")
        print(f"Translated: {result}")

if __name__ == "__main__":
    main()
