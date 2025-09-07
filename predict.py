# predict.py
import argparse, json, sys, os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from keras.saving import register_keras_serializable  # <-- needed to register the lambda fn

# Register the SAME function signature used in training
@register_keras_serializable(package="sent140", name="clean_text_infer")
def clean_text_infer(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, r"http\S+", "")
    text = tf.strings.regex_replace(text, r"@\w+", "")
    text = tf.strings.regex_replace(text, r"#", "")
    text = tf.strings.regex_replace(text, r"[^a-z\s]", "")
    text = tf.strings.strip(text)
    return text

def load_any(model_path: str):
    # Keras 3 prefers .keras / .h5
    if model_path.endswith(".keras") or model_path.endswith(".h5"):
        # Because we registered clean_text_infer above, no custom_objects needed
        return tf.keras.models.load_model(model_path)
    # If someone passes a SavedModel dir, try TFSMLayer
    from keras.layers import TFSMLayer
    return TFSMLayer(model_path, call_endpoint="serving_default")

def run_model(model, texts):
    # Pass a tf.string tensor (avoids data-adapter issues with Python lists)
    x = tf.convert_to_tensor(texts, dtype=tf.string)
    y = model(x, training=False)
    # y is a Tensor; get numpy() and flatten
    if isinstance(y, dict):  # just in case of dict outputs
        y = next(iter(y.values()))
    return y.numpy().ravel()

def main():
    p = argparse.ArgumentParser(description="Classify tweet sentiment.")
    p.add_argument("texts", nargs="*", help="Tweet(s) to classify. If omitted, read from stdin.")
    p.add_argument("--model", default="sent140_end2end.keras",
                   help="Path to .keras/.h5 (preferred) or a SavedModel directory.")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--return-04", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    model = load_any(args.model)

    texts = args.texts or [line.strip() for line in sys.stdin if line.strip()]
    if not texts:
        print("No input tweets.", file=sys.stderr)
        sys.exit(1)

    probs = run_model(model, texts)
    preds01 = (probs >= args.threshold).astype(int)
    labels = (preds01 * 4) if args.return_04 else preds01

    if args.json:
        print(json.dumps(
            [{"text": t, "prob_pos": float(p), "label": int(l)} for t, p, l in zip(texts, probs, labels)],
            ensure_ascii=False, indent=2))
    else:
        for t, p, l in zip(texts, probs, labels):
            print(f"{p:.3f}\t{int(l)}\t{t}")

if __name__ == "__main__":
    main()
