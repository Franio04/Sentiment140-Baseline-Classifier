# train_and_use.py
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models, Model
from keras.saving import register_keras_serializable  # <-- for Lambda deserialization

# ---------------- 0) Load dataset + info ----------------
ds, info = tfds.load(
    "sentiment140",
    split="train",
    as_supervised=True,
    download=False,
    with_info=True,
)
print("Data dir:", info.data_dir)
print("Train examples:", info.splits['train'].num_examples)

# ---------------- 1) Preprocessing ----------------
def clean_text(text, label):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, r"http\S+", "")
    text = tf.strings.regex_replace(text, r"@\w+", "")
    text = tf.strings.regex_replace(text, r"#", "")
    text = tf.strings.regex_replace(text, r"[^a-z\s]", "")
    text = tf.strings.strip(text)
    return text, label

ds_clean = ds.map(clean_text, num_parallel_calls=tf.data.AUTOTUNE)

# ---------------- 2) Binarize labels ----------------
def binarize_label(text, y):
    y = tf.cast(tf.equal(y, 4), tf.int32)  # 4->1, 0->0
    return text, y

ds_bin = ds_clean.map(binarize_label, num_parallel_calls=tf.data.AUTOTUNE)

# ---------------- 3) Train/val split ----------------
VAL_SIZE = 50_000
N = info.splits['train'].num_examples
TRAIN_SIZE = N - VAL_SIZE

train_stream = ds_bin.take(TRAIN_SIZE)
val_stream   = ds_bin.skip(TRAIN_SIZE).take(VAL_SIZE)

# ---------------- 4) TextVectorization ----------------
MAX_VOCAB = 30_000
MAX_LEN   = 64
vectorizer = layers.TextVectorization(
    max_tokens=MAX_VOCAB,
    standardize=None,
    output_mode="int",
    output_sequence_length=MAX_LEN,
)
# build vocab on a subset
text_only_for_adapt = train_stream.map(lambda t, y: t).take(100_000).batch(1024)
vectorizer.adapt(text_only_for_adapt)

def vectorize_batch(text, y):
    return vectorizer(text), y

# ---------------- 5) Final pipelines ----------------
BUFFER = 10_000
BATCH  = 512

train_ds = (
    train_stream
      .shuffle(BUFFER)
      .batch(BATCH)
      .map(vectorize_batch, num_parallel_calls=tf.data.AUTOTUNE)
      .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    val_stream
      .batch(BATCH)
      .map(vectorize_batch, num_parallel_calls=tf.data.AUTOTUNE)
      .prefetch(tf.data.AUTOTUNE)
)

# ---------------- 6) Model ----------------
EMB_DIM = 128
classifier = models.Sequential([
    layers.Embedding(MAX_VOCAB, EMB_DIM),  # no input_length (deprecated)
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")
])
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

steps_per_epoch = (TRAIN_SIZE + BATCH - 1) // BATCH
val_steps       = (VAL_SIZE   + BATCH - 1) // BATCH

history = classifier.fit(
    train_ds,
    epochs=3,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=val_steps,
)

# ---------------- 7) Build end-to-end (strings -> prob) ----------------
@register_keras_serializable(package="sent140")  # <-- makes Lambda reloadable
def clean_text_infer(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, r"http\S+", "")
    text = tf.strings.regex_replace(text, r"@\w+", "")
    text = tf.strings.regex_replace(text, r"#", "")
    text = tf.strings.regex_replace(text, r"[^a-z\s]", "")
    text = tf.strings.strip(text)
    return text

inp = layers.Input(shape=(), dtype=tf.string, name="text")
cleaned = layers.Lambda(clean_text_infer, name="clean", output_shape=())(inp)  # explicit shape
token_ids = vectorizer(cleaned)
probs = classifier(token_ids)
end2end = Model(inp, probs, name="sent140_end2end")

# ---------------- 8) Classify right now ----------------
def classify_tweets(texts, threshold=0.5, return_04=False, model=end2end):
    if isinstance(texts, (str, bytes)):
        texts = [texts]
    x = tf.convert_to_tensor(texts, dtype=tf.string)   # pass tf.string tensor
    probs = model(x, training=False).numpy().ravel()
    y01 = (probs >= threshold).astype(int)
    return probs, (y01 * 4) if return_04 else y01

# quick demo
p, y = classify_tweets("i absolutely love this phone!")
print(float(p[0]), int(y[0]))
texts = ["this update ruined the app", "wow amazing customer service!", "meh it's okay"]
probs, labels = classify_tweets(texts, return_04=True)
for t, pr, lb in zip(texts, probs, labels):
    print(f"{pr:.3f} -> {lb} | {t}")

# ---------------- 9) Save once (Keras v3 format) ----------------
SAVE_PATH = "sent140_end2end.keras"
end2end.save(SAVE_PATH)
print("Saved to:", SAVE_PATH)

# ---------------- 10) Load later & use ----------------
# (No custom_objects needed because we registered the function)
loaded = tf.keras.models.load_model(SAVE_PATH)

def classify_with_loaded(texts, threshold=0.5, return_04=False):
    if isinstance(texts, (str, bytes)):
        texts = [texts]
    x = tf.convert_to_tensor(texts, dtype=tf.string)   # tensor, not raw list
    probs = loaded(x, training=False).numpy().ravel()
    y01 = (probs >= threshold).astype(int)
    return probs, (y01 * 4) if return_04 else y01

# quick demo with loaded
probs, preds = classify_with_loaded(
    ["i absolutely love this phone!", "terrible experience"],
    return_04=True
)
print("Loaded demo:", probs, preds)
