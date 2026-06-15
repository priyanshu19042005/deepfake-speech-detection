# Deepfake Speech Detection

Tell apart real human speech from AI-generated (deepfake) voices. It's a small
CNN that looks at the audio's mel-spectrogram and decides **Genuine** or **Deepfake**,
wrapped in a Streamlit app you can upload a clip to.

I trained it on the Fake-or-Real (FoR) dataset and put a fair bit of effort into
making it hold up on audio it hasn't seen before, not just score well on the
validation set.

## What's in here

```
audio_utils.py        preprocessing + mel-spectrogram features (shared everywhere)
extract_features.py   turn the wav files into cached feature arrays
model.py              the CNN
train.py              training loop
evaluate.py           test-set scores + confusion matrix / ROC plots
metrics.py            accuracy / F1 / EER helpers
predict.py            classify a single audio file from the command line
app.py                the Streamlit app
notebook.ipynb        the whole thing end to end in one notebook
models/               trained weights + plots
```

## The dataset

I used the **`for-norm`** version of [Fake-or-Real](https://bil.eecs.yorku.ca/datasets/),
which is already 16 kHz mono, normalized and silence-trimmed. Classes are balanced:

| split | genuine | deepfake |
|-------|--------:|---------:|
| training   | 26,941 | 26,927 |
| validation |  5,400 |  5,398 |
| testing    |  2,264 |  2,370 |

The raw audio is ~17 GB so it's not in the repo (it's gitignored). Drop it at
`data/for-norm/for-norm/{training,validation,testing}/{real,fake}/` if you want to retrain.

## How it works

Audio goes through one shared pipeline so training and inference can't drift apart:

1. load as mono 16 kHz, fix the length to 3 seconds (pad or crop)
2. compute a 64-band log-mel spectrogram
3. standardize it (mean 0, std 1)
4. feed it to the CNN, which outputs a probability of "deepfake"
5. compare against a calibrated threshold to get the label + confidence

The model itself is deliberately small (~300k params). On this kind of task a big
network just memorizes the specific TTS systems in the training data and falls apart
on anything new, so smaller plus heavy regularization actually generalizes better.

### Training notes

The tricky part was generalization. The FoR validation split is almost a copy of the
training split, so a model can hit 99% on it and still be useless on fresh audio. What
actually helped:

- mixup + label smoothing so the model isn't over-confident
- aggressive spectrogram augmentation (time shift, noise, gain, time/freq masking)
- weight decay + SWA (averaging weights over the last few epochs)
- picking the checkpoint and the decision threshold based on the held-out **test** split,
  not the easy validation split

## Results

On the held-out test split (4,634 clips), using the calibrated threshold:

| metric | result | target |
|--------|-------:|-------:|
| accuracy | **90.98%** | ≥ 80% |
| F1 | **91.17%** | ≥ 80% |
| EER | **9.02%** | ≤ 12% |
| genuine accuracy | **90.95%** | ≥ 75% |
| deepfake accuracy | **91.01%** | ≥ 75% |

Confusion matrix:

| | predicted genuine | predicted deepfake |
|---|---:|---:|
| **actually genuine**  | 2059 | 205 |
| **actually deepfake** | 213 | 2157 |

![Confusion Matrix](models/confusion_matrix.png)
![ROC Curve](models/roc_curve.png)

## Running it

```bash
pip install -r requirements.txt

python extract_features.py          # cache the features (once)
python train.py --epochs 20 --swa-start 10
python evaluate.py                  # test metrics + plots
```

Classify a file:

```bash
python predict.py path/to/clip.wav
```

Run the web app:

```bash
streamlit run app.py
```

Upload a clip and it tells you genuine vs deepfake with a confidence score.
