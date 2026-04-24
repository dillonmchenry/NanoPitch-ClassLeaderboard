# NanoPitch — Modifications Write-Up

## 1. Overview

The goal of my modifications to Nanopitch was to improve the model's real-world pitch tracking quality while maintaining its ability to run in a browser real-time environment. I wanted to raise the Raw Pitch Accuracy and lower the median cent, but also keep those metrics strong across the full evaluation SNR range (-5, 0, +5, +10, +20db) rather than focusing solely on clean audio. All changes to the Nanopitch workflow took place on the training side (targets, loss, augmentation, LR schedule) so the exported weights still loaded into the C/WASM interface without needing a rebuild.

Overall, my approach experimented with stabilizing the training recipe (implementing staging, loss balancing, VAD weighting) then layering in further improvements (late-phase augmentation, GRU increase, epoch increase) once the baseline was well-behaved.

## 2. Changes Made

### Improvements that helped

- **Augmentation + clean probability**
  - Random-SNR noise mixing via `logaddexp` in `augment_mel_batch` helped the model deal with more realistic noise conditions.
  - In order to preserve clean accuracy I added (`--aug-clean-prob`) so the data wasn't dominated by noise.
  - `--aug-clean-prob-late`, `--aug-clean-late-frac` raise the clean probability in the final 20% of training, letting the model polish pitch on cleaner data once the model is mostly established.
- **Epoch amount**
  - Extending to **150 epochs** gave VAD, VDR and RPA more time to improve.
  - As I ramped up the epochs, I started to notice late-stage pitch drift and the Tensorboard graph flattened ~150.
- **GRU size**
  - Bumped `--gru-size` from **96 to 112**, increasing modeling capacity.
  - Saw across-the-board improvements (VDR, RPA, medians) which suggests that the initial configuration was limited by capacity.
  - I figured that the larger hidden state would be able to hold more pitch context and clear up some confusions around octaves or short note dropouts. This change did slightly raise the real time factor in the browser, but only a modest amount.
- **Loss weights**
  - Tuned `--w-vad` to 0.08 and `--w-pitch` to 1.1 to keep VAD learning stable while leaning slightly more on pitch supervision during the joint phase.
  - Small steps here moved VDR and pitch medians in opposite directions, so I settled on these values after a few sweep runs.
  - Since both heads share the backbone, the weight ratio decides which objective wins when gradients pull against each other. VAD is only 1-dim and pitch is 360-dim, so the raw losses aren't on the same scale — nudging the ratio toward pitch felt necessary to protect cent-level precision without killing VAD progress.
- **Splitting VAD + pitch epochs**
  - Landed on **5 epochs VAD-only → 20 epochs pitch-only → joint**; this beat both 5+10 and 10+10 on VDR, RPA, and median cents.
  - My read is that VAD converges fast and, if left to train jointly from scratch, its gradients dominate the backbone before pitch has had a chance to shape useful features. A short VAD warmup followed by a longer pitch-only stretch lets the backbone build pitch-aware representations first, so by the time joint training starts the two heads interfere with each other a lot less.
- **Cosine annealing (learning rate)**
  - Per-batch cosine from `--lr 1e-3` down to `--lr-min 1e-5` across the full run (`T_max = batches * epochs`).
  - A high early LR helps the model move quickly across the loss landscape, and the smooth tail lets it settle instead of bouncing around. I made sure `T_max` scales with `--epochs` so longer runs don't silently decay too fast or stall at a too-high floor.
- **Voiced/unvoiced balancing**
  - Once we learned that the dataset is roughly 60% voiced, I ended up pushing the weight **below** 1.0 (landed at **0.6**) — the opposite of the README's default guidance, which assumes a voiced-minority dataset.
  - My intuition is that BCE already over-counts the majority class, so in a voiced-heavy dataset `pos_weight < 1` takes some pressure off the abundant voiced frames and gives unvoiced/boundary frames more say. That seems to sharpen the decision right at voicing transitions, which is where VAD mistakes tend to corrupt downstream pitch scoring.

### Experiments that did not help

- **Segment length**
  - Changing `--seq-len` beyond the chosen value did not produce consistent gains to justify the memory/runtime cost. The model must already see enough context for singing phrases, so the longer clips mostly added compute without new learning.
- **Dropout layers** 
  - Added and then removed from the model; net effect was weaker tracking metrics at this model size. As discussed in class, applying heavy dropout to the spectrograms distorted the singing data too much, so I mostly relied on the noise augmentation as a source of normalization.
- **Widened pitch target Gaussian**
  - Raised `sigma_bins` from 1.2 to 2.0 to soften the pitch labels; VDR jumped dramatically but RPA fell and median cent error rose, so I reverted for the final model. My assumption is that spreading the Gaussian across neighboring bins made "being voiced" easier to learn (which shows up as higher VDR), while blurring the signal that tells the model which exact bin is correct, so cent-level precision suffered.
- **Viterbi decode hyperparameters**
  - Tuned `transition_width`, `voicing_threshold`, and onset penalties to see if decoder smoothing could squeeze out more accuracy, but the tradeoffs just reshuffled which SNR rows won and lost. Since the decoder is a smoother sitting on top of already-reasonable posteriors, and since any change would have required a WASM rebuild to ship, I left the defaults in place.
- **Higher `--lr` / `--lr-min`**
  - As a late experiment I slightly raised both the starting and minimum learning rates to check for missed late-game gains; VDR ticked up but RPA, gross error, and realtime median cents all got worse. I think the bigger tail learning rate kept the backbone exploring instead of settling, which VAD tolerates because its decision is coarse, but pitch precision seemed sensitive to small weight changes and degrades.

## 3. Baseline Performance vs Best Training Run

Best run configuration: **150 epochs, 5+20 staging, `gru_size=112`, cosine LR, `w_vad=0.08` / `w_pitch=1.1`, `vad_pos_weight=0.6`, `aug_clean_prob=0.15` with late-phase ramp**.

### Improved Run

#### Viterbi (Offline)


| Condition   | VAD Acc   | VDR       | RPA       | RCA       | Gross    | Med. ¢   |
| ----------- | --------- | --------- | --------- | --------- | -------- | -------- |
| -5 dB       | 95.1%     | 65.1%     | 94.9%     | 95.0%     | 5.1%     | 31.1     |
| +0 dB       | 95.0%     | 67.3%     | 94.0%     | 94.3%     | 6.0%     | 34.3     |
| +5 dB       | 95.1%     | 70.5%     | 94.9%     | 95.0%     | 5.1%     | 27.1     |
| +10 dB      | 97.1%     | 71.3%     | 97.0%     | 97.1%     | 3.0%     | 6.3      |
| +20 dB      | 97.6%     | 73.9%     | 96.9%     | 97.0%     | 3.1%     | 5.2      |
| clean       | 98.5%     | 78.4%     | 98.0%     | 98.3%     | 2.0%     | 5.3      |
| **overall** | **96.4%** | **71.1%** | **96.0%** | **96.1%** | **4.0%** | **17.9** |


#### Viterbi (Realtime — matches browser)


| Condition   | VAD Acc   | VDR       | RPA       | RCA       | Gross    | Med. ¢   |
| ----------- | --------- | --------- | --------- | --------- | -------- | -------- |
| -5 dB       | 95.1%     | 65.0%     | 93.1%     | 93.4%     | 6.9%     | 14.8     |
| +0 dB       | 95.0%     | 66.8%     | 91.5%     | 92.3%     | 8.5%     | 45.4     |
| +5 dB       | 95.1%     | 70.2%     | 93.3%     | 93.4%     | 6.7%     | 27.1     |
| +10 dB      | 97.1%     | 70.9%     | 95.3%     | 95.5%     | 4.7%     | 7.0      |
| +20 dB      | 97.6%     | 73.4%     | 94.3%     | 94.4%     | 5.7%     | 11.1     |
| clean       | 98.5%     | 78.0%     | 97.0%     | 97.4%     | 3.0%     | 5.4      |
| **overall** | **96.4%** | **70.7%** | **94.1%** | **94.4%** | **5.9%** | **18.3** |


### NanoPitch Baseline

#### Viterbi (Offline)


| Condition   | VAD Acc   | VDR       | RPA       | RCA       | Gross    | Med. ¢   |
| ----------- | --------- | --------- | --------- | --------- | -------- | -------- |
| -5 dB       | 79.2%     | 59.0%     | 87.4%     | 89.5%     | 12.6%    | 94.3     |
| +0 dB       | 81.4%     | 63.1%     | 92.2%     | 93.4%     | 7.8%     | 32.5     |
| +5 dB       | 83.6%     | 67.3%     | 92.2%     | 92.2%     | 7.8%     | 28.7     |
| +10 dB      | 90.3%     | 68.8%     | 92.9%     | 93.2%     | 7.1%     | 18.2     |
| +20 dB      | 89.3%     | 69.5%     | 93.1%     | 93.1%     | 6.9%     | 9.7      |
| clean       | 98.6%     | 87.3%     | 95.4%     | 95.6%     | 4.6%     | 8.2      |
| **overall** | **87.1%** | **69.2%** | **92.3%** | **92.9%** | **7.7%** | **30.8** |


#### Viterbi (Realtime — matches browser)


| Condition   | VAD Acc   | VDR       | RPA       | RCA       | Gross     | Med. ¢   |
| ----------- | --------- | --------- | --------- | --------- | --------- | -------- |
| -5 dB       | 79.2%     | 58.1%     | 86.2%     | 88.1%     | 13.8%     | 91.2     |
| +0 dB       | 81.4%     | 61.8%     | 86.5%     | 87.7%     | 13.5%     | 106.9    |
| +5 dB       | 83.6%     | 66.1%     | 90.4%     | 90.9%     | 9.6%      | 29.1     |
| +10 dB      | 90.3%     | 67.5%     | 89.0%     | 90.4%     | 11.0%     | 30.4     |
| +20 dB      | 89.3%     | 68.4%     | 90.8%     | 90.9%     | 9.2%      | 11.0     |
| clean       | 98.6%     | 85.0%     | 94.7%     | 95.0%     | 5.3%      | 8.2      |
| **overall** | **87.1%** | **67.8%** | **89.7%** | **90.5%** | **10.3%** | **45.7** |


