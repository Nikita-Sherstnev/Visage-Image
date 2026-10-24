# Data

How the training set was built, and which model produced each part of a caption.

## Images

1. **Face crops.** Square head crops were cut with a YOLOv10l face detector. `scripts/crop_faces.py` reproduces this step with the release settings: the face box is expanded to cover hair and chin, made square, and clamped to the frame. Heads smaller than 256 px are skipped.
2. **Deduplication.** Near-duplicates were removed with DINOv2-base CLS embeddings, at cosine 0.98 across the whole set and 0.92 within one identity.
3. **Single subject.** Crops with a second large face (YOLOv10 face, confidence 0.35) were rejected. Crops with no detection were kept, because face detectors often miss faces in tight crops.
4. **Caps.** Each identity was capped at 1500 images at curation time and 1000 per training epoch, so a few heavily photographed people do not dominate.
5. **Quality.** Sharpness was scored with TOPIQ (`topiq_nr_swin-face`, via pyiqa). The lowest ~0.8% (score below 0.25) were dropped. The rest were kept, and the score became a caption word (below).
6. **Manual review.** A review pass flagged crops with no face, crops showing a different person than the rest of their identity folder (ArcFace similarity below 0.4), and crops with black borders.

The DiT trained on about 530k images (528,531): about 474k from 56k identities with several images each, and about 55k single images.

**Held-out split.** 881 identities (about 12.4k images) were held out entirely. Held-out FID is measured against these identities, people the DiT never saw.

## Captions

Each long caption is a fixed-order list of 23 slots (see [PROMPTING.md](PROMPTING.md)). The slots come from three kinds of source.

### Measured attributes

Attributes that a dedicated model measures more reliably than a VLM reads them were measured and written as fixed vocabulary:

| Slot | Model | Rule |
|------|-------|------|
| sharpness | TOPIQ face IQA | score < 0.45 `Blurry`, 0.45–0.62 `Slightly soft`, higher: no word |
| subject (age, sex) | MiVOLO | age floored to the decade |
| head turn and pitch | 6DRepNet | facing / three-quarter / profile; chin up / level / chin down |
| head tilt | MediaPipe face mesh | roll of the eye axis |
| gaze | UniGaze | eye contact under 15° off-axis |
| eyes closed | MediaPipe blendshapes | `eyeBlink` ≥ 0.40; overrides gaze |
| framing | MediaPipe face mesh | face fraction of the crop |
| photo type | SigLIP2 zero-shot | studio / amateur phone / video frame; zero pixel chroma means black-and-white |

The face for these models was located with InsightFace (SCRFD). The crop was padded by 0.6 before detection, which raised the detection rate from 71% to 97% on tight crops.

Some candidates were tried and rejected. HSEmotion expression labels did not match the images, so expression is read by the VLM. Skin-tone measurement (ITA) followed the lighting more than the skin, so skin is also read by the VLM.

### VLM attributes

The remaining 15 slots were written by a vision-language model as one JSON object per image: action, skin, hair, eye colour, eyebrows, facial hair, nose, lips, expression, makeup, clothing, other people in frame, objects, background and lighting. The same object also holds a one-sentence short caption and a watermark field. The watermark field is used only to scrub watermark text from captions.

1. **Format design.** The format and its instruction were developed with Qwen3.6-27B (4-bit GGUF), which was clearly more detailed than small models (e.g. 23 vs 10 distinct eyebrow descriptions on 30 images). It was too slow to label the full set.
2. **Fine-tuned 4B captioner.** Qwen3.5-4B-Base was fine-tuned with LoRA (rank 16, alpha 32, 3 epochs, lr 1e-4, vision tower frozen) to emit the 15-key JSON directly from the image. The training targets were about 550 hand-corrected labels, with a separate 40-image holdout. Only the answer tokens carry loss, and inference sends exactly the training instruction. In a blind comparison of disagreeing labels, the fine-tuned 4B was right 19 times to 6 against the previous captioner. It labelled the whole set.
3. **Casual prompts.** Stock Qwen3.5-4B-Base wrote short, casual prompts (1–8 words) from each image and its long caption. These cover vocabulary the structured captions never use: "beautiful", "headshot", "8k", "pic", "guy". Only sharp images got them, about 74k in all, so casual prompts never teach blur.

The prompts contain no few-shot examples, because the models copied example content onto unrelated images. Phrases the models overused were stripped after generation instead of being banned in the prompt, because a ban raised their rate.

### Rendering

Rendering joins the slots in the fixed order and normalizes values (for example, canonical lighting phrases). It also applies these rules:

- **Modal values are left out:** sharp, facing the camera, eye contact, head-and-shoulders. An empty slot therefore means the default, which is also what a short user prompt looks like.
- **Scrubbing:** verbal tics (unprompted "slight tilt", "neutral expression", "close-up"), names of people, contact details and watermark text are removed.

The median long caption is about 54 words, capped at 128 tokens. Rendering is separate from labelling, so the format can change without re-running any model.

### Text encoder

Captions are encoded by layer 8 of Qwen3-0.6B. In a screen that tested whether each encoder's states separate attribute spans, it scored a span AUC of 0.865, against 0.77 for T5-small. Middle layers beat the final layer.

## Training mix

- Each sample uses its long caption or a short one (the VLM's sentence, or a casual prompt where one exists), 50/50 from step 0.
- 20% of samples have their caption dropped for classifier-free guidance. Dropped samples are drawn from sharp images, so the unconditional model defaults to sharp.
- Steps 200k–220k are a cosine anneal on sharp and very-sharp images only, and the 2k steps of adversarial post-training that follow use the same subset.
- Latents and text states are cached ahead of training.

## Checks on the trained model

- **Names:** prompting with the name of a public figure does not reproduce their face. Each seed gives a different person, as it does with no name in the prompt.
- **Watermarks:** prompts that ask for watermarks produce overlay-like marks but no legible text or recognizable logos.
- **Clause order:** see [PROMPTING.md](PROMPTING.md#long-caption-format).
