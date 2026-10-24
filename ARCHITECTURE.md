# How Visage-Image is built

This is the long version of the model card. It covers what every block in the VAE and the DiT does, why it is there, and what was tried before settling on it. A paper is cited wherever an idea came from one, and adaptations are noted where the design departs from the source.

One theme runs through the whole document: **design choices were settled by measurement.** Bigger models did not help. Better reconstruction did not mean better generation. iFID tracks latent smoothness, not how fast a DiT learns. The architecture below is what held up under those measurements.

---

## Part I — The VAE

### The job

A diffusion model working directly at 256×256 pixels has 196,608 numbers to denoise. That is too many for a 237M-parameter transformer to handle well, so the image is compressed first. The VAE turns each image into a small grid of latent vectors, the DiT learns to generate those grids, and the VAE's decoder turns them back into pixels ([Rombach et al., 2021](https://arxiv.org/abs/2112.10752)).

The VAE has two goals that pull against each other:

1. **Reconstruct faithfully.** Eyes, lashes and catchlights are where faces fail first, and the VAE caps how much detail the DiT can ever produce.
2. **Be easy to learn.** The DiT has to model the *distribution* of latents. A latent can reconstruct perfectly and still be a poor space to generate in. [Yao et al.](https://arxiv.org/abs/2501.01423) call this the "optimization dilemma".

Most of this half of the document is about the trade-off between those two.

### The shape: f16 × 96

A 256px image becomes a **16×16×96** latent: 16× downsampling per side, 96 channels. That is 0.375 floats per pixel. FLUX.2's VAE is f8×32 (0.5 floats per pixel), so Visage-Image compresses 25% harder.

**Why f16?** Each latent cell becomes one DiT token with patch size 1. A 16×16 grid gives 256 tokens, which a DiT of this size attends over cheaply, and there is no patchify layer to blur neighbouring cells together. An f8 latent needs patch size 2 to reach the same token count, and patch size 1 on an f8 latent measured at 3.1× the cost ([see below](#patch-size-1-and-no-patchify)). f16 gets the fine token granularity for free.

**Why 96 channels?** f16×128 came first, and it reconstructed better. In short DiT screens, though, the 96-channel latent learned faster, with or without the skip connections described below. A screen trains a small DiT for a few thousand steps and scores generation FID. 96 is where the trade between reconstruction quality and learnability was fixed. The reason 128 lost comes up again [below](#where-it-landed).

### Encoder: a wavelet pyramid that ends in attention

The encoder is about 26M parameters. It follows [LiteVAE](https://arxiv.org/abs/2405.14477): instead of making a CNN rediscover image frequencies, it is handed them.

**1. Haar wavelet pyramid.** The image goes through three levels of an orthonormal Haar DWT. Each level splits the signal into a low-pass band and three detail bands (horizontal, vertical and diagonal edges), so 3 RGB channels become 12 sub-bands at half resolution. The low band is decomposed again at each level.

*Why:* faces live and die on high-frequency structure. The wavelet gives the encoder explicit edge maps at every scale from the first layer, so no capacity goes into learning them. The transform is free and invertible, so nothing is lost.

*A detail that matters:* the DWT runs in fp32 even under bf16 autocast. The detail bands are small differences between neighbouring pixels, and bf16's 8-bit mantissa rounds many of them to zero.

**2. Per-level feature extractors.** Each pyramid level gets its own small CNN: a conv from 12 sub-bands to the level's width (120, 240 and 376 channels), then two residual blocks (GroupNorm, SiLU, 3×3 convs; [He et al.](https://arxiv.org/abs/1512.03385)). Average pooling brings every level onto the f8 grid.

*Why separate extractors:* fine and coarse bands want different filters. One network shared across scales would force a compromise.

**3. Fuse, then reason globally.** The three levels are concatenated, projected to 496 channels by a 1×1 conv, and passed through 4 residual blocks and 2 self-attention blocks at f8.

*Why attention here:* convolutions only see locally. Attention lets the encoder relate the left eye to the right eye, and the face to the lighting, before it commits to a code.

**4. Down to f16, the DC-AE way.** A `DCAEDown` block convolves to C/4 channels and `pixel_unshuffle`s each 2×2 neighbourhood into channels. A parameter-free shortcut runs alongside it: the same space-to-channel fold, averaged down to the output width. This is the residual autoencoding idea from [DC-AE](https://arxiv.org/abs/2410.10733). The learned path only has to add a correction to a fold that already preserves the input. Two more residual blocks and two more attention blocks follow at f16.

*Why not a strided conv:* a strided conv must learn to preserve information through the downsample. The fold preserves it by construction.

**5. Two skips straight from the pixels.** Both shortcuts come from [Qwen-Image-VAE-2.0](https://arxiv.org/abs/2605.13565):

- **Global skip:** `pixel_unshuffle(image, 16)` folds each 16×16 pixel block into one 768-vector, and a 1×1 conv adds it to the deepest feature map.
- **Latent skip:** the same fold, averaged down to 96 channels, is added directly to the latent.

*Why:* the latent starts life as a coarse thumbnail of the image, and the network learns the residual. Early training is faster and more stable.

*The trade-off:* the skips make the latent "mostly pixels". A linear map from folded pixels explains 72–81% of the latent's variance, against 16% for FLUX.2. That looked like a learnability risk, so no-skip variants were built and screened. On a same-recipe c96 pair, the skip version scored 72.25 gen-FID and the no-skip version 71.81, a tie. Removing the skip also cost reconstruction (eyes −0.81 dB in one arm), so the skips stayed. The learnability gain came from the channel count, not from the skips.

**6. A deterministic latent.** `to_moments` outputs only a mean. There is no log-variance, no sampling and no KL term.

*Why:* this is the biggest departure from the classic VAE ([Kingma & Welling](https://arxiv.org/abs/1312.6114)).

- **KL costs reconstruction.** At KL weight 1e-5, the decoder reconstructed from a latent that was about 40% noise by variance. Lowering it to 1e-6 cut pixel error by 29%.
- **KL did not help learnability.** On this architecture, KL at 1e-7 and 1e-6 both worsened iFID (21.19 and 22.48 against a 20.94 parent). The encoder sidestepped the penalty by inflating its latent scale.

The latent is regularised by **DINOv2 alignment** instead (see [the losses](#the-losses-four-terms-and-no-adversary)). The Qwen-Image-VAE-2.0 paper drops KL for the same reason: it limits capacity and conflicts with the alignment target.

**7. Per-channel normalisation.** Dataset mean and std for each channel are stored as buffers. `encode` returns zero-mean, unit-std latents, and that is what the DiT sees.

### Decoder: a ViT that paints 8×8 patches

The decoder holds most of the VAE's parameters, and it went through the most revisions.

**Why a transformer at all?** The first decoders were convolutional, like nearly every VAE. A conv decoder has to run its last layers at full image resolution, which is slow and memory-hungry. A ViT decoder ([Dosovitskiy et al.](https://arxiv.org/abs/2010.11929)) stays on the token grid and emits pixels only at the very end. With the same encoder and losses, and the decoder as the only variable, the ViT won on every validation metric and trained about 5× faster.

It starts slowly. The ViT decoder trails the conv decoder until about step 5000 and shows a visible 8px grid early on. A verdict at 2k steps would have rejected it.

**1. Embed.** A linear layer lifts each 96-channel latent cell to 768 dimensions.

**2. Six blocks at f16.** Each block is pre-norm attention plus an MLP, with current standard components:

- **RMSNorm** ([Zhang & Sennrich](https://arxiv.org/abs/1910.07467)): cheaper than LayerNorm and works as well.
- **QK-norm** ([Dehghani et al.](https://arxiv.org/abs/2302.05442)): RMSNorm on queries and keys, so attention logits cannot blow up. It is cheap insurance in bf16.
- **Axial 2D RoPE** ([Su et al.](https://arxiv.org/abs/2104.09864)), base 100: position enters as relative rotations along x and y. The low base suits a 16-cell grid. It also lets the decoder *run* at 512px, though quality does not hold there.
- **SwiGLU MLP** ([Shazeer](https://arxiv.org/abs/2002.05202)).
- **LayerScale** at 1e-4 ([Touvron et al.](https://arxiv.org/abs/2103.17239)): each residual branch starts almost silent, so a deep stack begins near the identity and trains stably.

**3. Split each token into four, then six more blocks at f8.** A linear layer maps each token to 4×768 values, and `pixel_shuffle` ([Shi et al.](https://arxiv.org/abs/1609.05158)) rearranges them into its 2×2 children. That gives 1024 tokens on a 32×32 grid, where six more blocks run.

*Why:* an f16 token covers 16×16 pixels, too coarse to place a lash line. The f8 stage gives attention a finer canvas. Going further did not work: attention at f4 (64×64) tied on detailed eye crops and lost 3.8 iFID, and a hybrid with an f4 conv tail lost 1.1 dB on eyes. f8 is the sweet spot.

**4. Linear patch head.** Each f8 token predicts its own 8×8×3 pixel patch directly, followed by `pixel_shuffle(8)`.

**5. Two residual conv refiners.** A linear head knows nothing about its neighbours, so without help the output shows seams at patch edges. Two small residual conv stacks (48 and 96 channels, GELU) run over the full-resolution output and smooth across those seams. A final `tanh` bounds the output.

*Why two:* the second refiner adds +0.09 dB on eyes and −0.5 iFID for 5% throughput. That is small but real, and it improves the whole image, not only the eyes.

**6. Zero-init where it counts.** The patch head and each refiner's last conv start at zero. At step 0 the decoder outputs flat mid-grey and the refiners are exact identities. Every pixel the model draws has to be learned.

### The losses: four terms and no adversary

```
loss = Charbonnier(x, x̂) + 3 · LPIPS(x, x̂) + 0.4 · (mcos + mdms)   [+ phase penalty from step 80k]
```

This is close to the Qwen-Image-VAE-2.0 recipe. It replaced a longer, heavier stack (see [below](#what-was-dropped)).

**Charbonnier**, a smooth L1, carries pixel fidelity. It is robust to outliers and does not over-smooth the way L2 does ([Lai et al.](https://arxiv.org/abs/1704.03915)).

**LPIPS** at weight 3 ([Zhang et al.](https://arxiv.org/abs/1801.03924)) carries perceptual fidelity: differences in deep VGG features, calibrated to human judgments. Replacing raw VGG-L1 with LPIPS cut iFID by 18% in one change. DINOv3 features as the perceptual loss lost to LPIPS by 0.7 iFID.

**DINOv2 alignment** at weight 0.4 is the latent-side term. A small projector (1×1 conv 96→1024, GELU, 1×1 conv) maps the latent onto the middle layer of a frozen [DINOv2](https://arxiv.org/abs/2304.07193) ViT-L/14. DINOv2 runs at 224px, so its patch grid is also 16×16. Two margin losses apply: **mcos** on per-token cosine and **mdms** on the matrix of pairwise token distances. The idea descends from [REPA](https://arxiv.org/abs/2410.06940) and [VA-VAE](https://arxiv.org/abs/2501.01423): a latent organised like a strong vision model's features is smoother and more semantic, and so easier to generate in.

*The margins matter.* They start strict and loosen over 200k steps to 0.2 of their slack (cosine 0.5, distance 0.3), then hold. In an earlier run the margin loosened all the way to slack, and the alignment gradient was zero for the last quarter of training, so all the benefit came from the strict early phase. Capping the loosening keeps the pressure on for the whole run.

**The phase-template penalty** targets one artifact. A linear 8×8 patch head settles on a fixed sub-pixel pattern: a faint template of about 1/255, identical in every patch. It shows up as a 2×4 px cross-hatch in flat dark skin. The artifact traces back to the head itself, and the refiners spend capacity cancelling it, imperfectly. From step 80k, a penalty on the mean residual per 8×8 phase cuts it from 0.99/255 to 0.16/255 within 50k steps.

#### What was dropped

The earlier VAE recipe was a heavier stack:

- KL regularisation
- a GAN critic
- [EQ-VAE](https://arxiv.org/abs/2502.09509)-style equivariance
- multi-resolution L1

Each piece once held a record: EQ halved iFID when first added, LPIPS + GAN composed, and flip-only equivariance briefly beat FLUX.2.

The Qwen recipe drops KL, the GAN and EQ, and it was applied to this encoder and decoder. It became the first VAE here to beat FLUX.2 on iFID while tying it on reconstruction. The paper describes the GAN as "unnecessary when the training budget is sufficiently large", and that holds here too.

The discarded parts did less than their records suggested:

- **The GAN moved colour, not texture.**
- **The EQ gain came from its downscale operator, not from the symmetry.** The EQ loss used a crude downscale that acted as a low-pass filter. A properly flip-equivariant encoder cost about 2 iFID with no gain on eyes. Rotation equivariance has no use on this corpus: 90th-percentile head roll is 34°, and only 0.22% of faces are near a quarter turn.

### Optimiser and schedule

- **[Muon](https://kellerjordan.github.io/posts/muon/)** runs on the decoder's hidden weight matrices, and AdamW on everything else. Muon orthogonalises each update, so all directions in a weight matrix learn at the same rate. Against an AdamW control it reached the same LPIPS 2–2.5× sooner for 7% more compute per step, roughly halving VAE training time.
- **Cosine LR** decays from 2e-4 to 0.1×, after 2000 warmup steps. Switching from constant LR to cosine was worth 2.6 iFID at no cost, the cheapest gain in the project.
- **Batch 8**, horizontal flips, 400k steps. Larger batches do not help: the GPU is already saturated at batch 8, and throughput falls as batch grows.

### What did not work

| idea | rationale | result |
|---|---|---|
| Bigger decoder (depth 6→12) | match FLUX.2's VAE size | FLUX.2's VAE is only 1.2× larger; the deeper decoder lost on rFID |
| Wider encoder | more capacity | worse rFID, 10% slower |
| f4 attention decoder | finer canvas for eyes | tie on eyes, −3.8 iFID |
| Released RAE / PAE tokenizers (frozen DINOv2 latents) | strong diffusability | −16.5 and −8.8 dB on eye crops; irises are redrawn |
| Sana's [DC-AE](https://arxiv.org/abs/2410.10733) f32 | 4× fewer tokens | degrades faces past use |
| Energy-score decoder (stochastic, sampled texture) | real skin texture | learned a checkerboard dither, −1.1 dB eyes |
| Noise-augmented decoder | robustness to DiT error | smoother output, worse low-noise eye detail |
| [REPA-E](https://arxiv.org/abs/2504.10483) (end-to-end VAE+DiT) | joint optimisation | collapsed in the DiT screen while the loss kept falling |

### Where it landed

On 512 held-out images at 256px, against FLUX.2's VAE:

- **Whole-image reconstruction:** a tie (PSNR 38.69 vs 38.67), with lower MAE.
- **Eye crops:** a tie (−0.02 dB).
- **iFID** ([arXiv:2603.05630](https://arxiv.org/abs/2603.05630)): 17.96 vs 21.99, a clear win on latent smoothness.
- **rFID:** 1.77 vs 0.93, a loss.

Training steps did most of this. Eye crops went from −1.64 dB at 80k to −0.02 dB at 400k, while iFID barely moved. That is unusual: earlier recipes always lost latent quality as reconstruction improved.

**iFID and learnability.** On an early three-VAE test, iFID correlated +0.998 with generation FID. The correlation did not hold. This VAE beats FLUX.2's on iFID by 4 points, yet in short DiT screens FLUX.2's latent is learned about 5× faster. iFID measures whether midpoints between latents decode well. It does not measure how quickly a DiT learns the distribution.

Latent perturbations show part of the reason. FLUX.2's decoder absorbs latent error into smooth, plausible skin. This decoder amplifies it into grain, so failed samples look broken and good samples look textured. Dropping from 128 to 96 channels halved that amplification, which is why c96 won the learnability screens.

**No available metric predicts learnability.** That is the main open problem on the VAE side.

The other limit: this is a 256px VAE. At 512px it loses 3.2 dB on eyes to FLUX.2.

---

## Part II — The DiT

### The job

Given a caption, produce a 16×16×96 latent that looks like it came from a real face photo. The DiT is a 236.8M-parameter transformer trained with rectified flow.

### Rectified flow, in one paragraph

Draw an image latent `x` and Gaussian noise `ε`. For a time `t ∈ [0, 1]`, mix them along a straight line: `x_t = (1 − t)·x + t·ε`. Train the network with plain MSE to predict the velocity `v = ε − x`, the direction along that line. To sample, start from pure noise at `t = 1` and integrate the predicted velocity back to `t = 0`. Straight paths make for a simple objective and few sampling steps ([Liu et al.](https://arxiv.org/abs/2209.03003); [Lipman et al.](https://arxiv.org/abs/2210.02747); scaled up in [SD3](https://arxiv.org/abs/2403.03206)).

### The shape: one stream, 18 blocks

Width 768, depth 18, 12 heads of 64 dimensions, SwiGLU MLPs at 4× width.

**Single stream.** Text tokens and image tokens are concatenated into one sequence and pass through the *same* weights. This is the approach of [U-ViT](https://arxiv.org/abs/2209.12152), which treats every input as a token. SD3's [MMDiT](https://arxiv.org/abs/2403.03206) gives each modality its own weights, and an earlier version of this project used MMDiT. In a model this small, shared weights put every parameter to work on every token.

### Patch size 1 and no patchify

The 16×16 latent enters as 256 tokens, one per cell, through a linear projection. No cells are merged.

*Why this matters:* the earlier f8 latent was 32×32, so it needed 2×2 patchify to reach 256 tokens. Converting a trained model to patch size 1 closed 36% of the blur gap to real photos, but cost 3.1× compute. The f16 VAE gives the same granularity at the patch-2 price. The VAE and the DiT were designed together.

### Inside a block

Each block runs attention over the full text+image sequence, then an MLP:

- **[adaLN-Zero](https://arxiv.org/abs/2212.09748)** from DiT: the timestep embedding produces a shift, a scale and a gate for each of the two sub-layers. The gates are zero-initialised, so every block starts as the identity. The LayerNorm has no learned affine of its own; the modulation provides it.
- **QK-RMSNorm**, as in the VAE decoder.
- **[Spiral RoPE](https://arxiv.org/abs/2602.03227)** on image tokens only. Standard axial RoPE gives each frequency pair either an x or a y direction. Spiral RoPE cycles the pairs through 4 orientations in [0, π) (0°, 45°, 90° and 135°), so relative position is encoded along the diagonals too. Text tokens get no position encoding here; their order already lives in the text encoder's states.

Two measurements on an earlier internal model with the same block design:

**What the timestep controls.** 92–96% of each block's modulation is constant across `t`, and the part that varies has rank ≤3. The modulators are mostly learned biases with a small timestep dial, so adaLN is not where the model's capacity lives.

**No dead blocks.** Skipping any single middle block costs only 1–2% loss. Skipping blocks 6–11 *together* costs 53%, seven times the sum of the singles. The middle blocks are redundant one at a time but load-bearing as a group, so they were not pruned.

### Timestep embedding

The timestep goes through sinusoidal features (256 dims, frequencies log-spaced from 0.01 to 100), then a small MLP. The wide frequency range lets the model resolve fine differences near `t = 0`, where detail is decided.

### The text path

The text path got the most design work, because it is where an earlier internal model broke.

**1. The encoder: Qwen3-0.6B, first 8 layers.** An earlier internal model used T5-v1.1-small. A probe compared candidate encoders: edit one slot in a caption ("blue beanie" → "red beanie", "facing" → "profile") and measure how well the encoder's states localise the change. T5-small scored a span AUC of 0.77 and [Qwen3-0.6B](https://arxiv.org/abs/2505.09388) 0.865 at layer 8. The probe also showed two things:

- **Middle layers beat the last.** A causal LM's final layers specialise in predicting the next token, while mid-network states hold the best description of the text so far. Layer 8 also means running only 8 of 28 layers, about 281M parameters. Qwen3.5-0.8B scored slightly higher (0.90), but a 439M encoder is out of proportion to a 237M DiT.
- **Binding improved.** With T5, the "beanie" token carried the same colour information regardless of the preceding words. With Qwen, it carries the right colour.

**2. The attention-sink prefix.** In causal LMs the first token soaks up attention and develops very large norms ([Xiao et al.](https://arxiv.org/abs/2309.17453)). A caption starting at position 0 would turn its first word into that sink. Every caption is therefore preceded by the fixed text "Describe the image.", and the prefix's states are dropped. Padding goes on the right and attention is causal, so a caption's states never depend on what it is batched with.

**3. Float8 rounding.** Training read text states from a cache stored in float8 (e4m3) to keep it small. At inference, `TextEncoder` rounds through float8 the same way, so the DiT sees exactly the distribution it was trained on.

**4. Projection and mask-free padding.** A linear layer maps 1024 → 768. Each caption is then padded to a length bucket (32, 64, 96 or 128) with **learned per-position pad embeddings**, not zeros and not a mask.

*Why:* with masking, a short caption's computation depends on how it was padded in the batch. With learned pads, a caption plus its pads is one fixed sequence, whatever the batch. In an earlier internal model with the same pads, they drew attention (twice as much as a real word) but carried no information: swapping in a mean pad or a bigger bucket changes nothing measurable. A key mask appears only when one batch mixes buckets.

**5. Sixteen registers.** Sixteen learned tokens are prepended to the text ([Darcet et al.](https://arxiv.org/abs/2309.16588)). The paper proposed registers to absorb the high-norm artifact tokens that ViTs develop, but that artifact did not appear here. Measured on an earlier internal model with the same registers, they do something else:

- They take about 14% of image-to-text attention (48% in block 0).
- Removing them raises the loss by 0.04, four times the cost of deleting the *entire caption*.
- Block 0 uses them to write a fixed, timestep-dependent bias into every image token, a learned attention sink for the image stream.

Eight registers recover nearly all of the effect, so 16 is slightly more than needed.

**6. No pooled text vector.** This is the main fix over an earlier internal model. That model also mean-pooled the T5 states and fed the result into every block's adaLN, as many DiTs do. On about 3% of short strings ("a woman", "portrait", "beautiful"), the pooled vector landed off-distribution. Block 0's activation RMS jumped from 1.5 to 49 and grew to about 640 by block 17, and the model sampled pure noise.

The pooled vector was the cause: when pooled vectors were swapped between prompts, the failure followed the pooled vector, not the tokens. On long captions, ablating the pooled path changed the loss by only 0.0002, so it carried almost nothing. This model has no pooled path, and the timestep is the only input to adaLN.

**7. Zero-init output.** The final adaLN and the output projection start at zero, so the untrained model predicts zero velocity.

### Training

**Timestep sampling.** `t` is drawn from a logit-normal distribution ([SD3](https://arxiv.org/abs/2403.03206)), which concentrates training on intermediate noise levels, where learning is hardest. It is then shifted toward noise by `sqrt(tokens / 64)`, which is 2.0 at 256 tokens. More tokens means more redundancy, so an image survives more noise and the schedule has to push further to destroy it.

**Caption dropout: 20%.** One caption in five is replaced with a zero embedding. This trains the unconditional branch used by [classifier-free guidance](https://arxiv.org/abs/2207.12598). Dropped samples are drawn from sharp images only (see *Sharp by default* below).

**Optimisers and schedule:**

- Muon (lr 1e-3) on the transformer's hidden matrices; AdamW (3e-4) on embeddings, projections and modulators.
- Cosine decay to 0.1×, no warmup.
- Batch 128 (64 × 2 accumulation), gradient clipping at 1. Step counts in this document, checkpoint names included, count batches of 64.
- EMA at 0.999, updated every 10 steps. The released weights are the EMA.

Screens are judged on EMA weights only, because raw-weight FID depends too heavily on the last batch.

**Captions: three registers from step 0.** Each face crop has several captions:

- a **long** structured description (age, pose, gaze, expression, hair, makeup, clothing, background, lighting), rendered from per-image attributes. Head pose, gaze, eye closure and age come from dedicated measurement models rather than a VLM's guess;
- a **short** caption;
- a **casual** prompt, phrased the way a person might type it.

Long and short/casual captions are mixed 50/50 from the first step. An earlier internal model learned short prompts only in a late fine-tune. Training on them from the start makes that fine-tune unnecessary: one run on top of the 200k model moved FID by −0.25, within noise.

**Sharp by default.** In an earlier internal model, a bare prompt like "a woman" gave grey, foggy, drawing-like images on about 10% of seeds. The cause was the caption template, not the architecture:

- Every long caption opened with a sharpness word ("Sharp", "Slightly soft", "Blurry").
- A short prompt left that slot blank, so the conditional and unconditional predictions agreed on sharpness and guidance had nothing to amplify.

The fix has three parts. Sharp captions now carry *no* sharpness word; only blur is named. Caption dropout draws only from sharp images, so the unconditional branch is sharp too. And the base run is followed by a **20k-step cosine anneal on sharp and very-sharp images only** (200k → 220k). Bare prompts now come out sharp with no prefix and no negative prompt. The anneal tied a casual-prompt fine-tune overall and gave better eyes.

**Adversarial post-training (220k → 222k).** The released checkpoint adds 2k steps of a latent adversarial loss in the style of [LADD](https://arxiv.org/abs/2403.12015). The critic is a frozen copy of the 220k DiT: features from its blocks 5, 11 and 17 go to small trainable heads that judge a noised copy of the model's one-step clean prediction against a noised real latent. Two details matter. The latents are mean-centred before the critic sees them; without that, an earlier run on FLUX.2 latents learned to win by shifting colour. And the weight is small: 0.006 next to the flow loss, about 4% of the total loss. A blind A/B on 20 random training captions went 16–0 for 222k, with 4 ties; the difference is sharpness, not composition. Longer runs (224k, 226k) began to move layout and identity, and the held-out flow loss rose only in the high-noise band (t 0.8–1), so the critic was reaching composition. The release stops at 222k.

### Sampling

The default is an SDE sampler: **churn 0.3, CFG 3, 50 steps, timestep shift 3.3.**

- **Timestep grid:** warped toward the noisy end, `t' = s·t / (1 + (s − 1)·t)` with s = 3.3, so more steps go to the high-noise region where composition is decided.
- **Churn** ([Karras et al.](https://arxiv.org/abs/2206.00364)): before each step, the sample is re-noised from `t` up to `t·(1 + churn)`, capped at 0.9, then integrated back. The re-noising is exact for this interpolant. The injected noise lets the sampler correct earlier errors instead of compounding them.
- **Why not plain Euler?** The deterministic ODE gives crisper images that score *higher* on sharpness metrics than real decoded photos, but they look over-cooked and unrealistic. On faces, visual judgment takes precedence over the metric.
- **Why exactly 50 steps?** Churn is applied per step, so more steps inject more total noise. At 80–120 steps the image washes out, and at 40 it is visibly worse. Steps and churn are coupled, and 50 with 0.3 is the tuned pair.
- **CFG 3.** 2.5–3 is the sweet spot; 4 and above over-cooks. For strong expressions, skipping guidance at `t > 0.84` (a [guidance interval](https://arxiv.org/abs/2404.07724)) keeps the cfg-2 face shape with cfg-3 detail, and fixes the stretched "angry" face. It is off by default because it is slightly worse on neutral prompts.

### Where it landed

By eye, the adversarial 222k release beat 220k 16–0 (4 ties) in a blind A/B. It also lowered held-out FID from 23.46 to 22.04 (2,048 images). On a 28-target attribute benchmark, it passes 77.0% on average (75.8% at 220k).

**What still fails, and why:**

- **"Angry"** comes out as yelling. Men in the captions are almost never "angry"; they are "intense" or "serious". Describing the face ("eyebrows furrowed, lips pressed together") works.
- **Sunglasses** end up pushed up onto the head.
- **"Biting lip"** has no training caption at all, so the model draws an object at the mouth.

The architecture-level failures are fixed, and what remains is vocabulary: the model does what the captions taught it. The next lever is the data, not the network.

---

## Takeaways

1. **A latent that scores better can learn slower.** iFID correlated +0.998 with generation FID on three VAEs, then stopped ranking learnability: this VAE beats FLUX.2's on iFID by 4 points, yet FLUX.2's latent is learned about 5× faster.
2. **The decoder decides how forgiving a latent is.** A decoder that renders latent error as grain makes every DiT error visible. Going from 128 to 96 channels halved that amplification, and the smaller latent learned faster even though it reconstructed worse.
3. **A component's gain can come from a side effect.** The GAN critic moved colour, not texture. EQ's gain came from the low-pass downscale inside its loss, not from equivariance. The recipe that dropped both set the best result.
4. **Single-block ablations hide shared work.** Skipping any one middle block of the DiT costs 1–2% loss, and skipping six together costs 53%. Pruning from the single-block numbers would have removed load-bearing capacity.
5. **A pooled text vector can carry almost nothing and still break the model.** On long captions, ablating it changed the loss by only 0.0002. On about 3% of short strings it landed off-distribution and the model sampled pure noise. The release drops it, and the timestep is the only input to adaLN.
6. **Schedules gave the cheapest gains.** Cosine LR was worth 2.6 iFID at no cost, and Muon roughly halved VAE training time. Together they gained more than any single architecture change to the VAE.
