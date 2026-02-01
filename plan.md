# **Gelato**

Replicate the "Canonical ABC" and "Image Slicing" pipeline from the paper LEGATO: LARGE-SCALE END-TO-END
GENERALIZABLE APPROACH TO TYPESET OMR.

## ** Changes from the Paper **
We will be using a different model (Gemma 3 270M) with siglip as visual encoder and perceiver resampler based in Flammingo as connector.

### **Phase 1: Data Preparation**
*Goal: Replicate the "Canonical ABC" and "Image Slicing" pipeline.*

1.  **Acquire and Clean Data:**
    * **Source:** Download the **PDMX** dataset (MusicXML files).
    * **Conversion (MusicXML $\to$ ABC):** You need to convert these to ABC notation. The paper uses `xml2abc.py`.
    * **Canonicalization Rules:** You must write a script to "clean" the ABC output as per the paper:
        * **Fixed Line Length:** Force a line break every **5 bars**.
        * **Explicit Line Breaks:** Insert a `$` character at every line break.
        * **Fixed Unit Note:** Set `L:1/8` in the header (normalize all durations).
        * **Text Removal:** Replace lyrics and titles with a generic `<text>` token to simplify the task.

2.  **Image Rendering & Slicing:**
    * **Render:** Use **MuseScore** (for PNG) and **abcm2ps** (for SVG $\to$ PNG) to generate the score images.
    * **Slicing Logic:** Do not feed the full page. Write a function that:
        1.  Crops the page into vertical segments with a **1:4 aspect ratio** (width:height).
        2.  Resizes these segments to match your SigLIP input (512 pixels wide).
        3.  Splits each segment into **$N$ patches** (The paper used 4 patches of 448x448; you will likely use 4 patches of 512x512).

### **Phase 2: Model Construction**
*Goal: Assemble the "SigLIP-Resampler-Gemma" architecture.*

3.  **The Vision Encoder (Frozen):**
    * Load `google/siglip2-large-patch16-512`.
    * **Freeze it immediately.** Do not train these weights. It is essentially your "feature extractor".

4.  **The Connector (Trainable):**
    * Implement the **Perceiver Resampler** based in Flammingo as we discussed.
    * **Input:** 1152-dim features (from SigLIP).
    * **Output:** 1024-dim vectors (to match Gemma 3 270M).
    * **Latents:** Set `num_latents = 256` (higher than standard captioning to preserve staccato/accidental details).

5.  **The Decoder (Trainable):**
    * Load `google/gemma-3-270m`.
    * **Config:** Since it is 270M, you can likely **full fine-tune** it (no LoRA needed) on a standard GPU (24GB VRAM should easily handle this batch size).

### **Phase 3: Tokenizer & Prompting**
*Goal: Teach Gemma to speak "Music".*

6.  **Modify the Tokenizer:**
    * Gemma’s tokenizer is multilingual, but you must add the **Legato special tokens** to the vocabulary so they are treated as single atomic units:
        * `<B>` (Begin ABC)
        * `<I>` (Image/Segment placeholder)
        * `<E>` (End ABC)
        * `$` (Line Break - crucial for structure)

7.  **Embeddings Resize:**
    * Call `model.resize_token_embeddings(len(tokenizer))` effectively to allocate space for these new tokens.

### **Phase 4: The Training Loop**
*Goal: Teach the model to predict the next ABC token.*

8.  **Data Loading:**
    * Your dataloader should yield a tuple: `(pixel_values, input_ids)`.
    * `pixel_values`: Shape `[Batch, 4, 3, 512, 512]` (The 4 patches).
    * `input_ids`: The tokenized ABC string, wrapped like:
        `<I> <I> <I> <I> <B> [ABC_TOKENS] <E>`.

9.  **Forward Pass & Loss:**
    * Pass images through SigLIP $\to$ Resampler.
    * Concatenate `[Visual_Embeddings, Text_Embeddings]`.
    * **Labeling:** Create a `labels` tensor. Set the "Visual" portion to `-100` (ignore index) so the model is not penalized for "predicting" the image. It should only learn to predict the ABC text.
    * Calculate **CrossEntropyLoss**.

### **Phase 5: Evaluation**
*Goal: Verify it works.*

10. **Inference pipeline:**
    * Feed the image patches.
    * Prompt the model with just `<I>...<I> <B>`.
    * Use **Greedy Decoding** or **Beam Search** (Beam size 3-10 recommended in the paper) to generate the ABC string until it hits `<E>`.

11. **Metric Check:**
    * Don't just rely on loss. Periodically render the output ABC back to an image (using `abcm2ps`) and visually compare it, or use the **TEDn** (Tree Edit Distance) metric mentioned in the paper if you want a rigorous benchmark.