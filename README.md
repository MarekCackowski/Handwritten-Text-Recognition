# CRNN-CN-TR-OCR
## High-Precision Transcription System for Challenging Manuscripts

This system is a solution designed for the high-precision transcription of complex historical (and modern as well) manuscripts. The core architecture is built upon a **triple-verification cascade**, integrating deep sequential learning, geometric morphological analysis, and byte-level semantic correction. 

Optimized for the **Overnight Precision** workflow, the system prioritizes absolute character accuracy and reliability over raw processing speed, making it an ideal tool for archival digitalization and academic research.

## Architecture and Technological Pillars

### Physical Layer: Optimal Seam Carving
To mitigate errors caused by overlapping lines (intersecting ascenders and descenders), the system employs a cost-graph-based seam carving algorithm. 
* **Mechanism:** It determines **non-linear seams** to separate lines, ensuring the total ink continuity of every word is preserved even in densely written manuscripts.

### Vision Foundation: Bayesian CRNN and MC Dropout
The core of the system is a **ResNet-BiGRU** model, enhanced with statistical uncertainty estimation via the **Monte Carlo Dropout** method.

* **Dual-Mode Logic:** The system supports two operational modes depending on the precision-speed trade-off:
    * **Standard Mode:** Performs a single forward pass for maximum throughput during real-time interaction.
    * **Overnight Precision Mode:** Executes **64 stochastic passes** for each word with the Dropout mechanism active to generate a robust probability distribution.
* **Bayesian Uncertainty:** Based on the variance of the stochastic results, the predictive uncertainty $U(x)$ is calculated. This acts as a high-precision detector for character-level anomalies and segmentation errors.

$$U(x) \approx \frac{1}{T} \sum_{t=1}^{T} (\hat{p}_t - \bar{p})^2$$

### Geometric Expert: Deep Capsule Network (CapsNet)
High-variance regions (flagged as "uncertain") are routed to a **Capsule Network** for detailed morphological verification.
* **Dynamic Routing:** In "Overnight" mode, the system utilizes 9 routing iterations to analyze hierarchical part-whole relationships (e.g., stroke slants and ligatures).
* **Objective:** CapsNet excels at disambiguating visually similar clusters (e.g., *m* vs *nn*, *u* vs *n*) that standard CNNs often fail to distinguish.

### Semantic Refiner: ByT5-Base Transformer
The final refinement stage is handled by a **ByT5-Base Transformer** operating at the byte level, providing high resistance to spelling errors and out-of-vocabulary characters.
* **Contextual Awareness:** A *Contextual Sliding Window* mechanism feeds the context of previous lines into the current word analysis.
* **Beam Search:** A width of **k=20** explores a broad linguistic probability tree, crucial for automatically splitting physically joined words.

### Adaptive Personalization: Active Learning Loop
The system implements a continuous learning mechanism based on real-time user feedback.
* **Style Adaptation:** Corrections made by the user in the GUI are automatically converted into new training pairs.
* **Incremental Fine-Tuning:** The model undergoes micro-calibration cycles, progressively lowering the Error Rate (CER) as it adapts to the unique calligraphy of a specific author.

---

## "Overnight Precision" Mode Characteristics
The system offers increased precision through intensive statistical sampling. This extends inference time but significantly minimizes the requirement for manual human correction.

| Parameter | Standard Mode | Overnight Precision |
| :--- | :--- | :--- |
| MC Dropout Sampling | 1 pass | 64 passes |
| CapsNet Routing | 3 iterations | 9 iterations |
| Beam Search Width | 3 candidates | 20 candidates |
| **Correction Time** | ~15 sec / page | **< 3 min / page** |
| **Target CER** | ~3.5% | **~2.5%** |

---

## Performance Metrics After Adaptation
After a full adaptation cycle to a specific handwriting style, the system achieves the following benchmarks:

* **CER (Character Error Rate):** ~2.5%
* **WER (Word Error Rate):** ~6.0%

The CER is calculated based on the Levenshtein distance:
$$CER = \frac{S + D + I}{N}$$
Where $S$ represents substitutions, $D$ deletions, $I$ insertions, and $N$ is the total number of characters in the ground truth.

---

## Process Stability and Reliability
Given the long-duration nature of processing tasks, the system includes built-in safety mechanisms:
* **Robust Logging:** Immediate saving of results to CSV/JSON files after every processed page.
* **Inference Optimization:** Models are exported to **ONNX** format, achieving a 3x increase in inference speed and lower VRAM/RAM consumption.
* **Fault Tolerance:** Automatic skipping of corrupted image files without interrupting the session.
* **VRAM Management:** Frequent clearing of GPU memory (`torch.cuda.empty_cache()`) to prevent Out-Of-Memory errors during heavy Bayesian sampling.

---

## Tech Stack
* **Core AI:** PyTorch, TensorFlow, ByT5 (Transformer), CapsNet
* **Optimization:** ONNX Runtime, SWA (Stochastic Weight Averaging), EMA
* **Computer Vision:** OpenCV, Albumentations, NumPy
* **App/Backend:** FastAPI, PyQt5, TensorBoard
