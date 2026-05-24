# CRNN-CN-TR-OCR
## Transcription System for Handwritten Texts

This system is a solution designed for the high-precision transcription of complex historical (and modern as well) manuscripts. The core architecture is built upon a **triple-verification cascade**, integrating deep sequential learning, geometric morphological analysis, and byte-level semantic correction. 

Optimized for the **Overnight Precision** workflow, the system prioritizes absolute character accuracy and reliability over raw processing speed, making it an ideal tool for archival digitalization.

## Architecture and Technological Pillars

### Physical Layer: Optimal Seam Carving
To mitigate errors caused by overlapping lines (intersecting ascenders and descenders), the system employs a cost-graph-based seam carving algorithm. 
* **Mechanism:** It determines **non-linear seams** to separate lines, ensuring the total ink continuity of every word is preserved even in densely written manuscripts.

### Vision Foundation: CRNN and Confidence Scoring
The core of the system is a **ResNet-BiGRU** model, optimized for accurate sequential feature extraction.
* **Uncertainty Detection:** The predictive uncertainty is calculated directly from the network's softmax output probabilities. Drops in character-level confidence scores act as a high-precision detector for potential anomalies and segmentation errors.

### Geometric Expert: Deep Capsule Network
Low-confidence regions (flagged as "uncertain" by the CRNN) are routed to a **Capsule Network** for detailed morphological verification.
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
The system offers increased precision through intensive morphological and semantic analysis. This extends inference time but significantly minimizes the requirement for manual human correction.

| Parameter | Standard Mode | Overnight Precision |
| :--- | :--- | :--- |
| CapsNet Routing | 3 iterations | 9 iterations |
| Beam Search Width | 3 candidates | 20 candidates |

---

## Interactive Local UI 
The system is optimized for direct execution on a personal computer workstation, avoiding external cloud dependencies to ensure maximum control and data privacy. 
* **Visual Diagnostics:** The graphical user interface explicitly highlights uncertain fragments (entire words or specific parts of them) within colored frames on the manuscript image. 
* **Pipeline Inspection:** By hovering the cursor over any framed word, the user triggers a detailed tooltip displaying the exact progression of the transcription through the architecture:
  * `crnn result:`
  * `crnn+capsnet result:`
  * `final result:` (after Transformer semantic correction)

---

## Process Stability and Reliability
Given the long-duration nature of processing tasks, the system includes built-in safety mechanisms:
* **Robust Logging:** Immediate saving of results to CSV/JSON files after every processed page.
* **Inference Optimization:** Models are exported to **ONNX** format for faster inference.
* **Fault Tolerance:** Automatic skipping of corrupted image files without interrupting the session.
* **VRAM Management:** Frequent clearing of GPU memory (`torch.cuda.empty_cache()`) to prevent Out-Of-Memory errors during deep routing and broad beam search iterations.

---

## Tech Stack
* **Core AI:** PyTorch, TensorFlow, CRNN, CapsNet, ByT5 (Transformer)
* **Optimization:** ONNX Runtime, SWA (Stochastic Weight Averaging), EMA
* **Computer Vision:** OpenCV, Albumentations, NumPy
* **App/Backend:** FastAPI, PyQt5, TensorBoard
