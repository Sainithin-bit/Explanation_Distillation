# DriveXplain: Label and Explanation Generator

This repository provides tools to automatically generate **labels** and **label + explanation pairs** from driving scene data. It is designed to support projects that involve classification, reasoning, or high-level understanding of driving behavior.

---

## 🛠️ Step-by-Step Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/DriveXplain.git
cd DriveXplain
```

### 2. Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Label + Explanation Generator

There are **two main scripts** you can run:

### 🔹 A. Run Explanation Classification (for label + explanation generation)

```bash
python DriveXplain_Classification.py
```

* ✅ Generates labels and their corresponding explanations.
* ✅ Uses a classification-based approach (e.g., based on trajectories, motion, or semantic cues).

---

### 🔹 B. Run `DriveXplain_Explanation_Generation.py` (for **both** labels and explanations)

```bash
python DriveXplain_Explanation_Generation.py
```

* ✅ End-to-end script that performs:

  * Label classification
  * Semantic and motion parsing
  * Explanation generation using structured or LLM-based reasoning

---

## 📂 Output

Output files will be saved in the `output/` directory in JSON format:

### Example Format

```json
{
  "video_id": {
    "label": "left turn",
    "explanation": "The vehicle is turning left at the intersection due to the road layout."
  },
  "video_id": {
    "label": "forward",
    "explanation": "The vehicle continues straight as there are no turns or obstructions."
  }
}
```

---

## 📌 Notes

* Input data (e.g., images, optical flow, segmentations) should be placed in the `input/` directory or paths specified in the script configs.
* Ensure your file paths in the scripts are correct before execution.

---

## 📄 License

MIT License. See the `LICENSE` file for more details.

---

## 👨‍💻 Maintainer

**Sainithin Artham**
📧 [sainithin.artham@gmail.com](mailto:sainithin.artham@gmail.com)


```

---

Let me know if you'd like to:
- Include example input files
- Add visual output (e.g., overlaid frames)
- Package the project with CLI support or Jupyter notebooks

I can help format that too.
```
