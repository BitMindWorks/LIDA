# LIDA – Text Detection AI

**LIDA** is a language identification model that takes text input and predicts the language it is written in.

---

### 🔍 What LIDA Does

- **Text Input**: Accepts text as input.
- **Language Prediction**: Identifies the language of the provided text.
- **Returns**: The predicted language and probability distribution for each possible language.

---

### ⚙️ Tech Stack

- **Python** 🐍
- **PyTorch** for model implementation
- **JSON** for configuration and metadata storage

---

### 🚀 Quick Start

1. **Clone the repository**:

    ```bash
    git clone https://github.com/BitMindWorks/LIDA.git
    cd LIDA
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the script**

    ```python
    text = 'this is an english text'
    language, language_probablities = predict_language(text)
    print(language)
    print(language_probablities)
    ```

---

### 📦 Output

- **Predicted Language**: The language code (e.g., `en`, `fr`, `es`).
- **Language Probabilities**: The probability distribution for each language.

---

### Supported Language

| Language | Code |
|----------|----------|
| English | en |
| Bodo | brx |
| Spanish | es |
| Hindi | hi |
| French | fr |
| Bangla | bn |

---

### 📄 License

LIDA is open-source and available under the **MIT License**.

---

> Made with 💡 by MrLwitwma