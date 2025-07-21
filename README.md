# eMoE-TinyLlama: Expert Mixture from TinyLlama

Transform TinyLlama into a Mixture of Experts (MoE) model by clustering feedforward neurons into specialized expert groups, improving performance without adding parameters.

## Project Structure

```
eMoE-tinyllama/
├── modal/
│   ├── emoe.py           # Main clustering implementation
│   ├── requirements.txt  # Python dependencies
│   └── model.md         # Model documentation
└── README.md            # This file
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd eMoE-tinyllama
   ```

2. **Install Modal CLI:**
   ```bash
   pip install modal
   modal setup
   ```

3. **Install dependencies:**
   ```bash
   cd modal
   pip install -r requirements.txt
   ```