🧠 Text Generation Transformer (From Scratch + MLOps)
=====================================================

A minimal, production-structured decoder-only Transformer built in PyTorch and trained on WikiText-2.

This project is designed to:

*   Deepen understanding of Transformer mechanics
    
*   Implement clean ML engineering practices
    
*   Apply reproducible data pipelines
    
*   Incrementally introduce MLOps discipline
    

📌 Project Goals
----------------

*   Build a decoder-only Transformer from scratch (using PyTorch primitives)
    
*   Train using next-token prediction objective
    
*   Implement structured data pipeline (raw → processed)
    
*   Ensure reproducibility
    
*   Prepare foundation for experiment tracking and deployment
    

📂 Project Structure
--------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   textgen-mlops/  │  ├── data/  │   ├── raw/           # Immutable source dataset  │   ├── processed/     # Tokenized tensors (.pt files)  │  ├── src/  │   ├── config/  │   │   └── config.yaml  │   ├── data/  │   │   └── dataset.py  │   ├── models/  │   ├── training/  │   ├── inference/  │   ├── utils/  │   │   └── seed.py  │  ├── tests/  ├── api/  ├── docker/  ├── requirements.txt  └── README.md   `

📚 Dataset
==========

We use:

**WikiText-2 (wikitext-2-raw-v1)**A standard language modeling benchmark dataset.

### Data Pipeline

On first run:

1.  Dataset is downloaded using HuggingFace datasets
    
2.  Saved to data/raw/
    
3.  Tokenized using GPT-2 tokenizer
    
4.  Token tensors saved to data/processed/
    

On subsequent runs:

*   Raw dataset loaded from disk
    
*   Tokenized tensors loaded directly (no reprocessing)
    

This ensures:

*   Reproducibility
    
*   Faster iteration
    
*   Clean raw vs processed separation
    

🔤 Tokenization Strategy
========================

We use the GPT-2 tokenizer.

Each document is tokenized individually and concatenated into a single continuous token stream.

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   self.input_ids = torch.cat(all_input_ids, dim=0)   `

This produces a long 1D tensor:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   [t1, t2, t3, ..., tN]   `

🧮 Language Modeling Objective
==============================

We train using **causal next-token prediction**.

For a sequence:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   [t1, t2, t3, t4]   `

Input (x):

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   [t1, t2, t3, t4]   `

Target (y):

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   [t2, t3, t4, t5]   `

This is implemented as:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   x = input_ids[start:end]  y = input_ids[start + 1:end + 1]   `

Mathematical Formulation
------------------------

Given a token sequence:

x=(x1,x2,...,xT)x = (x\_1, x\_2, ..., x\_T)x=(x1​,x2​,...,xT​)

The model is trained to maximize:

∏t=1TP(xt+1∣x1,...,xt)\\prod\_{t=1}^{T} P(x\_{t+1} \\mid x\_1, ..., x\_t)t=1∏T​P(xt+1​∣x1​,...,xt​)

Loss function used:

L=−∑t=1Tlog⁡P(xt+1∣x≤t)\\mathcal{L} = - \\sum\_{t=1}^{T} \\log P(x\_{t+1} \\mid x\_{\\leq t})L=−t=1∑T​logP(xt+1​∣x≤t​)

This is equivalent to **Cross-Entropy Loss** over next-token predictions.

📦 Dataset Construction
=======================

Sequences are chunked into fixed-length blocks:

If:

*   Total tokens = NNN
    
*   Sequence length = LLL
    

Then:

num\_sequences=⌊N−1L⌋\\text{num\\\_sequences} = \\left\\lfloor \\frac{N - 1}{L} \\right\\rfloornum\_sequences=⌊LN−1​⌋

This ensures valid shifted targets.

Chunks are non-overlapping:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   [0:L]  [L:2L]  [2L:3L]  ...   `

This matches standard GPT-style training.

🔁 Reproducibility
==================

We fix all major randomness sources:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   random.seed(seed)  np.random.seed(seed)  torch.manual_seed(seed)  torch.cuda.manual_seed_all(seed)   `

This ensures consistent:

*   Weight initialization
    
*   Data shuffling
    
*   Dropout behavior (as much as possible)
    

Reproducibility is critical for ML system reliability.

⚙️ Configuration Management
===========================

Hyperparameters are stored in:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   src/config/config.yaml   `

Example:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   data:    dataset_name: wikitext    dataset_config: wikitext-2-raw-v1    seq_length: 128  training:    batch_size: 32   `

No hardcoded magic numbers inside training code.

🚀 Current Status (End of Day 1)
================================

✅ Raw dataset persistence✅ Tokenization pipeline✅ Processed tensor caching✅ Fixed-length sequence chunking✅ Shifted next-token targets✅ Reproducibility setup✅ Config-driven structure