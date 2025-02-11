# GenAI Demo - AI-Powered Document Summarization

## 📌 Introduction
GenAI Demo is a **Generative AI-based** solution that efficiently summarizes documents using **Oracle Cloud Infrastructure (OCI)** and large language models (LLMs). Designed to facilitate the processing of lengthy texts, this project is ideal for businesses and professionals needing automated key information extraction.

## ⚡ Key Features
- 🔥 **Summarizes long documents** in seconds.
- 🧠 **Leverages LLMs** to understand and synthesize information.
- 📄 **Supports multiple document formats (PDF, docx, text, xls)**.
- 🛠 **Simple and modular interface** for easy integration.

## Pre-requisites
1. Oracle Cloud account - [Portal link](https://signup.cloud.oracle.com/)
2. OCI Generative AI - [Python SDK](https://pypi.org/project/oci/)
3. No connected to VPN


## 🚀 Installation & Usage

### Clone the repository
```bash
git clone https://github.com/davidbarbosat97/docSummarizer.git
cd docSummarizer
```
### Running the application
Open docSummarizer folder in CLI
```bash
cd docSummarizer/
```
Install Langchain & OCI SDK
```bash
pip install -U langchain oci
```
Install requirements
```bash
pip install -r requirements. txt
```
Create your .oci folder and its components (if never created yet). You will need to download the private key from OCI
```bash
mkdir .oci
cd .oci/
vi config
...
[DEFAULT]
user=ocid1.us......
fingerprint=fd...:9b
key_file=./.oci/oci_api_key.pem
tenancy=ocid1.tenancy.oc1...
region=us-chicago-1
...

touch oci_key.pem
```
Run the app (copy your OCI Compartment ID)
```bash
streamlit run docSummarizer_Chat_model.py
```
