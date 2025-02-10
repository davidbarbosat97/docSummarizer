
import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
#from langchain_community.llms import OCIGenAI
from langchain_community.chat_models import ChatOCIGenAI
from langchain_core.messages import HumanMessage
from pypdf import PdfReader
from io import BytesIO
from typing import Any, Dict, List
import re
from langchain.docstore.document import Document


AUTH_TYPE = "API_KEY" # The authentication type to use, e.g., API_KEY (default), SECURITY_TOKEN, INSTANCE_PRINCIPAL, RESOURCE_PRINCIPAL.
CONFIG_PROFILE = "DEFAULT"


## Convert pdf
@st.cache_data
def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output


## Convert .txt
@st.cache_data
def parse_txt(file: BytesIO) -> List[str]:
    text = file.read().decode("utf-8")  # Leer y decodificar el contenido del archivo
    # Merge hyphenated words
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Fix newlines in the middle of sentences
    text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    
    return text.split("\n\n")  # Devuelve lista de párrafos separados por doble salto de línea


## Convert .py
@st.cache_data
def parse_py(file: BytesIO) -> List[str]:
    text = file.read().decode("utf-8")  # Leer el archivo .py como texto
    
    # Remover espacios innecesarios al final de cada línea
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    
    # Eliminar líneas en blanco excesivas (más de dos seguidas)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.split("\n\n")  # Devuelve una lista de bloques de código separados por doble salto de línea



### Print chunks of the doc
@st.cache_data
def text_to_docs(text: str,chunk_size,chunk_overlap) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""

    print(chunk_size)
    print(chunk_overlap)
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=chunk_overlap,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Ansh Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks


def custom_summary(docs, llm, custom_prompt, chain_type, num_summaries):
 
    print("I am inside custom summary")
    custom_prompt = custom_prompt + """:\n {text}"""
    print("user custom Prompt is ------>")
    print(custom_prompt)
    COMBINE_PROMPT = PromptTemplate(template=custom_prompt, input_variables = ["text"])
    print("user combine Prompt is ------>")
    print(COMBINE_PROMPT)
    MAP_PROMPT = PromptTemplate(template="Summarize:\n{text}", input_variables=["text"])
    print("user MAP_PROMPT Prompt is ------>")
    print(MAP_PROMPT)
    if chain_type == "map_reduce":
        chain = load_summarize_chain(llm,chain_type=chain_type,
                                     map_prompt=MAP_PROMPT,
                                     combine_prompt=COMBINE_PROMPT)
    else:
        chain = load_summarize_chain(llm,chain_type=chain_type)
    print("Chain is --->")
    print(chain)
    summaries = []
    for i in range(num_summaries):
        summary_output = chain({"input_documents": docs}, return_only_outputs=True)["output_text"]
        print("Summaries------------->")
        print(summary_output)
        summaries.append(summary_output)
    
    return summaries


def main():

    ## Set the layouts
    st.set_page_config(layout="wide")
    hide_streamlit_style = """
            <style>
            [data-testid="stToolbar"] {visibility: hidden !important;}
            footer {visibility: hidden !important;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    st.title("Document Summarization App")
    st.subheader("with OCI models")
    logo_url = './oracle_logo.png'
    st.sidebar.image(logo_url)
    
    llm_name = st.sidebar.selectbox("LLM",["cohere.command-r-08-2024",
        "cohere.command-r-plus-08-2024", 
        "meta.llama-3.1-405b-instruct", 
        "meta.llama-3.1-70b-instruct",
        "meta.llama-3.2-90b-vision-instruct",
        "meta.llama-3.3-70b-instruct"])
    
    chain_type = st.sidebar.selectbox("Chain Type", ["map_reduce", "stuff", "refine","map_rerank"])
    chunk_size = st.sidebar.slider("Chunk Size", min_value=20, max_value = 5000,
                                   step=10, value=2000)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=5, max_value = 5000,
                                   step=10, value=200) 
    
    #top_p = st.sidebar.slider("Top p", min_value = 0.0,
                                              #max_value=1.0,
                                              #step=0.05,
                                              #value=0.75) 
    #top_k = st.sidebar.slider("Top k", min_value = 0.0,
                                              #max_value=1.0,
                                              #step=0.05,
                                              #value=0.0)        
    user_prompt = st.text_input("Enter the document summary prompt:", value= "Could you please give a brief about the following document? ")
    temperature = st.sidebar.number_input("Set the GenAI Temperature",
                                              min_value = 0.0,
                                              max_value=1.0,
                                              step=0.1,
                                              value=0.5)
    max_token = st.sidebar.slider("Max Token size", min_value=400, max_value = 4000,step=10, value=600) 
    compartment_id = st.sidebar.text_input("Enter the OCI compartment id", value= "")

                                             
    opt = "Upload-own-file"
    pages = None
    if opt == "Upload-own-file":
        uploaded_file = st.file_uploader(
        "**Upload a Pdf, txt or python file :**",
            type=["pdf", "txt", "py"],
            )
        if uploaded_file:
            if uploaded_file.name.endswith(".txt"):
                doc = parse_txt(uploaded_file)
            elif uploaded_file.name.endswith(".pdf"):
                doc = parse_pdf(uploaded_file)
            elif uploaded_file.name.endswith(".py"):
                doc = parse_py(uploaded_file)  # Llama a la nueva función para archivos .py
                
            pages = text_to_docs(doc, chunk_size, chunk_overlap)
            print("Pages are here")
            print(pages)


            page_holder = st.empty()
            if pages:
                print("Inside if PAges")
                st.write("PDF loaded successfully")
                with page_holder.expander("File Content", expanded=False):
                    pages

        ## Call GenAI model 
                llm = ChatOCIGenAI(
    model_id=llm_name,
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id = compartment_id,
    model_kwargs={
        "temperature": temperature,
        "max_tokens": max_token,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "top_p": 0.75,
        "top_k": 0,
        "seed": None
      },
      auth_type=AUTH_TYPE,
      auth_profile=CONFIG_PROFILE
)
                messages = [
      HumanMessage(content=f"{user_prompt}\n\nDocument: {pages}"),
    ]

                #response = llm.invoke(messages)

                if st.button("Summarize"):
                    with st.spinner('Summarizing....'):
                        result = custom_summary(pages, llm, user_prompt, chain_type, 1)
                        st.write("Summary:")
                    for summary in result:
                        st.write(summary)
            else:
                st.warning("No file found. Upload a file to summarize!")
            
if __name__=="__main__":
    main()




