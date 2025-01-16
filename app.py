import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
import os
import requests
from io import StringIO
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is loaded
if not api_key:
    st.error("OpenAI API key not found. Please add it to your .env file.")
    st.stop()

# Streamlit UI
st.title("Document Query")

input_type = st.selectbox("Choose input type", ["PDF file", "Text file", "URL"])

raw_text = ''
page_texts = []  # To store text and its corresponding page number

if input_type == "PDF file":
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        # Read PDF file
        pdfreader = PdfReader(uploaded_file)
        # Extract text from PDF
        for i, page in enumerate(pdfreader.pages):
            content = page.extract_text()
            if content:
                raw_text += content
                page_texts.append({"page_number": i + 1, "text": content})  # Store page number and text


elif input_type == "Text file":
    uploaded_file = st.file_uploader("Choose a Text file", type="txt")
    if uploaded_file is not None:
        # Read text file
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        raw_text = stringio.read()

elif input_type == "URL":
    url = st.text_input("Enter URL")
    if url:
        response = requests.get(url)
        if response.status_code == 200:
            raw_text = response.text
        else:
            st.error("Failed to fetch content from URL")

if raw_text:
    # Split text using character text splitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )

    texts = text_splitter.split_text(raw_text)

    # Create OpenAIEmbeddings object
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # Create FAISS vector store
    document_search = FAISS.from_texts(texts, embeddings)
    
    # Add QA Chain with concise responses
    concise_prompt_template = """
    You are a helpful assistant. Use the provided context to answer the question with a single, concise value.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    concise_prompt = PromptTemplate(
        template=concise_prompt_template, 
        input_variables=["context", "question"]
    )

    llm = OpenAI(api_key=api_key, temperature=0, max_tokens=15)
    llm_chain = LLMChain(llm=llm, prompt=concise_prompt)

    # Create StuffDocumentsChain explicitly
    chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")


    # Load queries from CSV file
    queries_csv_path = "/home/seaflux/Documents/Document-Query-Q-A_Quotey/questions_generated.csv"  # Ensure this path is correct
    try:
        query_df = pd.read_csv(queries_csv_path)
        if "Query" in query_df.columns:
            queries = query_df["Query"].tolist()
            # Create a dictionary to map queries to coverage_id
            query_to_id = dict(zip(query_df["Query"], query_df["ID"]))
        else:
            st.error("The CSV file must have a 'Query' column.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading queries from CSV: {e}")
        st.stop()

    responses = []

    for query in queries:
        docs = document_search.similarity_search(query)
        response = chain.run(input_documents=docs, question=query)
        
        # Get the coverage_id for the current query
        coverage_id = query_to_id.get(query, "N/A")  # Default to "N/A" if not found
        
        # Now, we also include the page number(s) where the response was found
        page_numbers = set()  # Use set to avoid duplicates
        for doc in docs:
            for page_text in page_texts:
                # Compare doc with each page's text chunk
                if doc.page_content.strip() in page_text["text"].strip():  # Strip any extra spaces or newlines
                    page_numbers.add(page_text["page_number"])

        # Store the response along with page numbers
        responses.append({
            # "Query": query, 
            "Response": response, 
            "coverage_id": coverage_id, 
            "Page Numbers": ', '.join(map(str, page_numbers))  # Store as a string of page numbers
        })

    # Convert responses to a DataFrame
    df = pd.DataFrame(responses)

    # Display the responses
    st.markdown("### Responses")
    # st.write(df)

    # Download CSV file
    # st.markdown("### Download Results")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name="query_responses.csv",
        mime="text/csv",
    )