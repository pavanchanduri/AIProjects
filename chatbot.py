import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

OPENAI_API_KEY = "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"
chunks = []

# Streamlit app for a PDF chatbot
# Upload PDF file
st.header("Chatbot")
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload your documents", type=["pdf"])

#Extract the text, create chunks, generate embeddings, and create a vector store
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Break in to chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # st.write(chunks)

    # generating embeddings
    open_ai_embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # create a vector store - FAISS
    vector_store = FAISS.from_texts(chunks, open_ai_embeddings)

    # read user query
    user_query = st.text_input("Ask a question about your documents")

    if user_query:
        # Get the most relevant chunks
        relevant_chunks = vector_store.similarity_search(user_query)

        # Display the relevant chunks
        st.write("Relevant Chunks:")

        # Define the LLM and the chain for question answering
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", # You can change the model to "gpt-4" if you have access
            # model="gpt-4", # Uncomment this line if you want to use GPT
            temperature=0, # Set temperature to 0 for deterministic output i.e., specific answers with less randomness
            max_tokens=200, # Set max tokens to limit the response length
            openai_api_key=OPENAI_API_KEY # Make sure to set your OpenAI API key here
        )
        chain = load_qa_chain(llm,type="stuff")
        # Run the chain with the relevant chunks and user query
        response = chain.run(question=user_query, input_documents=relevant_chunks)
        # Display the response
        st.write("Response:"+ response)
