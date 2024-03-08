import os
import streamlit as st
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


PATH = "temp"


class MedicalAssistantApp:
    def __init__(self, storage_path="temp"):
        self.storage_path = storage_path
        self.openai_api_key = st.secrets["OPENAI_API_KEY"]
        self.prompt_template = "As a medical assistant, explain the key points and implications of {context} to an individual in their 50s or 60s. Ensure your explanation is brief, understandable, and empathetic, avoiding technical jargon as much as possible. Emphasize the main findings and what they mean for the individual's well-being, while providing reassurance and encouraging any questions for further clarity."

    def setup(self):
        """Set up the Streamlit UI components."""
        st.title("Medical Assistant")
        uploaded_files = st.file_uploader(
            "Upload patient reports", type=["txt"], accept_multiple_files=True
        )
        return uploaded_files

    def process_uploaded_files(self, uploaded_files):
        """Process and store uploaded files."""
        if uploaded_files is not None:
            self._ensure_directory(self.storage_path)
            for uploaded_file in uploaded_files:
                self._save_uploaded_file(uploaded_file)

            loader = self._initialize_loader()
            docs = loader.load()
            self._cleanup_files(uploaded_files)
            return docs
        else:
            st.info("Please upload patient reports to proceed.")
            return []

    def analyze_documents(self, docs):
        """Analyze the documents using a LLM chain."""
        if docs:
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=self.openai_api_key)
            prompt = ChatPromptTemplate.from_messages(
                [("system", self.prompt_template)]
            )
            chain = create_stuff_documents_chain(llm, prompt)

            with st.spinner("Processing..."):
                response = chain.invoke({"context": docs})
                st.write(response)

    @staticmethod
    def _ensure_directory(path):
        """Ensure the directory exists."""
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def _save_uploaded_file(uploaded_file):
        """Save the uploaded file to the filesystem."""
        file_path = os.path.join(PATH, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' Uploaded Successfully.")

    @staticmethod
    def _initialize_loader():
        """Initialize the document loader."""
        text_loader_kwargs = {"autodetect_encoding": True}
        loader = DirectoryLoader(
            PATH,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs=text_loader_kwargs,
        )
        return loader

    @staticmethod
    def _cleanup_files(uploaded_files):
        """Remove processed files and their directory."""
        for uploaded_file in uploaded_files:
            os.remove(os.path.join(PATH, uploaded_file.name))
        os.rmdir(PATH)


def main():
    app = MedicalAssistantApp()
    uploaded_files = app.setup()
    docs = app.process_uploaded_files(uploaded_files)
    app.analyze_documents(docs)


if __name__ == "__main__":
    main()
