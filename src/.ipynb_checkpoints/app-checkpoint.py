import os
import utils
import streamlit as st
from streaming import StreamHandler
import uuid
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredPDFLoader,UnstructuredPowerPointLoader,UnstructuredWordDocumentLoader,TextLoader,UnstructuredFileLoader
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings,OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import io
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd

st.set_page_config(page_title="ChatPDF", page_icon="📄")
st.header('Chat with your documents')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=13000,chunk_overlap=0,separators=["\n\n","\n"," ",""])

class CustomDataChatbot:
    
    def __init__(self):
        utils.configure_openai_api_key()
        self.openai_model = "gpt-3.5-turbo"
        self.resource_files_base_path = "../resources/savedfiles"

    def save_file(self, file,random_id):
        
        if not os.path.exists(self.resource_files_base_path):
            os.makedirs(self.resource_files_base_path)
        
        file_path = f'{self.resource_files_base_path}/random_id-{file.name}'
        
        with open(file_path, 'wb') as f:
            f.write(file.getvalue())
            
        return file_path
        
    def get_docs(self,uploaded_file):
        randomId = str(uuid.uuid1()).replace("-","")
        
        file_path = self.save_file(uploaded_file,randomId)
        
        file_extention = file_path.replace("/","").split(".")[-1]
        docs = []
        if file_extention == "txt":
            docs.extend(TextLoader(file_path=file_path,encoding="UTF-8").load_and_split(text_splitter))
        elif file_extention == "pdf":
            docs.extend(UnstructuredPDFLoader(file_path=file_path).load_and_split(text_splitter))
        elif file_extention == "pptx":
            docs.extend(UnstructuredPowerPointLoader(file_path=file_path).load_and_split(text_splitter))
        elif file_extention == "docx":
            docs.extend(UnstructuredWordDocumentLoader(file_path=file_path).load_and_split(text_splitter))
        else:
            docs = None
        return docs

    @st.spinner('Analyzing documents..')
    def setup_agent_for_csv_excel(self,uploaded_file,):
       
        
        llm = ChatOpenAI(model_name=self.openai_model, temperature=0.7, streaming=False)
        randomId = str(uuid.uuid1()).replace("-","")
        file_path = self.save_file(uploaded_file,randomId)
        file_extention = file_path.replace("/","").split(".")[-1]
        if file_extention != "csv" and  file_extention != "xlsx":
            return
        df = None
        if file_extention == "xlsx":
            df = pd.read_excel(file_path,sheet_name=0)
        elif file_extention == "csv":
            df = pd.read_csv(file_path)
        if df is None:
            return
        agent = create_pandas_dataframe_agent(df=df,llm=llm,handle_parsing_errors=True,verbose=False)
        
        return agent
        
    @st.spinner('Analyzing documents..')
    def setup_qa_chain(self, uploaded_file):

    
        splits = self.get_docs(uploaded_file)
        if  splits is None:
            return
        # Create embeddings and store in vectordb
        vectordb =  FAISS.from_documents(documents=splits,embedding=OpenAIEmbeddings())

        # Define retriever
        retriever = vectordb.as_retriever(
            search_type='mmr',
            search_kwargs={'k':20, 'fetch_k':20}
        )

        # Setup memory for contextual conversation        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )

        # Setup LLM and QA chain
        llm = ChatOpenAI(model_name=self.openai_model, temperature=0.7, streaming=True)
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=False,chain_type="map_reduce")
        return qa_chain

    @utils.enable_chat_history
    def main(self):

        # User Inputs
        uploaded_file = st.sidebar.file_uploader(label='Upload files', type=["csv", "xlsx", "pdf", "txt","pptx","docx"], accept_multiple_files=False)
        if not uploaded_file:
            st.error("Please upload documents to continue!")
            st.stop()

        user_query = st.chat_input(placeholder="Ask me anything!")

        if uploaded_file and user_query:
            qa_chain = self.setup_qa_chain(uploaded_file)
            agent = self.setup_agent_for_csv_excel(uploaded_file)

            utils.display_msg(user_query, 'user')

            with st.chat_message("assistant"):
                if agent is not None:
            
                    container = st.empty()
                    try:
                       response =  agent.run(user_query)
                       container.markdown(response)
                       st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as error:
                        container.error("Error occured: "+str(error))
                else:
                    st_cb = StreamHandler(st.empty())
                    response = qa_chain.run(user_query, callbacks=[st_cb])
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    obj = CustomDataChatbot()
    obj.main()