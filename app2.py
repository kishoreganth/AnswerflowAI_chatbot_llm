import streamlit as st
import streamlit_tags as st_tag
import openai
from langchain.llms import OpenAI
from langchain.callbacks.streamlit import StreamlitCallbackHandler

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from ui import css, bot_template, user_template
from langchain.chains import RetrievalQA
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from config import *
from dotenv import load_dotenv, find_dotenv
import os
from knowledge_base import *
from streamlit_chat import message
from usp.tree import sitemap_tree_for_homepage





def get_conversation_chain(vectorstore):
    # llm = OpenAI(model_name = "gpt-3.5-turbo-0613")model_name="gpt-3.5-turbo"
    llm = OpenAI(model_name="gpt-4")
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, respond only in spanish language.

    Chat History:
    {chat_history}
    Context:
    {context}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(template=_template)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        # return_source_documents = True,
        condense_question_prompt= CONDENSE_QUESTION_PROMPT


    )
    return conversation_chain

def get_source_doc(llm_response):
    print(llm_response["result"])
    print("\nSources: ")
    source = []
    for s in llm_response["source_documents"]:
        print(s.metadata["source"])
        source.append(s.metadata["source"])
    return set(source)

def display_conversation(history):
    for i in range(len(history["assistant"])):
        message(history["user"][i], is_user=True)
        message(history["assistant"][i])
# , key=str(i) + "_user"
@st.cache_data(show_spinner=True)

def handle_userinput(prompt):
    # prompt = "Act as donald trump and asnwer like how donald trump would with the information provided for the given query : "
    # final_query = prompt + user_question
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        st.write(st.session_state.messages[-1]["content"])
        prompt = st.session_state.messages[-1]["content"]
        response = st.session_state.conversation({'query': prompt})
        source = get_source_doc(response)

    return response


    # for i, message in enumerate(st.session_state.chat_history):
    #     if i % 2 == 0:
    #         st.write(user_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)
    #     else:
    #         st.write(bot_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)


# --- FUNCTION TO ADD LOGO with SIDEBAR -----------
def add__logo():
    # no params
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                # background-image: url(http://placekitten.com/200/200);
                background-image: url("https://ideogram.ai/api/images/direct/rKXLrHwCQ6CuVX7mfvB_yA");
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 20px 20px;
                background-size: 150px 150px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "Answerflow ai";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 75px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# -- MAIN FUNCTION -- START OF THE PROGRAM
def main():
    # LOAD ENV VARIABLE  for OPENAI_API_KEY
    _ = load_dotenv(find_dotenv())
    openai.api_key = os.environ["OPENAI_API_KEY"]

    # PAGE CONFIGURATION OF STREAMLIT
    st.set_page_config(page_title="Chat with Company Data",
                       page_icon=":books:")
    st.title(":books:| 🦜 Chat with Data")
    st.write(css, unsafe_allow_html=True)

    add__logo()

    global qa
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "messages" not in st.session_state or st.sidebar.button("Clear conversation history"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    pdf_files = st.file_uploader(
        "Upload your PDFs here and click on 'Process'",
        type ="PDF",
        accept_multiple_files=True
    )

    URLs = st_tag.st_tags(
        label='Enter Web URLs   (max 4)',
        text='Press enter to add URLs',
        value=[],
        maxtags=4,
        key='2')

    if st.button("Process"):
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain_type_kwargs = {"prompt": PROMPT}
        if not URLs and not pdf_files:
            st.write("Upload a file or URL")

            # vectordb =get_vector()
            # qa = RetrievalQA.from_chain_type(
            #     llm=OpenAI(),
            #
            #     chain_type="stuff",
            #     retriever=vectordb.as_retriever(),
            #     return_source_documents=True,
            #     chain_type_kwargs=chain_type_kwargs
            # )
            # st.session_state.conversation = qa
            # st.write("No input data given to train the data. We will use existing data")
        else:
            dict_url = []
            for i in range(len(URLs)):
                sub_url = []
                tree = sitemap_tree_for_homepage(URLs[i])
                for page in tree.all_pages():
                    sub_url.append(page.url)
                dict_url.extend(sub_url)
            # st.write(dict_url)
            # final = []
            # for i in range(len(dict_url)):
            #     st.write(dict_url[i])
            vector_store = get_knowledge_from_pdfs(pdf_files, dict_url)

            if vector_store != 0:
                llm = ChatOpenAI(
                    temperature=0, model="gpt-4", openai_api_key=openai.api_key, streaming=True
                )
                qa = RetrievalQA.from_chain_type(
                    llm=llm,

                    chain_type="stuff",
                    retriever=vector_store.as_retriever(),
                    return_source_documents=True,
                    chain_type_kwargs=chain_type_kwargs,

                )
                st.session_state.conversation = qa
            else:
                st.write(" Please upload files")

            if qa:
                st.write("Chatbot ready, proceed to chat ✅")
            else:
                st.write("Chatbot is not ready, can you please retry")



    # UserINPUT handling -------------
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    if prompt:=st.chat_input(placeholder="Ask your dataset"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        # st.write(st.session_state.messages[-1]["content"])

        with get_openai_callback() as cb:

            # response = handle_userinput(prompt)
            # message(response["result"], is_user=False, key="ai")
            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                # st.write(st.session_state.messages[-1]["content"])

                # st.write(st.session_state.messages[-1]["content"])
                # prompt = st.session_state.messages[-1]["content"]
                response = st.session_state.conversation({'query': prompt})
                source = get_source_doc(response)
                # st.chat_message("assistant").write(response["result"])

                st.write(response["result"])
                st.session_state.messages.append({"role": "assistant", "content": response["result"]})

            print(cb)
            print(cb.prompt_tokens)




if __name__ =="__main__":
    main()
