from app import st
from app import add__logo
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from config import *


st.set_page_config(page_title="Data Sources",
                       page_icon="📊📈")
st.title("Add new Data sources")
add__logo()



# Display vectordb
def show_source(vector_db):
    v_dict = vector_db.docstore._dict
    data_rows =[]
    for k in v_dict.keys():
        doc_name = v_dict[k].metadata['source'].split("/")[-1]
        page_number = v_dict[k].metadata['page']+1
        content = v_dict[k].page_content
        data_rows.append({"Chunk_id": k,"document":doc_name, "page":page_number,"content":content})
    vector_df =pd.Dataframe(data_rows)
    return vector_df

embeddings = OpenAIEmbeddings()
vector_db = FAISS.load_local(VECTOR_DB_PATH, embeddings)
df = show_source(vector_db)
st.dataframe(df)


