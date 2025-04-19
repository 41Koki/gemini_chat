import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import fitz


# 同一フォルダにある.envファイルを読み込む
load_dotenv()
# モデルはGemini-1.5-flashを指定
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# pdfファイルを読み込み、検索可能なナレッジベースを作成する関数
def create_knowledge_base(file_path):
    """
    テキストファイルを読み込み、検索可能なナレッジベースを作成
    """
    with fitz.open(file_path) as doc:
        raw_text = "\n".join(page.get_text() for page in doc) # PDFのテキストを取得
    # テキストをチャンクに分割
    # chunk_sizeはチャンクのサイズ、chunk_overlapはオーバーラップする文字数
    texts = CharacterTextSplitter(
        separator="\n",
        chunk_size=100,
        chunk_overlap=20
    ).split_text(raw_text)
    # テキストをDocumentオブジェクトに変換
    # Documentオブジェクトは、page_contentとmetadataを持つ
    docs = [Document(page_content=t) for t in texts]
    # HuggingFaceEmbeddings でベクトル化
    # FAISSは、ベクトルストアの一種で、ベクトル検索を行うためのライブラリ
    return FAISS.from_documents(docs, HuggingFaceEmbeddings(model_name="all-mpnet-base-v2"))

knowledge_base = create_knowledge_base("pc.pdf")

# 初期メッセージ
system_message = SystemMessage(content="あなたは関西弁のアシスタントです。")

# 会話履歴を格納するリスト
conversation_history = [system_message]

# StreamlitのUIを作成
st.title("langchain-streamlit-app")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザーからの入力を受け取る
# chat_inputは、ユーザーからの入力を受け取るためのStreamlitのウィジェット
prompt = st.chat_input("お困りのことがあれば教えてください。")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    # ユーザーからの入力を受け取ったら、ナレッジベースから関連する情報を取得
    retriever = knowledge_base.as_retriever()
    context_text = "\n\n".join([doc.page_content for doc in retriever.invoke(prompt)]) # ユーザーからの入力に関連する情報を取得

    gen_prompt = f"質問: {prompt}\n\n以下は、参考情報です。\n\n{context_text}\n" 
    information = f"以下は、参考情報です。\n{context_text}"
    conversation_history.append(HumanMessage(content=gen_prompt)) # 会話履歴にユーザーからの入力と参考情報を追加


    with st.chat_message("user"):
        st.markdown(prompt) # ユーザーからの入力を表示
    with st.chat_message("assistant"):
        response = llm.invoke(conversation_history) # モデルに会話履歴を渡して応答を生成
        # 会話履歴にAIの応答を追加
        conversation_history.append(AIMessage(content=response.content))
        st.markdown(response.content) # AIの応答を表示
        if st.button("参考情報を表示"):
            st.markdown(information)

    st.session_state.messages.append({"role": "assistant", "content": response.content}) 