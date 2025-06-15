from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import time


# 同一フォルダにある.envファイルを読み込む
load_dotenv()
# モデルはGemini-1.5-flashを指定
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

def get_lecture_title(file_path):
    """
    PDFファイルから授業のタイトルを取得する関数
    """
    if file_path.endswith(".pdf"):
        n = file_path.replace(".pdf", "")
    elif file_path.endswith(".docx"):
        n = file_path.replace("_aud.docx", "")
    # もしnにaudが含まれていれば、授業のタイトルを取得しない
    if "aud" in n:
        return f"第{n}回講義録音"
    else:
        return f"第{n}回講義資料"

@st.cache_resource
def load_embeddings_and_index():
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    index = FAISS.load_local("faiss_index/", embeddings, allow_dangerous_deserialization=True)
    return index.as_retriever(search_type="similarity", search_kwargs={"k": 10})


if "knowledge_base" not in st.session_state:
    open_know_st = time.time()
    retriever = load_embeddings_and_index()
    open_know_end = time.time()
    print(f"ナレッジベース読み込み時間: {open_know_end - open_know_st:.2f}秒")
else:
    print("セッションキャッシュからナレッジベースとリトリーバーを取得")

retriever = st.session_state.retriever

# 初期メッセージ
system_message = SystemMessage(content="あなたは授業アシスタントです。\n\
                                        授業の内容に関する質問に答えることができます。\n\
                                        質問に対する答えは、参考情報をもとに生成してください。\n\
                                        参考資料で、pdfとdocxで同じ数字がファイル名に含まれるものは、同じ内容について記載されています。\n\
                                        もし、参考情報に関係する単語が全くない場合は、関係する単語が見つからなかったと回答してください。\n\
                                        もし、少しだけ関連する単語がある場合は、関連する単語を使って回答してください。\n")

                                        
# 会話履歴を格納するリスト
# のちにボタンを押したとき、会話履歴がリセットされるのを防ぐために、
# session_stateにconversation_historyを格納
# session_stateは、Streamlitのセッション状態を管理するための辞書型オブジェクト
# 最初だけ初期メッセージを追加
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [system_message]
conversation_history = st.session_state.conversation_history

# StreamlitのUIを作成
st.title("知能情報システム-QAChatBot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 会話履歴を画面に表示するためのリストであるmessagesを作成
# messagesは、ユーザーとAIのメッセージを格納するリスト
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザーからの入力を受け取る
# chat_inputは、ユーザーからの入力を受け取るためのStreamlitのウィジェット
prompt = st.chat_input("お困りのことがあれば教えてください。")

if prompt:
    pro_st = time.time()
    st.session_state.messages.append({"role": "user", "content": prompt})
    # ユーザーからの入力を受け取ったら、ナレッジベースから関連する情報を取得
    print("get_retriever")
    retriever_st = time.time()
    context_text = "\n\n".join([
                                f"[{doc.metadata.get('source')}]\n{doc.page_content}" 
                                for doc in retriever.invoke(prompt)]) # ユーザーからの入力に関連する情報を取得
    print("get_context_text")
    context_st = time.time()
    print(f"コンテキスト取得時間: {context_st - retriever_st:.2f}秒")
    gen_prompt = f"質問: {prompt}\n\n以下は、参考情報です。\n\n{context_text}\n" 
    context_end = time.time()
    print(f"プロンプト生成時間: {context_end - context_st:.2f}秒")
    st.session_state.information = f"以下は、参考情報です。\n{context_text}"
    conversation_history.append(HumanMessage(content=gen_prompt)) # 会話履歴にユーザーからの入力と参考情報を追加


    with st.chat_message("user"):
        st.markdown(prompt) # ユーザーからの入力を表示
    with st.chat_message("assistant"):
        response_st = time.time()
        response = llm.invoke(conversation_history) # モデルに会話履歴を渡して応答を生成
        response_end = time.time()
        print(f"応答生成時間: {response_end - response_st:.2f}秒")
        # 会話履歴にAIの応答を追加
        conversation_history.append(AIMessage(content=response.content))
        st.markdown(response.content) # AIの応答を表示
    st.session_state.messages.append({"role": "assistant", "content": response.content}) 

# 参考情報を表示するボタンを作成
# if promptの外に表示しないと、ボタンが押されても情報が表示されない
if st.button("参考情報を表示"):
    if "information" in st.session_state:
        st.markdown(st.session_state.information)

if st.button("🔄 キャッシュをクリアする"):
    st.cache_resource.clear()
    st.experimental_rerun()

