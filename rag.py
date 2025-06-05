import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import AssemblyAIAudioTranscriptLoader
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import whisper
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import fitz
from fpdf import FPDF


# 同一フォルダにある.envファイルを読み込む
load_dotenv()
# モデルはGemini-1.5-flashを指定
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

#def get_audio_transcript(file_list):
    #"""
    #AssemblyAIの音声ファイルを読み込み、テキストに変換
    #"""
    #l = []
    #for file in file_list:
        #print(f'Transcribing{file}')
        #l.append(AssemblyAIAudioTranscriptLoader(file_path=file).load()[0])
    #return l

def get_audio_gpt_transcript(file_list):
    """
    音声ファイルを読み込み、テキストに変換
    """
    model = whisper.load_model("base")
    for file in file_list:
        print(f'Transcribing {file}')
        result = model.transcribe(file, language="ja")
        yield Document(page_content=result["text"])
    return "Transcription completed."

         
# 音声ファイルのリストを取得
file_list = ["7.m4a", "8.m4a"]

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Transcription", ln=True, align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

# PDFに書き込み
pdf = PDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.set_font("Arial", size=12)

transcripts = list(get_audio_gpt_transcript(["7.m4a", "8.m4a"]))

# transcripts は Document オブジェクトのリスト
for doc in transcripts:
    i = 0
    pdf.multi_cell(0, 10, f"[{i+1}] {doc.metadata.get('source', 'N/A')}\n{doc.page_content}\n")
    pdf.ln(5)
    i += 1

# 保存
pdf.output("transcripts.pdf")

# 音声ファイルをテキストに変換
#transcripts = get_audio_gpt_transcript(file_list)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# テキストをチャンクに分割
aud_texts = text_splitter.split_documents(transcripts)

# aud_textsをpdfとして出力
# ...existing code...
# aud_textsをpdfとして出力

def save_aud_texts_to_pdf(aud_texts, output_path):
    doc = fitz.open()
    for chunk in aud_texts:
        page = doc.new_page()
        # chunkがDocument型の場合はpage_content属性を使う
        text = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
        page.insert_text((72, 72), text, fontsize=12)
    doc.save(output_path)
    doc.close()

save_aud_texts_to_pdf(aud_texts, "audio_transcripts.pdf")
# ...existing code...

# pdfファイルを読み込み、検索可能なナレッジベースを作成する関数
def create_knowledge_base(file_path1, file_path2, aud_texts):
    """
    テキストファイルを読み込み、検索可能なナレッジベースを作成
    """
    with fitz.open(file_path1) as doc1:
        raw_text = "\n".join(page.get_text() for page in doc1) # PDFのテキストを取得
    with fitz.open(file_path2) as doc2:
        raw_text += "\n" + "\n".join(page.get_text() for page in doc2)
    raw_text += "\n" + "\n".join([doc.page_content for doc in transcripts])
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

knowledge_base = create_knowledge_base("7.pdf", "8.pdf", transcripts)

# 初期メッセージ
system_message = SystemMessage(content="あなたは授業アシスタントです。\n\
                                        授業の内容に関する質問に答えることができます。\n\
                                        質問に答えるために、以下の情報を参考にしてください。\n\
                                        参考情報は、授業の内容に関する情報です。\n\
                                        質問に対する答えは、参考情報をもとに生成してください。\n\
                                        もし、参考情報に答えがない場合は、追加資料には該当情報がないことを明確に伝えてから回答してください。\n")

# 会話履歴を格納するリスト
# のちにボタンを押したとき、会話履歴がリセットされるのを防ぐために、
# session_stateにconversation_historyを格納
# session_stateは、Streamlitのセッション状態を管理するための辞書型オブジェクト
# 最初だけ初期メッセージを追加
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [system_message]
conversation_history = st.session_state.conversation_history

# StreamlitのUIを作成
st.title("langchain-streamlit-app")

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
    st.session_state.messages.append({"role": "user", "content": prompt})
    # ユーザーからの入力を受け取ったら、ナレッジベースから関連する情報を取得
    retriever = knowledge_base.as_retriever()
    context_text = "\n\n".join([doc.page_content for doc in retriever.invoke(prompt)]) # ユーザーからの入力に関連する情報を取得

    gen_prompt = f"質問: {prompt}\n\n以下は、参考情報です。\n\n{context_text}\n" 
    st.session_state.information = f"以下は、参考情報です。\n{context_text}"
    conversation_history.append(HumanMessage(content=gen_prompt)) # 会話履歴にユーザーからの入力と参考情報を追加


    with st.chat_message("user"):
        st.markdown(prompt) # ユーザーからの入力を表示
    with st.chat_message("assistant"):
        response = llm.invoke(conversation_history) # モデルに会話履歴を渡して応答を生成
        # 会話履歴にAIの応答を追加
        conversation_history.append(AIMessage(content=response.content))
        st.markdown(response.content) # AIの応答を表示
    st.session_state.messages.append({"role": "assistant", "content": response.content}) 

# 参考情報を表示するボタンを作成
# if promptの外に表示しないと、ボタンが押されても情報が表示されない
if st.button("参考情報を表示"):
    if "information" in st.session_state:
        st.markdown(st.session_state.information)
