以下、追加インストールが必要なもの

pip install langchain_google_genai

pip install streamlit

pip install langchain_community

pip install PyMuPDF

pip install sentence-transformers

pip install faiss-cpu

pip install assemblyai

pip install openai-whisper

pip install python-docx

pip install python-docx langchain

pip install sentence_transformers

pip install --upgrade pip

pip install --upgrade transformers accelerate

pip install git+https://github.com/openai/whisper.git

pip install soundfile numpy

pip install librosa


そのほか必要なものがあったら随時インストール


音声ファイルの文字おこしについて



streamlit run main.py
初めに音声ファイルを.wav形式に変換する必要がある。本当は自分のパソコン内で完結させたかったが、うまくいかないので外部のサイトで変換。これは今後改善したい

次にコマンドプロンプトで、
ffmpeg -i 入力音声ファイル -ar 16000 出力ファイル名
として周波数？調整



最終的には、音声ファイルとpdfから質問に答えてくれるチャットボットを作成したい。なお、講義で述べられていない内容に関しては、述べられていないと宣言するようにしておく。
