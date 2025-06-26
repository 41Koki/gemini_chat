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


まず全体の流れとしては、

new_audio.pyで音声データを.docxファイルに書き起こし、

それをもとにmake_knowledge.pyで.pdfと.docxを読み込んでknowledge_baseを作成・保存

最後にmain.pyで質問を受け取り答えを出力できるようにしている

knowledge_baseさえあればmain.pyを実行するだけでアプリを動かせる

new_audio.pyでは、まず.m4aを.wavに変換する

これはlibrosaというライブラリが音声データをいったんAudio型の配列に変換し、それをもう一度wav形式で音声データとして出力しなおしている

わざわざいったんwavとして保存する理由は、その方がほかで使うときに便利だったからである。このプログラムだけならわざわざ保存する必要はない

次に読み込んだ音声データ（Audio型）から30秒分取り出して(sampling_rateを利用)

これを順に書き起こしていくが、この時processorとgenerateを使う

processorとは、音声の入力データを特徴量に変換(feature_extractor)そして、テキストとtokenIDとの相互変換(tokenizer)を行うもの

model.generateは、モデルによる推論を実行している（これに関してはencorder-decoderモデルを想像してそれに入力していると思えばいい）

最後に出力されたもの（多分トークン列）をprocessorで文字に変換している

ここでこの音声認識モデルをファインチューニングできないか考えた。既存のモデルではK-means法を県民図法と書いてしまうなどの問題があった。そこで専門用語を学習させたかった。



