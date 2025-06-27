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

そこで、audio_model.ipynbとmake_model.ipynbの二つのプログラムをgoogle colab上で作り実行した(GPUを使いたかったため)

まず、audio_model.ipynbで音声データと元モデルでの予測結果をスプレッドシートにまとめる。

そうして作成したデータをもとにmake_model.ipynbでファインチューニングを行う。

audio_model.ipynbは、new_audio.pyとやっていることはあまり変わらず、スプレッドシートに保存するための作業があるだけである。

make_model.ipynbについては以下の手順でプログラムを実行している

まずデータフレームとしてスプレッドシートを保存する

<img width="857" alt="{297F5068-7087-4B26-895E-E57810130C1B}" src="https://github.com/user-attachments/assets/c178ae50-d7a6-49ca-8791-a8ceb27c4826" />

ここでは、trainデータとvalidationデータの分割と、pathで保存されている音声データをAudio型に変換して新たに列を追加

<img width="222" alt="{735BFE55-1C18-43BB-8CC4-62966B5B09E5}" src="https://github.com/user-attachments/assets/3b11b553-8459-44f2-b5d9-a62c68563253" />

次に二つのデータを特殊な辞書形式で保存

<img width="539" alt="{41F0E4C5-7C73-4014-A5E5-206A9A4FA8FD}" src="https://github.com/user-attachments/assets/89578f7d-6c6a-42cb-8386-778cad69b613" />

ここでモデルをダウンロードする

<img width="783" alt="{B05172BD-8265-4071-A274-E18C585CA3CF}" src="https://github.com/user-attachments/assets/99c83f38-02f7-4d7a-b267-785e12343c60" />

ここで引数がbatchなのは、学習データからbatchごとに取り出したものが自動で割り振られるからである

<img width="882" alt="{EECCCE0A-53C3-4F33-96DC-38F9E06B4127}" src="https://github.com/user-attachments/assets/14aba54a-7620-4c09-83a8-94fb872bc5ed" />

batchから取り出された一つ一つのデータは、この写真のようになっており、castで作られたaudioリストの中の情報を取り出すために、audio = batch["audio"]としている。

また、Audio型の音声データに関してはfeature_extractorで音声特徴量に、正解ラベルの文字データはtokenに変換している。

<img width="720" alt="{26423A4E-64EC-47F7-9458-0D26AD3B3396}" src="https://github.com/user-attachments/assets/75151765-df29-436e-9485-0df827332a4d" />

次にこれらの長さをパディングによってそろえて、BOSトークンなどの特殊トークンの削除などを行う。なお、この特殊トークンはモデルによって違うっぽくて

    if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():

            labels = labels[:, 1:]

としても認識してくれなかったので、いったんトークン列を表示させて目視で確認してから消した。

<img width="679" alt="{73C4F496-B675-4A0E-9F7C-E3752C0C9F28}" src="https://github.com/user-attachments/assets/eee8a07e-2b93-4440-8ec5-bb22ff07906f" />


ここまででデータの前処理が終わった。


次に、モデルを評価する関数を作る必要がある。単語の予測精度としてwerとcerの2種類あり、werは空白ごとに区切ってその制度を予測する。

cerは一文字ずつの精度判定であり、どちらの評価指標を使えばいいかは微妙なラインである。

werは空白ごとの評価なので日本語を空白ごとに分割しなければならない。そのため分かち書きを行う必要がある。この分かち書きとしてsudachipyを用いた。

<img width="456" alt="{3228C501-5A08-4884-AA77-F87FB953B158}" src="https://github.com/user-attachments/assets/7f66dfae-4dff-4db6-a00f-e6713994c31a" />

この関数のmodeとしてsudachipyのtokenizerを設定する。

<img width="302" alt="{9FA8A935-6931-4002-AA39-0E66E260C85E}" src="https://github.com/user-attachments/assets/ec1e4d7f-6350-480b-a516-a71186f3ab23" />

これで学習することでファインチューニングできる。結果的にはこのファインチューニングは全然うまくいかず、逆に既存モデルを壊すことになってしまった。















