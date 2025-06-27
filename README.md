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

    import random

    # データフレームの長さに応じて、0.7 の確率で True を返すリストを作る
    msk = [random.random() < 0.7 for _ in range(len(df))]
    inverse_msk = [not x for x in msk]

    # cast_columnは"colab_path"に含まれているデータをAudio形式に変換している（その後処理が後ろ）
    train_dataset = Dataset.from_pandas(df[msk]).cast_column("colab_path", Audio(sampling_rate=16000)).rename_column("colab_path", "audio").remove_columns(["sampling_rate"])
    validate_dataset = Dataset.from_pandas(df[inverse_msk]).cast_column("colab_path",Audio(sampling_rate=16000)).rename_column("colab_path","audio").remove_columns(["sampling_rate"])

ここでは、trainデータとvalidationデータの分割と、pathで保存されている音声データをAudio型に変換して新たに列を追加

    datasets = DatasetDict({
        "train": train_dataset,
        "validate": validate_dataset
        })

次に二つのデータを特殊な辞書形式で保存

    import torch
    import torchaudio
    model_id = "kotoba-tech/kotoba-whisper-v2.0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import whisper
    from whisper.audio import N_FRAMES, pad_or_trim, log_mel_spectrogram
    from whisper.tokenizer import get_tokenizer
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    model.config.attn_implementation = "flash_attention_2"
    # 下の*.jsonは、モデルに含まれているファイルであり、実際にこれらをダウンロードして音声認識をする

ここでモデルをダウンロードする

    def prepare_dataset(batch):
        audio = batch["audio"]

        # 音響特徴量抽出
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # 正解のテキストをlabel idにエンコード
        batch["labels"] = processor.tokenizer(batch["correct"]).input_ids
        return batch

ここで引数がbatchなのは、学習データからbatchごとに取り出したものが自動で割り振られるからである

    {'url': 'chunk1.wav', 'audio': {'path': '/content/drive/MyDrive/audio_model/output_data/chunk1.wav', 'array': array([0.0060333 , 0.01427276, 0.00360751, ...,0.00340703, 0.00336247,0.00466691]), 'sampling_rate': 16000}, 'correct': '主なクラスタリングの分類としましては階層手段と非階層的手段があって階層的手法の中には短連結法とか完全連結法とか重心法とかがあって階層的手法としてK-means法とかそういうものがあります言い忘れてましたがクラスタリング分類えまあクラスタリングっていうのは教師なし学習のうちの一つになっています', 'whisper': '','__index_level_0__': 1}

batchから取り出された一つ一つのデータは、上のようになっており、castで作られたaudioリストの中の情報を取り出すために、audio = batch["audio"]としている。

また、Audio型の音声データに関してはfeature_extractorで音声特徴量に、正解ラベルの文字データはtokenに変換している。

    prepared_datasets = datasets.map(prepare_dataset, remove_columns=datasets.column_names["train"], num_proc=1)
    # datasetsをmapに対して使用するためにdatasets.mapとしている
    # map()はHugging FaceのDataset型にのみ利用可能で、DatasetDictの中の各キーの中のDatasetに対して一行ずつ順に関数を割り当てていってる
    # num_procは並列処理の数（増やすとうまくいかないらしいから1のまま)

次にこれらの長さをパディングによってそろえて、BOSトークンなどの特殊トークンの削除などを行う。なお、この特殊トークンはモデルによって違うっぽくて

    if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

としても認識してくれなかったので、いったんトークン列を表示させて目視で確認してから消した。

    import torch

    from dataclasses import dataclass
    from typing import Any, Dict, List, Union

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

            #print("padding開始")
            #print('batchごとのfeatures')
            #print(features[0]["labels"])
            # 音響特徴量側をまとめる処理
            # (一応バッチ単位でパディングしているが、すべて30秒分であるはず)
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            # トークン化された正解ラベルをバッチ単位でパディング
            # つまり長さをそろえるために、短いところは0で補うみたいなことをしている
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            #print('正解データに関して取り出したやつ')
            #print(label_features)
            #print('それをpaddingしたやつ')
            #print(labels_batch)

            # attention_maskが0の部分は、トークンを-100に置き換えてロス計算時に無視させる
            # attention_maskはpadding下部分を0そうでない部分を1としたもの(tokenizer.padで自動で作成される)
            # -100を無視するのは、PyTorchの仕様
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            #print('0にしたやつを-100に')

            #print(labels[0,:])

            # BOSトークンがある場合は削除
            # labels[:, 0]は、labelsがテンソルなので（2次元ベクトル（行列））、すべての行に対し最初の列の値をとってくるようにという意味
            # labelsには一行あたり1文（元学習用データの一文が保存されてる）
            # .all() → 全バッチで BOS であることを確認.cpu().item() → Python の bool に変換して if 文で使う
            # 先頭2トークンが BOS と言語トークンなら削除
            # 50258がBOSのIDで、50364が言語のID

            if ((labels[:, 0] == 50258) & (labels[:,1] == 50364)).all():
              labels = labels[:, 2:]

            #print(labels[0,:])

            # 整形したlabelsをバッチにまとめる
            batch["labels"] = labels

            #print('psdding終了')

            return batch


ここまででデータの前処理が終わった。


次に、モデルを評価する関数を作る必要がある。単語の予測精度としてwerとcerの2種類あり、werは空白ごとに区切ってその制度を予測する。

cerは一文字ずつの精度判定であり、どちらの評価指標を使えばいいかは微妙なラインである。

werは空白ごとの評価なので日本語を空白ごとに分割しなければならない。そのため分かち書きを行う必要がある。この分かち書きとしてsudachipyを用いた。

    def tokenize_japanese(sent_list):
        return [
            " ".join([m.surface() for m in tokenizer_obj.tokenize(sent, mode)])
            for sent in sent_list
        ]

この関数のmodeとしてsudachipyのtokenizerを設定する。

    from transformers import Seq2SeqTrainer

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=prepared_datasets["train"],
        eval_dataset=prepared_datasets["validate"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    trainer.train()

これで学習することでファインチューニングできる。結果的にはこのファインチューニングは全然うまくいかず、逆に既存モデルを壊すことになってしまった。















