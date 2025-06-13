import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pydub import AudioSegment

# 設定
model_id = "kotoba-tech/kotoba-whisper-v2.0"
device = "cuda" if torch.cuda.is_available() else "cpu"

# プロセッサとモデルのロード
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
model.config.attn_implementation = "flash_attention_2"

import soundfile as sf

audio = AudioSegment.from_file("test_audio.m4a", format="m4a")
audio.export("converted.wav", format="wav")

# 音声ファイルの読み込み
audio_input, sample_rate = sf.read("test_audio.wav")



# 音声データの前処理
input_features = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)

# 推論の実行
generated_ids = model.generate(input_features)
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(transcription)