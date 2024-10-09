# author time:2024-07-17
# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
#
# inference_pipeline = pipeline(
#     task=Tasks.emotion_recognition,
#     model="/mnt/data1/zpf/emotion2vec_plus_large/model.pt")  # Alternative: iic/emotion2vec_plus_seed, iic/emotion2vec_plus_base, iic/emotion2vec_plus_large and iic/emotion2vec_base_finetuned
#
# rec_result = inference_pipeline('/mnt/data1/zpf/emo_label/aud.wav', output_dir="./outputs", granularity="frame", extract_embedding=False)
# print(rec_result)

from pydub import AudioSegment
from funasr import AutoModel
import os
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--wav", type=str, default="/mnt/data1/zpf/emotion2vec_plus_large/example/aud.wav")


opt = parser.parse_args()


model = AutoModel(model="/mnt/data1/zpf/emotion2vec_plus_large") # Alternative: iic/emotion2vec_plus_seed, iic/emotion2vec_plus_base, iic/emotion2vec_plus_large and iic/emotion2vec_base_finetuned

wav_file = opt.wav
audio = AudioSegment.from_wav(wav_file)
# 创建输出目录
output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)
# 逐帧处理音频
rec_results = []

# 设置每帧的持续时间 (毫秒)
frame_duration_ms = 1000  # 1秒

for i, start_ms in enumerate(range(0, len(audio), frame_duration_ms)):
    end_ms = min(start_ms + frame_duration_ms, len(audio))
    frame = audio[start_ms:end_ms]

    # 将帧保存为临时文件
    temp_wav_file = f"/mnt/data1/zpf/emotion2vec_plus_base/temp_frame_{i}.wav"
    frame.export(temp_wav_file, format="wav")
    print(temp_wav_file)
    # 生成结果
    rec_result = model.generate(temp_wav_file, output_dir=output_dir, granularity="frame", extract_embedding=False)
    print(rec_result)
    rec_results.append(rec_result)

    # 删除临时文件
    os.remove(temp_wav_file)
# rec_result = model.generate(wav_file, output_dir="./outputs", granularity="frame", extract_embedding=False)
# print(rec_results)