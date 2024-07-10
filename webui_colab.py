# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))

import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav


logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(levelname)s %(message)s')

def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }

def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

max_val = 0.8
def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(target_sr * 0.2))], dim=1)
    return speech

inference_mode_list = ['3s极速复刻', '跨语种复刻']
instruct_dict = {'预训练音色': '1. 选择预训练音色\n2.点击生成音频按钮',
                 '3s极速复刻': '1. 本地上传参考音频，或麦克风录入\n2. 输入参考音频对应的文本以及希望声音复刻的文本\n3.点击“一键开启声音复刻💕”',
                 '跨语种复刻': '1. 本地上传参考音频，或麦克风录入\n2. **无需输入**参考音频对应的文本\n3.点击“一键开启声音复刻💕”',
                 '自然语言控制': '1. 输入instruct文本\n2.点击生成音频按钮'}
def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]

def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text, seed):
    tts_text = "".join([item1 for item1 in tts_text.strip().split("\n") if item1 != ""]) + ".。"
    prompt_text = "".join([item2 for item2 in prompt_text.strip().split("\n") if item2 != ""])
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    # if instruct mode, please make sure that model is iic/CosyVoice-300M-Instruct and not cross_lingual mode
    if mode_checkbox_group in ['自然语言控制']:
        if cosyvoice.frontend.instruct is False:
            gr.Warning('您正在使用自然语言控制模式, {}模型不支持此模式, 请使用iic/CosyVoice-300M-Instruct模型'.format(args.model_dir))
            return (target_sr, default_data)
        if instruct_text == '':
            gr.Warning('您正在使用自然语言控制模式, 请输入instruct文本')
            return (target_sr, default_data)
        if prompt_wav is not None or prompt_text != '':
            gr.Info('您正在使用自然语言控制模式, prompt音频/prompt文本会被忽略')
    # if cross_lingual mode, please make sure that model is iic/CosyVoice-300M and tts_text prompt_text are different language
    if mode_checkbox_group in ['跨语种复刻']:
        if cosyvoice.frontend.instruct is True:
            gr.Warning('您正在使用跨语种复刻模式, {}模型不支持此模式, 请使用iic/CosyVoice-300M模型'.format(args.model_dir))
            return (target_sr, default_data)
        if instruct_text != '':
            gr.Info('您正在使用跨语种复刻模式, instruct文本会被忽略')
        if prompt_wav is None:
            gr.Warning('您正在使用跨语种复刻模式, 请提供prompt音频')
            return (target_sr, default_data)
        gr.Info('您正在使用跨语种复刻模式, 请确保合成文本和prompt文本为不同语言')
    # if in zero_shot cross_lingual, please make sure that prompt_text and prompt_wav meets requirements
    if mode_checkbox_group in ['3s极速复刻', '跨语种复刻']:
        if prompt_wav is None:
            gr.Warning('prompt音频为空，您是否忘记输入prompt音频？')
            return (target_sr, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('prompt音频采样率{}低于{}'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            return (target_sr, default_data)
    # sft mode only use sft_dropdown
    if mode_checkbox_group in ['预训练音色']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('您正在使用预训练音色模式，prompt文本/prompt音频/instruct文本会被忽略！')
    # zero_shot mode only use prompt_wav prompt text
    if mode_checkbox_group in ['3s极速复刻']:
        if prompt_text == '':
            gr.Warning('prompt文本为空，您是否忘记输入prompt文本？')
            return (target_sr, default_data)
        if instruct_text != '':
            gr.Info('您正在使用3s极速复刻模式，预训练音色/instruct文本会被忽略！')

    if mode_checkbox_group == '预训练音色':
        logging.info('get sft inference request')
        set_all_random_seed(seed)
        output = cosyvoice.inference_sft(tts_text, sft_dropdown)
    elif mode_checkbox_group == '3s极速复刻':
        logging.info('get zero_shot inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        output = cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k)
    elif mode_checkbox_group == '跨语种复刻':
        logging.info('get cross_lingual inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        output = cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k)
    else:
        logging.info('get instruct inference request')
        set_all_random_seed(seed)
        output = cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text)
    audio_data = output['tts_speech'].numpy().flatten()
    return (target_sr, audio_data)

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# <center>🌊💕🎶 [CosyVoice](https://www.bilibili.com/video/BV1vz421q7ir/) 3秒音频，开启最强声音复刻</center>")
        gr.Markdown("## <center>🌟 只需3秒参考音频，一键开启超拟人真实声音复刻，支持中日英韩粤语，无需任何训练！</center>")
        gr.Markdown("### <center>🤗 更多精彩，尽在[滔滔AI](https://www.talktalkai.com/)；滔滔AI，为爱滔滔！💕</center>")

        with gr.Row():
            tts_text = gr.Textbox(label="请填写您希望声音复刻的文本内容", lines=3, placeholder="想说却还没说的，还很多...")
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='请选择声音复刻类型', value=inference_mode_list[0])
            instruction_text = gr.Text(label="📔 操作指南", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='选择预训练音色', value=sft_spk[0], scale=0.25, visible=False)
            with gr.Column(scale=0.25):
                seed_button = gr.Button(value="\U0001F3B2", visible=True)
                seed = gr.Number(value=0, label="随机推理种子", info="若数值保持不变，则每次生成结果一致", visible=True)

        with gr.Row():
            prompt_text = gr.Textbox(label="请填写参考音频对应的文本内容", lines=3, placeholder="告诉我参考音频说了些什么吧...")
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='请从本地上传您喜欢的参考音频，注意采样率不低于16kHz')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='通过麦克风录制参考音频，程序会优先使用本地上传的参考音频')
            generate_button = gr.Button("一键开启声音复刻💕", variant="primary")
        instruct_text = gr.Textbox(label="输入instruct文本", lines=1, placeholder="请输入instruct文本.", value='', visible=False)


        audio_output = gr.Audio(label="为您生成的专属音频🎶")

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text, seed],
                              outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
        gr.Markdown("### <center>注意❗：请不要生成会对个人以及组织造成侵害的内容，此程序仅供科研、学习及个人娱乐使用。请自觉合规使用此程序，程序开发者不负有任何责任。</center>")
        gr.HTML('''
            <div class="footer">
                        <p>🌊🏞️🎶 - 江水东流急，滔滔无尽声。 明·顾璘
                        </p>
            </div>
        ''')
    demo.queue()
    demo.launch(share=True, show_error=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='iic/CosyVoice-300M',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    cosyvoice = CosyVoice(args.model_dir)
    sft_spk = cosyvoice.list_avaliable_spks()
    prompt_sr, target_sr = 16000, 22050
    default_data = np.zeros(target_sr)
    main()
