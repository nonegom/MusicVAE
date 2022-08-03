# MusicVAE Practice
## MusicVAE: A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music

[MusicVAE](https://github.com/magenta/magenta/tree/main/magenta/models/music_vae)는 음악 시퀀스의 잠재 공간을 학습해서 다음과 같은 일을 수행할 수 있다.
- 사전 분포로부터 무작위 샘플링
- 기존 시퀀스 간의 보간
- 속성 벡터 또는 잠재 제약 모델을 통해 기존 시퀀스 조작을 포함한 다양한 대화형 음악 창작 모드를 제공

짧은 시퀀스(예: 2-bar "루프")의 경우 양방향 LSTM 인코더와 LSTM 디코더를 사용한다. 긴 시퀀스를 위해 새로운 Hierarchical LSTM을 사용하는데 이는 모델이 장기적인 구조를 학습하는데 도움이 되기 때문이다. 또한 Hierarchical 디코더의 최저 수준 임베딩에 대한 여러 디코더를 훈련시켜 기기 간의 상호 의존성을 모델링한다.

## 1. 환경 세팅
Colab을 활용해야 하기 때문에 SSH연결을 수행해서 작업을 진행합니다. MusicVAE를 활용하기 위해 Magenate라이브러리 설치가 필요합니다.

```
pip install magenta
```

## 2. Model Training
4bar 드럼 비트를 생성을 위해 기본적으로 config파일이 생성되어 있습니다. `./MusicVAE/configs.py` 내부 코드를 살펴보면 config에는 `['model', 'hparams', 'note_sequence_augmenter', 'data_converter','train_examples_path', 'eval_examples_path', 'tfds_name']` 다음과 같은 7개의 파라미터가 주어집니다. 그 중 내부의 `groovae_4bar` config를 보면 다음과 같이 세팅되어 있음을 알 수 있습니다. (groovae는 GrooveLSTMDecoder라는 것과 GrooveConverter라는 data_converter라는 Grooveae만을 위한 모델을 사용하고 있습니다. 이는 드럼 비트를 받아들이기 위해 별도로 생성된 모델로 보입니다. )
```py
# GrooVAE configs
CONFIG_MAP['groovae_4bar'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(),
                   lstm_models.GrooveLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=16 * 4,  # 4 bars w/ 16 steps per bar
            z_size=256,
            enc_rnn_size=[512],
            dec_rnn_size=[256, 256],
            max_beta=0.2,
            free_bits=48,
            dropout_keep_prob=0.3,
        )),
    note_sequence_augmenter=None,
    data_converter=data.GrooveConverter(
        split_bars=4, steps_per_quarter=4, quarters_per_bar=4,
        max_tensors_per_notesequence=20,
        pitch_classes=data.ROLAND_DRUM_PITCH_CLASSES,
        inference_pitch_classes=data.REDUCED_DRUM_PITCH_CLASSES),
    tfds_name='groove/4bar-midionly',
)
```

### 2-1. Use TF datasets
Pretrained 모델이 있기는 하지만, 전체적인 과정을 이해하는데 의미가 있기에 모델을 직접 Train 합니다. 4bar 드럼 비트를 생성하는게 목적이기 때문에 다음과 같이 training script를 실행하면 됩니다.[텐서플로우 데이터셋](https://www.tensorflow.org/datasets)에서 제공하는 데이터를 활용해서 모델을 쉽게 생성해낼 수 있습니다. 
```
music_vae_train \
--config=groovae_4bar \
--run_dir=/tmp/grooevae_4bar/ \
--mode=train \
--tfds_name=groove/4bar-midionly
```

### 2-2. Use Custom Dataset
#### PreProcessing
텐서플로우 데이터셋을 활용하지 않고 midi를 통해 학습할 수도 있습니다. 먼저 [데이터 세트 구축 지침](https://github.com/magenta/magenta/blob/main/magenta/scripts/README.md)에 따라 MIDI 파일 컬렉션을 노트 시퀀스의 TF 레코드로 변환하는 전처리 과정이 필요합니다. 

```
INPUT_DIRECTORY=/root/magenta/groove
# TFRecord file that will contain NoteSequence protocol buffers.
SEQUENCES_TFRECORD=/groove_midonly.tfrecord

convert_dir_to_note_sequences \
  --input_dir=$INPUT_DIRECTORY \
  --output_file=$SEQUENCES_TFRECORD \
  --recursive
```
커스텀 데이터셋으로 [Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove#dataset)를 활용했습니다. 그 중 데이터의 일부만 포함한 [groove-v1.0.0-midionly](https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip) 데이터를 활용해 전처리를 진행했습니다. 

#### Training
MIDI파일을 TF record로 변환했다면, 다음과 같은 스크립트를 실행해서 사용자의 데이터셋으로도 학습할 수 있습니다.   
```
music_vae_train \
--config=groovae_4bar \
--run_dir=/tmp/grooevae_4bar/ \
--mode=train \
--examples_path=/groove_midonly.tfrecord
```
Training default global_step은 200000으로 설정되어 있으며, 1step당 3초가 소용되기 때문에 시간이 매우 오래 걸립니다. 따라서 간단히 코드를 구현하고 싶다면, step 수를 조절할 필요가 있습니다. 

## 3. Generate Model
모델을 생성했다면 실행해볼 수 있습니다. 모델 실행을 위해 Pretrained된 모델을 활용해서 예제를 돌려볼 수 있습니다. `groovae_4bar`모델을 활용해 생성을 진행해보도록 하겠습니다.
```
# Pretrained 모델 다운로드 (groovae_4bar)
wget https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/groovae_4bar.tar
```
Pretrained 모델(혹은 위에서 Train한 모델)이 준비됐다면 다음과 같은 스크립트를 통해서 드럼 비트를 생성해날 수 있습니다. (num_outputs를 통해 sample의 개수를 조절할 수 있습니다) 
```
music_vae_generate \
--config=groovae_4bar \
--checkpoint_file=/root/groovae_4bar.tar \
--mode=sample \
--num_outputs=4 \
--output_dir=/root/sample/
```

## 4. Interpolate Model 
생성된 midi파일을 통해서 Interpolate를 진행할 수 있습니다. Interpolate를 진행할 때는 `input_mid1`과 `input_mid2`의 형식이 같아야 합니다. groove_4bar 모델의 경우 드럼으로만 구성된 4-bar 길이의 midi파일이 들어와야 한다는 의미입니다. 
```
music_vae_generate \
--config=groovae_4bar \
--checkpoint_file=/root/groovae_4bar.tar \
--mode=interpolate \
--num_outputs=5 \
--input_midi_1=/root/sample/input1.mid \
--input_midi_2=/root/sample/input2.mid \
--output_dir=/root/sample_interporate/
```