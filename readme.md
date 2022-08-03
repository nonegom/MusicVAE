# MusicVAE Practice
## MusicVAE: A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music

[MusicVAE](https://github.com/magenta/magenta/tree/main/magenta/models/music_vae)는 음악 시퀀스의 latent 공간을 학습해서 다음과 같은 일을 할 수 있게 해주는 모델입니다. 
- 사전 분포로부터 **무작위 샘플링(Sampling)**
- 기존 시퀀스 간의 **보간(Interpolate)**
- 속성 벡터 또는 잠재 제약 모델을 통해 기존 시퀀스 조작을 포함한 다양한 대화형 음악 창작 모드 제공
  

MusicVAE는 [A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music](https://arxiv.org/abs/1803.05428) 논문을 활용한 모델입니다. 이전까지는 Auto Encoder가 순차적 데이터를 모델링하는 방법으로 덜 사용되었었는데, 일반적으로 악보와 같은 이산 토큰 시퀀스는 autoregressive 디코더를 사용해야 했습니다. 이는 부분적으로 autoregression이 때로 충분히 강력해서 AutoEncoder가 latent code를 무시하는 경우가 발생헀기 때문이었습니다. (짧은 시퀀스에서 일부 성공을 보였지만, deep latent variable 모델은 아직 매우 긴 시퀀스에 성공적인 결과를 보여주지는 못했습니다.) 따라서 해당 논문은 hierarchical recurrent decoder를 갖춘 novel sequential autoen-coder를 도입하여, 순환 VAE로 long-term 구조를 모델링함으로 앞서 언급한 문제를 극복하고자 하는 방법을 제시했습니다.  

우선, 짧은 시퀀스(예: 2-bar "루프")의 경우 양방향 LSTM 인코더와 LSTM 디코더를 사용합니다. 긴 시퀀스를 위해 새로운 Hierarchical LSTM을 사용하는데 이는 모델이 장기적인 구조를 학습하는데 도움을 주기 위해서입니다. 또한 Hierarchical 디코더의 최저 수준 임베딩에 대한 여러 디코더를 훈련시켜 기기 간의 상호 의존성을 모델링한다고 합니다.

## 1. 환경 세팅
※ Colab에 SSH연결을 수행해서 작업을 했습니다.  
MusicVAE를 활용하기 위해 환경설정이 필요합니다. `pip install magenta`를 통해 Magenate 라이브러리 설치를 수행합니다.

```
pip install magenta
```

## 2. Model Training
이번에 만드는 모델은 4bar(마디) 드럼 비트를 생성하는 모델입니다.4bar 드럼 비트를 생성을 위해 기본적으로 config 파일이 생성되어 있습니다. `./MusicVAE/configs.py` 내부 코드를 살펴보면 config에는 `['model', 'hparams', 'note_sequence_augmenter', 'data_converter','train_examples_path', 'eval_examples_path', 'tfds_name']` 다음과 같은 7개의 파라미터가 주어집니다. 그 중 내부의 `groovae_4bar` config는 아래와 같이 세팅되어 있습니다. 
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
`groovae_4bar` config를 보면, 디코더로는 GrooveLSTMDecoder가 data_converter로는 GrooveConverter가 Grooveae만을 위한 모델로 사용되고 있습니다. 이는 드럼 비트를 받아들이기 위해 별도로 생성된 모델로 보입니다. 논문에 의하면 MusicVAE의 데이터로 2bar 및 16-bar Monophonic(하나의 채널로 연주)한 멜로디, 2bar 및16-bar 드럼 패턴, 16bar "트리오(멜로디, 베이스라인, 드럼 패턴의 개별 시퀀스)시퀀스" 를 사용합니다. 따라서 해당 데이터에 따라 사용하는 MIDI피치나 클래스가 다르기 때문에 모델별로 별도의 모델 구조가 필요한 것으로 보여집니다.

실제 코드를 보면 다음과 같은 설명이 있음을 확인할 수 있습니다.
```py
class GrooveLstmDecoder(BaseLstmDecoder):
  """Groove LSTM decoder with MSE loss for continuous values.

  At each timestep, this decoder outputs a vector of length (N_INSTRUMENTS*3).
  The default number of drum instruments is 9, with drum categories defined in
  drums_encoder_decoder.py

  For each instrument, the model outputs a triple of (on/off, velocity, offset),
  with a binary representation for on/off, continuous values between 0 and 1
  for velocity, and continuous values between -0.5 and 0.5 for offset.
  """
```
```py
class GrooveConverter(BaseNoteSequenceConverter):
  """Converts to and from hit/velocity/offset representations.

  In this setting, we represent drum sequences and performances
  as triples of (hit, velocity, offset). Each timestep refers to a fixed beat
  on a grid, which is by default spaced at 16th notes.  Drum hits that don't
  fall exactly on beat are represented through the offset value, which refers
  to the relative distance from the nearest quantized step.

  Hits are binary [0, 1].
  Velocities are continuous values in [0, 1].
  Offsets are continuous values in [-0.5, 0.5], rescaled to [-1, 1] for tensors.
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
커스텀 데이터셋으로 [Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove#dataset)를 활용했습니다. 학습 시간을 위해 데이터의 일부만 포함한 [groove-v1.0.0-midionly](https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip) 데이터를 활용하기로 했습니다. 데이터를 다운로드 받아 전처리를 수행합니다. 

#### Training
MIDI 파일을 TF record로 변환했다면, 다음과 같은 스크립트를 실행해서 사용자의 데이터셋으로도 학습을 진행할 수 있습니다.    
```
music_vae_train \
--config=groovae_4bar \
--run_dir=/tmp/grooevae_4bar/ \
--mode=train \
--examples_path=/groove_midonly.tfrecord \
--hparpms= learning_rate=0.0005
```
※ Training default global_step은 200000으로 설정되어 있으며, 1step당 3초가 소요되기 때문에 시간이 매우 오래 걸립니다. 따라서 간단히 코드를 구현하고 싶다면, step 수를 조절할 필요가 있습니다. 혹은 100 step별로 체크포인트가 저장되기에 도중에 멈추고, 체크포인트 파일을 이용할 수 있습니다.   

## 3. Generate Model
모델 학습을 완료했다면(혹은 Pretrained 모델을 다운로드) 모델을 실행해볼 수 있습니다. 아래 스크립트는 빠른 모델 실행을 위해 Pretrained된 모델을 활용하기 위한 명령어입니다. `groovae_4bar`모델을 활용해 생성을 진행했습니다. (추후 개인 모델의 경우 `--checkpoint_file`의 경로 수정)
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
생성된 midi파일을 통해서 Interpolate를 진행할 수 있습니다. Interpolate를 진행할 때는 `input_mid1`과 `input_mid2`개의 midi파일이 필요합니다. 또한, 유의해야할 점이 있는데 `input_mid1`과 `input_mid2`의 형식이 같아야 합니다. groove_4bar 모델의 경우 드럼으로만 구성된 4-bar 길이의 midi파일이 들어와야 한다는 의미입니다. 
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

## 5.Model Sample
이번 과제의 경우 전처리, 학습, 생성 3가지 프로세스를 진행하게 됩니다. pre-trained모델만 활용하면 '전처리' 및 '학습'을 하지 않기 때문에, 이를 해보고자 총 2가지의 모델로 생성을 수행해봤습니다. 첫번째로, Pre-trained모델인 `groovae_4bar`을 활용해서 모델을 생성했고, 두번째로, `groove_midonly`데이터셋을 활용해 만든 `groove_custom`모델을 사용했습니다. (두번째 모델의 경우 loss=41인 모델로, 40000step의 체크포인트를 활용했습니다) 

#### Pretrained-model sample
[groovae_4bar_sample](https://drive.google.com/drive/folders/1rHt6qzFX56tMSXflXk1s7StzkjoAPhO0?usp=sharing)

#### Custom-model sample
[groove_custom_model](https://drive.google.com/file/d/12bzx8Q_-kJj-isiiOscNlcKfNDPqD2_B/view?usp=sharing)
[groove_custom_sample](https://drive.google.com/drive/folders/1IKjPXCtHT6jTNyyIDeNQcWewAu94NbUY?usp=sharing)

다른 2개의 모델을 만들었기에, 기존 pre-trained 모델과 새로 학습한 모델을 비교해볼 수 있었습니다. 우선, 기존 pre-trained모델의 성능이 더 좋았습니다. 새로 학습한 생성한 모델의 경우 기존 pre-trained 모델보다 데이터나, 학습 시간이 충분하지 않았었기 때문에 당연하다고 생각합니다. 그보다 sample을 들어보면, custom sample의 경우 드럼의 패턴이나 소리가 기본적이고 단조로운(스네어와 하이햇 위주) 반면, pretrained sample의 경우 패턴이나 소리가 좀 더 기교있고 다채롭습니다(하이햇 뿐만 아니라 톰톰 등도 포함). 이는 기존 학습했던 데이터의 특성이 반영되었기 때문이라고 생각합니다. custom-model에 사용한 midi데이터의 경우 총 용량이 3MB밖에 되지 않는 데이터이기 때문에 더 다양한 패턴과 소리를 충분히 학습하지 못했을 가능성이 있습니다. 
