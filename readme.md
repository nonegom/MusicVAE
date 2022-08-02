# MusicVAE Practice
## MusicVAE: A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music

[MusicVAE](https://github.com/magenta/magenta/tree/main/magenta/models/music_vae)는 음악 시퀀스의 잠재 공간을 학습해서 다음과 같은 일을 수행할 수 있다.
- 사전 분포로부터 무작위 샘플링
- 기존 시퀀스 간의 보간
- 속성 벡터 또는 잠재 제약 모델을 통해 기존 시퀀스 조작을 포함한 다양한 대화형 음악 창작 모드를 제공

짧은 시퀀스(예: 2-bar "루프")의 경우 양방향 LSTM 인코더와 LSTM 디코더를 사용한다. 긴 시퀀스를 위해 새로운 Hierarchical LSTM을 사용하는데 이는 모델이 장기적인 구조를 학습하는데 도움이 되기 때문이다. 또한 Hierarchical 디코더의 최저 수준 임베딩에 대한 여러 디코더를 훈련시켜 기기 간의 상호 의존성을 모델링한다.

## 1. 환경 세팅
Colab을 활용해야 하기 때문에 SSH연결을 수행해서 작업을 진행한다. Magenta library를 사용하기 위해서는 다음과 같은 환경 세팅을 진행해야 합니다. (해당 코드는 과제에 사용된 코드만 담고 있습니다)

```
# repo를 clone해옵니다
git clone https://github.com/tensorflow/magenta.git
```
위치를 magenta폴더로 옮기고, 설치 명령을 실행하여 dependency 설치합니다.
```
pip install -e .
```

## 2. Model Training
### 2-1. Use TF datasets
Pretrained 모델이 있기는 하지만, 전체적인 과정을 이해하는데 의미가 있기에 모델을 직접 Train 합니다. 4bar 드럼 비트를 생성하는게 목적이기 때문에 다음과 같이 training script를 실행하면 됩니다.[텐서플로우 데이터셋](https://www.tensorflow.org/datasets)에서 제공하는 데이터를 활용해서 모델을 쉽게 생성할 수 있습니다. 
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
SEQUENCES_TFRECORD=/tmp/notesequences.tfrecord

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
--examples_path=/tmp/music_vae/groove_midonly.tfrecord
```

## 3. Generate Model
