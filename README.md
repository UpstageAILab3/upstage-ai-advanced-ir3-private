[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Tm6AYAOm)
# Title (Please modify the title)
## Team

| <img src="https://drive.google.com/uc?export=view&id=1LHSq2m119E8Vc590kM6PCwBF9quxjY4R" alt="정인웅" width="150" height="100"> | <img src="https://drive.google.com/uc?export=view&id=1Fg7LNgBWTcaBBXEaVLDwI73o6nklBApt" alt="이범희" width="150" height="100">  | <img src="https://drive.google.com/uc?export=view&id=1MYvU2Uur7AySKIpaYOxOyUqvlQHeQvGP" alt="박건민" width="150" height="100"> |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [정인웅](https://github.com/Messengerwoong)             |            [이범희](https://github.com/tmttd)             |            [박건민](https://github.com/UpstageAILab)             |
|                          모든 역할 + gemini api 구현                             |                    모든 역할 + 프롬프트 엔지니어링                             |                       모든 역할 + 각종 알고리즘 구현                             |

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/FVjNDCrt)
# Document Type Classification | 문서 타입 분류
## Team

| <img src="https://drive.google.com/uc?export=view&id=1LHSq2m119E8Vc590kM6PCwBF9quxjY4R" alt="정인웅" width="150" height="100"> | <img src="https://drive.google.com/uc?export=view&id=1Fg7LNgBWTcaBBXEaVLDwI73o6nklBApt" alt="이범희" width="150" height="100"> | <img src="https://drive.google.com/uc?export=view&id=1G2GLuzvsoSigbdlHFneNkNdUCnRss_Z3" alt="안수민" width="150" height="100"> | <img src="https://drive.google.com/uc?export=view&id=1rfLaDJocTPO2c0ctPnCOebACb39Uz3f2" alt="진수훈" width="150" height="100"> |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [정인웅](https://github.com/Messengerwoong)             |            [이범희](https://github.com/tmttd)             |            [안수민](https://github.com/soomnia)             |            [진수훈](https://github.com/huniii32)             |
| - Data Processing, Augementation <br> - Modeling | - Modeling | - Modeling  | - Data Processing, Augementation <br> - Modeling |


## 0. Overview
### Environment
- Ubuntu 20.04.6 LTS
- GPU: RTX 3090 / 24 GB
- Memory: 60G

### Requirements
```python
shutil
PIL
time
random
numpy==1.26.0
sklearn
timm==0.9.12
cv2
torch==2.2.0
matplotlib==3.9.1
gc
seaborn==0.13.2
torchvision==0.17.0
tqdm==4.65.0
collections
albumentations==1.3.1
pandas==2.1.4
os
```

## 1. Competiton Info

### Overview
- Scientific Knowledge Question Answering | 과학 지식 질의 응답 시스템 구축(https://stages.ai/competitions/322/overview/description)

RAG 시스템이 사용자 질문에 적합한 레퍼런스를 찾고 이를 토대로 답을 잘하는 지 보는 목적

- `#비공개대회` `#UpstageAILab3기` `#information Retrieval`

### Timeline

- July 30, 2024 - Start Date
- August 11, 2024 - Final submission deadline

## 2. Components

### Directory
```
```


## 3. Data descrption

### Dataset overview

> **document.jsonl**
{"docid": "42508ee0-c543-4338-878e-d98c6babee66", "src": "ko_mmlu__nutrition__test", "content": "건강한 사람이 에너지 균형을 평형 상태로 유지하는 것은 중요합니다. 에너지 균형은 에너지 섭취와 에너지 소비의 수학적 동등성을 의미합니다. 일반적으로 건강한 사람은 1-2주의 기간 동안 에너지 균형을 달성합니다. 이 기간 동안에는 올바른 식단과 적절한 운동을 통해 에너지 섭취와 에너지 소비를 조절해야 합니다. 식단은 영양가 있는 식품을 포함하고, 적절한 칼로리를 섭취해야 합니다. 또한, 운동은 에너지 소비를 촉진시키고 근육을 강화시킵니다. 이렇게 에너지 균형을 유지하면 건강을 유지하고 비만이나 영양 실조와 같은 문제를 예방할 수 있습니다. 따라서 건강한 사람은 에너지 균형을 평형 상태로 유지하는 것이 중요하며, 이를 위해 1-2주의 기간 동안 식단과 운동을 조절해야 합니다."}
{"docid": "4a437e7f-16c1-4c62-96b9-f173d44f4339", "src": "ko_mmlu__conceptual_physics__test", "content": "수소, 산소, 질소 가스의 혼합물에서 평균 속도가 가장 빠른 분자는 수소입니다. 수소 분자는 가장 가볍고 작은 원자로 구성되어 있기 때문에 다른 분자들보다 더 빠르게 움직입니다. 이러한 이유로 수소 분자는 주어진 온도에서 가장 빠른 평균 속도를 가지고 있습니다. 수소 분자는 화학 반응에서도 활발하게 참여하며, 수소 연료로도 널리 사용됩니다. 따라서 수소 분자는 주어진 온도에서 평균 속도가 가장 빠른 분자입니다."}
> **eval.jsonl**

{"eval_id": 78, "msg": [{"role": "user", "content": "나무의 분류에 대해 조사해 보기 위한 방법은?"}]}
{"eval_id": 213, "msg": [{"role": "user", "content": "각 나라에서의 공교육 지출 현황에 대해 알려줘."}]}

> **test**

3140장의 이미지

### EDA
![EDA](/docs/img/eda.png)

### Data Processing

> **Augraphy**
- BrightnessTexturize(p=1.0)
- LowInkRandomLines(p=1.0)
- LowInkPeriodicLines(p=1.0)
- BindingsAndFasteners(p=1.0)
- LightingGradient(p=1.0)

> **Rotate**

[45, 90, 135, 180, 225, 270, 315]

> **Flip**

hflip, vflip

> **Gaussian Blur**
[0.5, 1, 1.5]



## 4. Modeling

### Model descrition
- chatgpt 3.5 turbo, chatgpt 4o
- gemini-pro 1.5, gemini-flash 1.5

### Modeling Process

- LLM모델을 위한 RAG기능 비교

## 5. Result
F1-Score
### Leader Board
> PUBLIC <br>
![PUBLIC](/docs/img/leaderboard-public.PNG)

> PRIVATE <br>
![PRIVATE](/docs/img/leaderboard-private.PNG)

> BEST SCORE <br>
![Best](/docs/img/best_score.PNG)
### Presentation

- [💾 PDF](/docs/pdf/[패스트캠퍼스]%20Upstage%20AI%20Lab%203기_CV%20경진대회_발표자료_8조.pdf)

## etc
### Reference
- [baseline code](https://stages.ai/competitions/319/data/baseline)

