## 개요

NeRF 모델의 inference 시간을 개선하기 위해 Online Knwoledge Distillation을 적용하여 구현 

## 모델 구조

* Teacher Network를 기존 NeRF에서 사용하는 Coarse Network와 같이 구현 (width : 256, layer : 8)
* Student Network는 width와 layer를 조정하여 Teacher Network보다 가벼운 구조로 설계
![스크린샷 2023-10-30 214917](https://github.com/dltmdgus0220/KnowledgeDistillation_NeRF/assets/101940401/3585e2ef-f792-44c6-9baa-b2e373038f9f)


## Dataset
* NeRF Dataset 중 정면에서 찍은 llff 데이터셋 사용 (https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

## Reference
* [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)
* [NeRF github](https://github.com/bmild/nerf)
