# RT-DETR - Detecção de Objetos

Pipeline de treinamento do **RT-DETR** (Real-Time DEtection TRansformer, via Ultralytics) para detecção de objetos. Ele prepara o dataset, treina o modelo e gera gráficos de análise a partir de um único ponto de entrada (`main.py`). O dataset usado nos resultados deste documento é o **Torso 21** (futebol de robôs, RoboCup), mas qualquer dataset no formato YOLO serve.

**Sumário**

1. [Estrutura do repositório](#1-estrutura-do-repositório)
2. [Como funciona o pipeline](#2-como-funciona-o-pipeline)
3. [Como rodar](#3-como-rodar)
4. [Parâmetros](#4-parâmetros)
5. [Resultados do treinamento (Torso 21)](#5-resultados-do-treinamento-torso-21)

---

## 1. Estrutura do repositório

```
VisaoComputacional/
├── Dateset/                     # Datasets no formato YOLO (um por pasta)
│   └── Torso21/
│       ├── data.yaml
│       ├── train/{images,labels}
│       ├── valid/{images,labels}   # criado pelo split
│       └── test/{images,labels}    # criado pelo split
└── RTDETR/
    ├── main.py                  # Ponto de entrada: configurações e orquestração das etapas
    ├── dataset_utils.py         # DatasetUtils: ajuste de labels, data.yaml e split train/val/test
    ├── trainer.py               # RTDETRTrainer: carrega o modelo pré-treinado e treina
    ├── analyzer.py              # RTDETRAnalyzer: gráficos e resumo a partir do results.csv
    ├── requirements.txt         # Dependências Python
    ├── Dockerfile               # Imagem de treinamento (versão de CUDA parametrizável)
    ├── docker-compose.yml       # Serviços train (GPU), train-cpu
    ├── .env.example             # Todos os parâmetros com seus valores padrão
    └── RTDETR_Treinamentos_<DATASET>/<RUN>/   # Saída de cada treinamento
```

---

## 2. Como funciona o pipeline

O `main.py` executa até três etapas, em ordem, cada uma ligada por uma flag:

| Etapa | Flag | O que faz |
|-------|------|-----------|
| **Preparação do dataset** | `FIX_DATASET` | Atualiza o `data.yaml` para apontar para `train/`, `valid/` e `test/`. Se `FORCE_SINGLE_CLASS` estiver ativo, reescreve todas as labels para uma única classe genérica. Em seguida move uma fração aleatória de `train/` para `valid/` (`VAL_RATIO`) e `test/` (`TEST_RATIO`). Se `valid/` já tiver imagens, o split é ignorado, então rodar de novo não re-divide o dataset. |
| **Treinamento** | `DO_TRAIN` | Carrega o checkpoint `MODEL_VERSION` (baixado automaticamente se não existir) e faz o fine-tuning com os hiperparâmetros configurados. Pesos, gráficos e `results.csv` vão para `PROJECT_DIR/RUN_NAME`. |
| **Análise** | `DO_ANALYSIS` | Lê o `results.csv` do run e salva `analise_personalizada.png` (GIoU loss, loss de classe e mAP por época, treino vs. validação), além de imprimir o melhor mAP@0.5 e mAP@0.5:0.95. |

> A preparação do dataset **altera os arquivos no disco** (move imagens e, com `FORCE_SINGLE_CLASS`, reescreve labels). Mantenha uma cópia do dataset original se precisar dele intacto.

---

## 3. Como rodar

### 3.1 Com Docker (recomendado)

**Pré-requisitos:** Docker com Compose v2 e, para GPU, driver NVIDIA + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
cd RTDETR

# Copie os parâmetros e ajuste o que quiser
cp .env.example .env

# Treina na GPU
docker compose up --build

# Treina na CPU
docker compose --profile cpu up --build train-cpu


```

Os parâmetros também podem ser passados direto na linha de comando, sem editar o `.env`:

```bash
EPOCHS=80 MODEL_VERSION=rtdetr-x.pt RUN_NAME=rtdetr_x_80ep docker compose up
```

O container monta esta pasta em `/workspace/app` e o dataset (`DATASET_HOST_DIR`, padrão `../Dateset`) em `/workspace/datasets`, então os resultados aparecem direto em `RTDETR_Treinamentos_<DATASET>/` no host, com o seu usuário como dono (`UID`/`GID`).

> Com o `RUN_NAME` padrão (`teste_<DATASET>`), um novo treino **sobrescreve** o run anterior de mesmo nome. Use outro `RUN_NAME` para preservar resultados.

**Escolhendo a versão de CUDA.** As wheels do PyTorch já trazem o runtime do CUDA, então a imagem só precisa saber de qual índice baixar o `torch`. O host precisa de um driver compatível com a versão escolhida.

| `TORCH_CUDA` | Quando usar |
|--------------|-------------|
| `cu118` | Drivers antigos (≥ 520) |
| `cu121` / `cu124` | Drivers ≥ 530 / ≥ 550 |
| `cu126` (padrão) | Drivers ≥ 560 |
| `cu128` | GPUs Blackwell (RTX 50xx) ou drivers ≥ 570 |
| `cpu` | Sem GPU (é o que o serviço `train-cpu` usa) |

Cada valor gera uma imagem com tag própria (`rtdetr-trainer:<TORCH_CUDA>`), então é possível ter várias lado a lado:

```bash
TORCH_CUDA=cu118 docker compose build
TORCH_CUDA=cu118 docker compose up
```

`TORCH_VERSION` e `TORCHVISION_VERSION` fixam versões específicas (vazio = a mais recente disponível para aquele CUDA).

### 3.2 Localmente (sem Docker)

```bash
cd RTDETR
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # ou cpu
pip install -r requirements.txt

python main.py

```

Sem variáveis de ambiente, valem os padrões definidos no topo do `main.py` (lá `FIX_DATASET` vem desligado e `DO_TRAIN` ligado).

---

## 4. Parâmetros

Todos os valores do topo do `main.py` podem ser sobrescritos por variável de ambiente com o mesmo nome. A lista completa, com os padrões, está em [`.env.example`](.env.example).

| Grupo | Variáveis |
|-------|-----------|
| **Dataset** | `DATASET_NAME`, `DATASET_ROOT`, `VAL_RATIO`, `TEST_RATIO`, `FORCE_SINGLE_CLASS`, `SINGLE_CLASS_NAME`, `SINGLE_CLASS_ID` |
| **Modelo e treino** | `MODEL_VERSION` (`rtdetr-l.pt`, `rtdetr-x.pt`), `EPOCHS`, `BATCH_SIZE` (`-1` = automático), `IMAGE_SIZE`, `PATIENCE`, `DEVICE` (`0`, `0,1`, `cpu`), `WORKERS`, `SEED` |
| **Otimização** | `OPTIMIZER`, `LR0`, `LRF`, `MOMENTUM`, `WEIGHT_DECAY` |
| **Saída** | `PROJECT_DIR`, `RUN_NAME` |
| **Etapas** | `FIX_DATASET`, `DO_TRAIN`, `DO_ANALYSIS` |
| **Docker (build)** | `PYTHON_VERSION`, `TORCH_CUDA`, `TORCH_VERSION`, `TORCHVISION_VERSION` |
| **Docker (execução)** | `GPU_COUNT`, `SHM_SIZE`, `UID`, `GID`, `DATASET_HOST_DIR` |

---

## 5. Resultados do Treinamento (Torso 21)

Esta seção apresenta o treinamento do **RT-DETR Large** no dataset **Torso 21**, configurado para detectar elementos e marcações em um campo de futebol de robôs (RoboCup). Resultados e pesos estão em `RTDETR_Treinamentos_Torso21/teste_Torso21`.

### 5.1 Visão Geral do Treinamento

* **Algoritmo:** RT-DETR (Ultralytics), detector baseado em Transformer, sem NMS.
* **Modelo Base:** `rtdetr-l.pt` (RT-DETR Large, pré-treinado na COCO).
* **Dataset:** Torso 21 (6 classes: `L`, `T`, `X`, `ball`, `goalpost`, `robot`).
* **Épocas executadas:** 43 de 60. O treino parou por **early stopping**: a melhor época foi a 23 e as 20 seguintes não melhoraram.
* **Tempo Total de Treinamento:** 14.721 segundos (~4 horas e 5 minutos), média de **~5,7 minutos por época**.
* **Hardware Utilizado:** GPU NVIDIA RTX 3060 (AMP habilitado).

### 5.2 Hiperparâmetros e Configurações

Registrados em `args.yaml`:

* **Épocas (`epochs`):** `60`, com **early stopping (`patience`)** de `20`.
* **Tamanho do Lote (`batch`):** `8` (menor que no YOLOv8, porque o RT-DETR ocupa bem mais VRAM).
* **Resolução da Imagem (`imgsz`):** `640 x 640`.
* **Otimizador (`optimizer`):** `auto`, com `lr0` `0.01`, `lrf` `0.01`, `momentum` `0.937` e `weight_decay` `0.0005`.
* **Semente (`seed`):** `0`.
* **Data Augmentation:** mosaico `1.0` (desligado nas 10 últimas épocas), erasing `0.4`, escala `0.5`, translação `0.1`, espelhamento horizontal `0.5`, mixup desabilitado.

### 5.3 Resultados e Métricas Obtidas

| Métrica | Época 1 | Época 23 (melhor, `best.pt`) | Época 43 (última) |
|---------|---------|------------------------------|-------------------|
| **Precisão (Precision)** | 85,59% | **93,52%** | 94,15% |
| **Revocação (Recall)** | 77,62% | **91,98%** | 92,51% |
| **mAP @0.5** | 83,03% | **95,14%** | 94,97% |
| **mAP @0.5:0.95** | 49,54% | **63,23%** | 63,02% |
| **GIoU Loss (Treino / Val)** | 0,899 / 0,488 | 0,381 (val) | 0,324 / — |
| **Class Loss (Treino / Val)** | 1,169 / 0,494 | 0,401 (val) | 0,385 / — |

* **Melhor mAP @0.5:** **95,65%** (época 41).
* **Melhor mAP @0.5:0.95:** **63,23%** (época 23). É esta época que o Ultralytics salva como `best.pt`.
* Em algumas épocas (14, 16, 20, 21, 32, 33, 42 e 43) as losses de validação saíram como `NaN` no `results.csv`, enquanto as métricas de mAP continuaram válidas. Por isso os gráficos de loss de validação têm lacunas.

### 5.4 Análise Gráfica dos Resultados

A pasta `teste_Torso21` tem os gráficos gerados pelo Ultralytics e pelo `analyzer.py`:

1. **`results.png` e `analise_personalizada.png`:** evolução das losses (treino vs. validação) e do mAP. O mAP@0.5 já passa de 90% nas primeiras épocas e depois quase não muda, enquanto o mAP@0.5:0.95 atinge o platô por volta da época 23.
2. **`confusion_matrix_normalized.png`:** confusão entre as classes e com o fundo.
3. **`BoxPR_curve.png` e `BoxF1_curve.png`:** Precisão-Revocação por classe e F1 por limiar de confiança.
4. **`val_batch*_pred.jpg` vs. `val_batch*_labels.jpg`:** predições comparadas às anotações reais.

### 5.5 Localização dos Pesos (Weights)

Os pesos treinados estão em `RTDETR_Treinamentos_Torso21/teste_Torso21/weights/`:
* **`best.pt`**: melhor época (23). **Recomendado para uso em testes.**
* **`last.pt`**: última época treinada (43).

### 5.6 Conclusões

1. **Precisão alta:** mAP@0.5 de ~95%, no mesmo nível do YOLOv8 Nano (94,8%).
2. **Localização menos precisa que o YOLOv8:** o mAP@0.5:0.95 (63,2%) ficou abaixo do YOLOv8 Nano (66,8%), mesmo com um modelo ~10x maior.
3. **Custo:** cada época levou ~5,7 min, contra ~44 s no YOLOv8 Nano. O early stopping evitou 17 épocas sem ganho.
