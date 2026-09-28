import os
from dataset_utils import DatasetUtils
from trainer import RTDETRTrainer
from analyzer import RTDETRAnalyzer


def env_str(name, default):
    """
        Reads a string from the environment, falling back to default when the variable is unset
        or empty. Lets Docker (or the shell) override any setting without editing this file.

    Args:
        name (str): Environment variable name.
        default (str): Value used when the variable isn't set.

    Returns:
        str: The variable's value or the default.
    """
    value = os.getenv(name)
    return value if value not in (None, "") else default


def env_int(name, default):
    """
        Same as env_str, converting the value to int.

    Args:
        name (str): Environment variable name.
        default (int): Value used when the variable isn't set.

    Returns:
        int: The variable's value or the default.
    """
    return int(env_str(name, default))


def env_float(name, default):
    """
        Same as env_str, converting the value to float.

    Args:
        name (str): Environment variable name.
        default (float): Value used when the variable isn't set.

    Returns:
        float: The variable's value or the default.
    """
    return float(env_str(name, default))


def env_bool(name, default):
    """
        Same as env_str, reading "1", "true", "yes" or "on" (any case) as True and anything else
        as False.

    Args:
        name (str): Environment variable name.
        default (bool): Value used when the variable isn't set.

    Returns:
        bool: The variable's value or the default.
    """
    return str(env_str(name, default)).strip().lower() in ("1", "true", "yes", "on")


# HIPERPARÂMETROS E CONFIGURAÇÕES

# Dataset Control Panel
DATASET_NAME = env_str("DATASET_NAME", "Torso21")      # "heart" ou "Torso21"
DATASET_ROOT = env_str("DATASET_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Dateset"))
DATASET_PATH = os.path.join(DATASET_ROOT, DATASET_NAME)
DATASET_YAML = os.path.join(DATASET_PATH, "data.yaml")

# Dataset Splitting
VAL_RATIO = env_float("VAL_RATIO", 0.20)  # Porcentagem para validação (20%)
TEST_RATIO = env_float("TEST_RATIO", 0.10) # Porcentagem para testes (10%)

# Force Generic Class
FORCE_SINGLE_CLASS = env_bool("FORCE_SINGLE_CLASS", False)
SINGLE_CLASS_NAME = env_str("SINGLE_CLASS_NAME", "object")
SINGLE_CLASS_ID = env_int("SINGLE_CLASS_ID", 0)

# Treinamento
MODEL_VERSION = env_str("MODEL_VERSION", "rtdetr-l.pt")  # Versão do modelo (l = large, x = xlarge)
EPOCHS = env_int("EPOCHS", 60)
BATCH_SIZE = env_int("BATCH_SIZE", 8)  # RT-DETR é pesado. Batch menor evita estourar a VRAM
IMAGE_SIZE = env_int("IMAGE_SIZE", 640)
PATIENCE = env_int("PATIENCE", 20)    # Épocas sem melhora até o early stopping
DEVICE = env_str("DEVICE", None)      # "0", "0,1", "cpu"... None = escolha automática
WORKERS = env_int("WORKERS", 8)       # Processos do dataloader
SEED = env_int("SEED", 0)

# Hiperparâmetros Avançados
OPTIMIZER = env_str("OPTIMIZER", "auto")          # 'SGD', 'Adam', 'AdamW', 'RMSProp', 'auto'
LR0 = env_float("LR0", 0.01)                      # Taxa de aprendizado inicial
LRF = env_float("LRF", 0.01)                      # Taxa de aprendizado final
MOMENTUM = env_float("MOMENTUM", 0.937)           # Momento
WEIGHT_DECAY = env_float("WEIGHT_DECAY", 0.0005)  # Regularização L2

# Diretórios de Saída
PROJECT_DIR = env_str("PROJECT_DIR", f"RTDETR_Treinamentos_{DATASET_NAME}")
RUN_NAME = env_str("RUN_NAME", f"teste_{DATASET_NAME}")

# Flags de Execução
FIX_DATASET = env_bool("FIX_DATASET", False)   # True: Prepara o dataset
DO_TRAIN = env_bool("DO_TRAIN", True)          # True: TREINA O MODELO
DO_ANALYSIS = env_bool("DO_ANALYSIS", True)    # True: gera os gráficos de análise


def main():
    """
        Runs the pipeline steps enabled by the flags above: dataset preparation (FIX_DATASET),
        training (DO_TRAIN) and result analysis (DO_ANALYSIS), in that order.
    """
    print("="*50)
    print(" PIPELINE DE DETECÇÃO DE OBJETOS - RT-DETR ")
    print("="*50)

    # Preparação do Dataset
    if FIX_DATASET:
        print("\n--- PREPARAÇÃO DO DATASET ---")
        utils = DatasetUtils(dataset_path=DATASET_PATH)

        # Fundindo todas as classes para uma detecção genérica
        if FORCE_SINGLE_CLASS:
            utils.fix_labels(target_class_id=SINGLE_CLASS_ID)
            utils.update_yaml(yaml_path=DATASET_YAML, single_class_name=SINGLE_CLASS_NAME)
        else:
            # Atualizando as pastas no yaml sem mudar as classes originais
            utils.update_yaml(yaml_path=DATASET_YAML)

        # Fazendo split 70/20/10 dinâmico
        utils.split_dataset(val_ratio=VAL_RATIO, test_ratio=TEST_RATIO)

    # Treinamento
    if DO_TRAIN:
        print("\n--- TREINAMENTO ---")
        trainer = RTDETRTrainer(model_version=MODEL_VERSION, dataset_yaml_path=DATASET_YAML)
        trainer.train(
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            imgsz=IMAGE_SIZE,
            project_dir=PROJECT_DIR,
            name=RUN_NAME,
            optimizer=OPTIMIZER,
            lr0=LR0,
            lrf=LRF,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            patience=PATIENCE,
            device=DEVICE,
            workers=WORKERS,
            seed=SEED
        )

    # Análise
    if DO_ANALYSIS:
        print("\n--- ANÁLISE DOS RESULTADOS ---")
        run_dir = os.path.join(PROJECT_DIR, RUN_NAME)
        analyzer = RTDETRAnalyzer(run_dir=run_dir)

        analyzer.plot_training_curves()
        analyzer.show_summary()

    print("\nPipeline finalizado!")

if __name__ == "__main__":
    main()
