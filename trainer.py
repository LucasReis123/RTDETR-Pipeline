import os
from ultralytics import RTDETR

class RTDETRTrainer:
    def __init__(self, model_version="rtdetr-l.pt", dataset_yaml_path=""):
        """
            Loads the pretrained RT-DETR checkpoint that will be fine-tuned on the dataset described
            by dataset_yaml_path. If the checkpoint isn't on disk, Ultralytics downloads it.

        Args:
            model_version (str): Pretrained checkpoint to start from (e.g. rtdetr-l.pt, rtdetr-x.pt).
            dataset_yaml_path (str): Full path to the dataset's data.yaml file.
        """
        self.model_version = model_version
        self.dataset_yaml_path = dataset_yaml_path

        print(f"Carregando modelo base: {self.model_version}")
        self.model = RTDETR(self.model_version)

    def train(self, epochs=50, batch_size=16, imgsz=640, project_dir="RTDETR_Treinamentos", name="run",
              optimizer='auto', lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
              patience=20, device=None, workers=8, seed=0):
        """
            Runs the Ultralytics training loop and saves weights, plots and results.csv in
            project_dir/name. An existing run with the same name is overwritten.

        Args:
            epochs (int): Number of training epochs.
            batch_size (int): Images per batch (-1 lets Ultralytics pick it from the GPU memory).
            imgsz (int): Square input resolution used for training and validation.
            project_dir (str): Parent directory where the run folder is created.
            name (str): Name of the run folder inside project_dir.
            optimizer (str): 'SGD', 'Adam', 'AdamW', 'RMSProp' or 'auto'.
            lr0 (float): Initial learning rate.
            lrf (float): Final learning rate as a fraction of lr0.
            momentum (float): SGD momentum / Adam beta1.
            weight_decay (float): L2 regularization factor.
            patience (int): Epochs without improvement before early stopping.
            device (str | None): '0', '0,1', 'cpu', ... None lets Ultralytics choose.
            workers (int): Number of dataloader worker processes.
            seed (int): Random seed used for reproducibility.

        Returns:
            ultralytics.utils.metrics.DetMetrics: Validation metrics of the final model.
        """
        print(f"Iniciando treinamento por {epochs} épocas...")

        results = self.model.train(
            data=self.dataset_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            project=project_dir,
            name=name,
            exist_ok=True,
            optimizer=optimizer,
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            patience=patience,
            device=device,
            workers=workers,
            seed=seed,
        )

        print(f"Treinamento finalizado. Resultados salvos em: {os.path.join(project_dir, name)}")
        return results

if __name__ == "__main__":

    # Teste
    dataset_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Dateset", "heart", "data.yaml")
    trainer = RTDETRTrainer(dataset_yaml_path=dataset_yaml)
