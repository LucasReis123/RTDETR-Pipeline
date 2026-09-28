import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class RTDETRAnalyzer:
    def __init__(self, run_dir):
        """
            Points the analyzer to a finished training run, whose results.csv is used by the
            plotting and summary methods.

        Args:
            run_dir (str): Run directory created by the training (e.g. RTDETR_Treinamentos/run).
        """
        self.run_dir = run_dir
        self.results_csv = os.path.join(self.run_dir, "results.csv")

    def plot_training_curves(self):
        """
            Plots GIoU loss, class loss, mAP@0.5 and mAP@0.5:0.95 per epoch (train vs. val when
            available) and saves the figure as analise_personalizada.png inside run_dir. Useful to
            spot overfitting.
        """
        print(f"Gerando gráficos de análise a partir de {self.results_csv}")
        if not os.path.exists(self.results_csv):
            print("Arquivo results.csv não encontrado. O treinamento completou com sucesso?")
            return

        # Limpa espaços no nome das colunas
        df = pd.read_csv(self.results_csv)
        df.columns = df.columns.str.strip()

        epochs = df['epoch']

        # Configuração do gráfico
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Análise do Treinamento RT-DETR', fontsize=16)

        # Box Loss (no RT-DETR é a GIoU Loss)
        axes[0, 0].plot(epochs, df['train/giou_loss'], label='Train GIoU Loss', color='blue')
        if 'val/giou_loss' in df.columns:
            axes[0, 0].plot(epochs, df['val/giou_loss'], label='Val GIoU Loss', color='orange')
        axes[0, 0].set_title('Loss de Localização (Box)')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, linestyle='--', alpha=0.7)

        # Class Loss
        axes[0, 1].plot(epochs, df['train/cls_loss'], label='Train Class Loss', color='blue')
        if 'val/cls_loss' in df.columns:
            axes[0, 1].plot(epochs, df['val/cls_loss'], label='Val Class Loss', color='orange')
        axes[0, 1].set_title('Loss de Classificação')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, linestyle='--', alpha=0.7)

        # mAP@0.5
        if 'metrics/mAP50(B)' in df.columns:
            axes[1, 0].plot(epochs, df['metrics/mAP50(B)'], label='mAP@0.5', color='green')
            axes[1, 0].set_title('Métrica mAP@0.5')
            axes[1, 0].set_xlabel('Epochs')
            axes[1, 0].set_ylabel('mAP')
            axes[1, 0].legend()
            axes[1, 0].grid(True, linestyle='--', alpha=0.7)

        # mAP@0.5:0.95
        if 'metrics/mAP50-95(B)' in df.columns:
            axes[1, 1].plot(epochs, df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='purple')
            axes[1, 1].set_title('Métrica mAP@0.5:0.95')
            axes[1, 1].set_xlabel('Epochs')
            axes[1, 1].set_ylabel('mAP')
            axes[1, 1].legend()
            axes[1, 1].grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        custom_plot_path = os.path.join(self.run_dir, "analise_personalizada.png")
        plt.savefig(custom_plot_path)
        print(f"[+] Gráfico de análise salvo em: {custom_plot_path}")

    def show_summary(self):
        """
            Prints the best mAP@0.5 and mAP@0.5:0.95 reached during training, read from results.csv.
        """
        if not os.path.exists(self.results_csv):
            return

        df = pd.read_csv(self.results_csv)
        df.columns = df.columns.str.strip()

        print("\n--- RESUMO DO TREINAMENTO ---")
        if 'metrics/mAP50(B)' in df.columns:
            best_map50 = df['metrics/mAP50(B)'].max()
            print(f"Melhor mAP@0.5: {best_map50:.4f}")

        if 'metrics/mAP50-95(B)' in df.columns:
            best_map50_95 = df['metrics/mAP50-95(B)'].max()
            print(f"Melhor mAP@0.5:0.95: {best_map50_95:.4f}")

if __name__ == "__main__":
    analyzer = RTDETRAnalyzer("RTDETR_Treinamentos/run")
