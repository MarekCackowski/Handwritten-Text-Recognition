import torch
import matplotlib.pyplot as plt
import numpy as np
import os


class ErrorVisualizer:
    """
    Klasa odpowiedzialna za walidację modelu i wizualizację przypadków,
    w których predykcja modelu nie zgadza się z etykietą (Ground Truth).
    """

    def __init__(self, model, device, tokenizer, mean=0.5, std=0.5):
        """
        Args:
            model: Twój model CRNN (nn.Module).
            device: 'cuda' lub 'cpu'.
            tokenizer: Obiekt posiadający metodę .decode(indices).
            mean: Średnia użyta do normalizacji (domyślnie 0.5).
            std: Odchylenie standardowe użyte do normalizacji (domyślnie 0.5).
        """
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.mean = mean
        self.std = std

    def _denormalize(self, img_tensor):
        """Konwertuje tensor z powrotem na obraz numpy do wyświetlenia."""
        # img_tensor: [C, H, W] -> numpy: [H, W, C]
        img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)

        # Odwrócenie normalizacji: x * std + mean
        img_np = (img_np * self.std + self.mean)

        # Przycięcie wartości do zakresu [0, 1] (dla bezpieczeństwa)
        img_np = np.clip(img_np, 0, 1)
        return img_np

    @staticmethod
    def _greedy_decode(output_logits):
        """
        Proste dekodowanie CTC (Greedy).
        Zwraca listę indeksów bez powtórzeń i bez tokenu blank.
        """
        # output_logits: [T, Klasy] -> argmax -> [T]
        pred_indices = torch.argmax(output_logits, dim=-1).tolist()

        decoded = []
        prev_char = -1
        blank_idx = 0  # Zakładam, że blank to 0. Jeśli masz inny, zmień to.

        for idx in pred_indices:
            if idx != blank_idx and idx != prev_char:
                decoded.append(idx)
            prev_char = idx
        return decoded

    def run(self, val_loader, num_samples=10, save_path=None):
        """
        Uruchamia wizualizację.

        Args:
            val_loader: DataLoader z danymi walidacyjnymi.
            num_samples: Maksymalna liczba błędów do wyświetlenia.
            save_path: Opcjonalna ścieżka do zapisu obrazka (np. 'errors_epoch_10.png').
        """
        self.model.eval()
        count = 0

        # Przygotowanie płótna
        fig = plt.figure(figsize=(12, 3 * num_samples))
        plt.subplots_adjust(hspace=0.5)

        with torch.no_grad():
            for batch_idx, (imgs, targets, target_lengths) in enumerate(val_loader):
                if count >= num_samples: break

                imgs = imgs.to(self.device)
                outputs = self.model(imgs)  # [T, N, C]

                # Transpozycja dla łatwiejszego iterowania: [N, T, C]
                outputs = outputs.permute(1, 0, 2)

                # Iteracja po elementach w batchu
                for i in range(imgs.size(0)):
                    if count >= num_samples: break

                    # 1. Predykcja
                    decoded_indices = self._greedy_decode(outputs[i])
                    pred_text = self.tokenizer.decode(decoded_indices)

                    # 2. Prawdziwy tekst (Ground Truth)
                    # Wyciąganie odpowiedniego fragmentu z płaskiego wektora targets
                    start = sum(target_lengths[:i])
                    end = start + target_lengths[i]
                    target_indices = targets[start:end].tolist()
                    target_text = self.tokenizer.decode(target_indices)

                    # 3. Jeśli jest błąd -> Rysujemy
                    if pred_text != target_text:
                        count += 1
                        ax = plt.subplot(num_samples, 1, count)

                        img_show = self._denormalize(imgs[i])
                        ax.imshow(img_show, cmap='gray')

                        # Tytuł z kolorami dla czytelności
                        ax.set_title(
                            f"GT: {target_text}\nPred: {pred_text}",
                            color='red', fontsize=14, fontweight='bold', loc='left'
                        )
                        ax.axis('off')

        if count == 0:
            print("Gratulacje! Model nie popełnił żadnego błędu w przeszukanych próbkach.")
            plt.close()
            return

        plt.tight_layout()

        if save_path:
            # Tworzenie katalogu, jeśli nie istnieje
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Zapisano wizualizację błędów do: {save_path}")
            plt.close()  # Zamykamy, żeby nie wisiało w pamięci
        else:
            plt.show()
