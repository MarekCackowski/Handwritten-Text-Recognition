import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import os

def profile_model_performance(model, dummy_input, device="cuda", num_runs=50):
    """ Mierzy czas inferencji (ms) i zużycie VRAM (MB) dla danego modelu. """
    dev = torch.device(device)
    is_cuda = dev.type == "cuda" and torch.cuda.is_available()

    # Przechodzimy w tryb ewaluacji i na odpowiednie urządzenie
    model.eval()
    model.to(dev)

    # Obliczamy rozmiar samych wag i buforów modelu w pamięci GPU
    weights_bytes = 0
    if is_cuda:
        weights_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) + \
                        sum(b.numel() * b.element_size() for b in model.buffers())

    # Przygotowujemy dane wejściowe na odpowiednim urządzeniu
    if isinstance(dummy_input, dict):
        dummy_input = {k: v.to(dev) for k, v in dummy_input.items()}
    elif isinstance(dummy_input, (tuple, list)):
        dummy_input = [x.to(dev) for x in dummy_input]
    else:
        dummy_input = dummy_input.to(dev)

    def _forward():
        if isinstance(dummy_input, dict):
            return model(**dummy_input)
        if isinstance(dummy_input, list):
            return model(*dummy_input)
        return model(dummy_input)

    # Rozgrzewka GPU
    with torch.inference_mode():
        for _ in range(10):
            _ = _forward()

    # Resetowanie statystyk pamięci GPU przed pomiarem
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        start_mem = torch.cuda.memory_allocated(dev)
    else:
        start_mem = 0

    # Pomiar czasu inferencji
    start_time = time.perf_counter()
    with torch.inference_mode():
        for _ in range(num_runs):
            _ = _forward()

    # Synchronizacja GPU, aby upewnić się, że wszystkie operacje zostały zakończone przed zakończeniem pomiaru czasu
    if is_cuda:
        torch.cuda.synchronize(dev)
    end_time = time.perf_counter()

    # Obliczanie szczytowego zużycia pamięci i średniego czasu inferencji
    peak_mem = torch.cuda.max_memory_allocated(dev) if is_cuda else 0
    activation_bytes = max(0, peak_mem - start_mem)
    avg_time_ms = ((end_time - start_time) / num_runs) * 1000.0
    used_vram_mb = (weights_bytes + activation_bytes) / (1024 * 1024)

    return avg_time_ms, used_vram_mb


def plot_my_pipeline_time_breakdown(crnn_ms, capsnet_ms, transformer_ms, save_dir="chapter5_results"):
    """ Rysuje wykres kołowy pokazujący wąskie gardła wewnątrz kaskady HTR. """
    os.makedirs(save_dir, exist_ok=True)

    modules = ['CRNN (Ekstrakcja i OCR)', 'CapsNet (Weryfikacja)', 'ByT5 (Korekta)']
    times = [crnn_ms, capsnet_ms, transformer_ms]
    total_time = sum(times)

    plt.figure(figsize=(8, 6))
    colors = ['#4C72B0', '#C44E52', '#55A868']

    max_idx = times.index(max(times))
    explode = [0.1 if i == max_idx else 0 for i in range(len(times))]

    plt.pie(times, explode=explode, labels=modules, colors=colors, autopct='%1.1f%%',
            shadow=False, startangle=140, textprops={'fontweight': 'bold'})

    plt.title(f"Rozkład czasu inferencji kaskady\n(Całkowity czas na 1 słowo: {total_time:.1f} ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "5_7_pipeline_time_breakdown.png"), dpi=300)
    plt.close()


def generate_and_save_chapter5_plots(metrics_dict, save_dir="plots"):
    """ Generuje wykresy słupkowe z zebranych metryk. """
    os.makedirs(save_dir, exist_ok=True)
    models = list(metrics_dict.keys())
    times = [metrics_dict[m]['time_ms'] for m in models]
    vrams = [metrics_dict[m]['vram_mb'] for m in models]

    # Wykres czasu inferencji
    plt.figure(figsize=(8, 5))
    plt.bar(models, times, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'][:len(models)])
    plt.ylabel("Czas inferencji na słowo [ms]")
    plt.title("Porównanie czasu przetwarzania modeli (Niższy = Lepszy)")
    for i, v in enumerate(times):
        plt.text(i, v + 0.5, f"{v:.1f} ms", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "inference_times.png"), dpi=300)
    plt.close()

    # Wykres szczytowego zużycia VRAM
    plt.figure(figsize=(8, 5))
    plt.bar(models, vrams, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'][:len(models)])
    plt.ylabel("Szczytowe zużycie VRAM [MB]")
    plt.title("Profilowanie pamięci GPU (Niższy = Lepszy)")
    for i, v in enumerate(vrams):
        plt.text(i, v + 5, f"{v:.0f} MB", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "vram_usage.png"), dpi=300)
    plt.close()
    
    print(f"Wykresy wygenerowano w folderze: {save_dir}/")


def measure_full_page_time(pipeline_function, page_words_tensors, device="cuda"):
    """ Mierzy rzeczywisty czas przetwarzania całej strony tekstu przez zintegrowany system. """
    if not page_words_tensors:
        return 0.0

    dev = torch.device(device)
    is_cuda = dev.type == "cuda" and torch.cuda.is_available()

    def _to_4d(t):
        if t.dim() == 2:
            return t.unsqueeze(0).unsqueeze(0).to(dev)
        if t.dim() == 3:
            return t.unsqueeze(0).to(dev)
        return t.to(dev)

    # Rozgrzewka GPU na jednym słowie
    dummy_input = _to_4d(page_words_tensors[0])
    with torch.inference_mode():
        for _ in range(3):
            _ = pipeline_function(dummy_input)

    if is_cuda:
        torch.cuda.synchronize(dev)
    start_time = time.perf_counter()

    # Przetwarzanie sekwencyjne wszystkich słów ze strony
    with torch.inference_mode():
        for word_tensor in page_words_tensors:
            _ = pipeline_function(_to_4d(word_tensor))

    if is_cuda:
        torch.cuda.synchronize(dev)
    end_time = time.perf_counter()

    return end_time - start_time


def run_full_performance_profiling(crnn_model, capsnet_model, transformer_model, pipeline_func, dummy_img_crnn, dummy_features, dummy_tokens, page_words_tensors, device="cuda", save_dir="chapter5_results"):
    """ Wykonuje pełne profilowanie czasowe i pamięciowe rurociągu. """
    print("Zbieranie metryk z modeli (profilowanie GPU).")
    
    metrics = {}
    t_crnn, v_crnn = profile_model_performance(crnn_model, dummy_img_crnn, device)
    metrics['CRNN (Ekstrakcja)'] = {'time_ms': t_crnn, 'vram_mb': v_crnn}
    
    t_caps, v_caps = profile_model_performance(capsnet_model, dummy_features, device)
    metrics['CapsNet (Weryfikacja)'] = {'time_ms': t_caps, 'vram_mb': v_caps}
    
    t_trans, v_trans = profile_model_performance(transformer_model, dummy_tokens, device)
    metrics['ByT5 (Korekta)'] = {'time_ms': t_trans, 'vram_mb': v_trans}
    
    print("Generowanie wykresów słupkowych czasu i VRAM.")
    generate_and_save_chapter5_plots(metrics, save_dir)
    
    print("Rysowanie rozkładu (bottleneck) kaskady.")
    plot_my_pipeline_time_breakdown(t_crnn, t_caps, t_trans, save_dir)
    
    print("Mierzenie czasu pełnej strony (OOD - Bentham/CVL).")
    page_time = measure_full_page_time(pipeline_func, page_words_tensors, device)
    print(f"Zmierzony czas przetwarzania strony: {page_time:.2f} s")