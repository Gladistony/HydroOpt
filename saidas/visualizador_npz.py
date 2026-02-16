"""
Visualizador de logs .npz gerados pelo benchmark do HydroOpt.

Compatível com o novo formato:
  - hist_fit: shape [Épocas, PopSize] (pode conter NaN para pop variável)
  - seed_usado: seed para reprodução
  - config: string com parâmetros

Usa nanmin/nanmean para lidar com NaN padding.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def visualizar_npz(caminho_npz=None):
    """
    Carrega um arquivo .npz e plota convergência + trajetória de partículas.

    Args:
        caminho_npz: Caminho direto para o .npz. Se None, mostra menu interativo.
    """
    pasta = "logs_detalhados_hydroopt"

    if caminho_npz is None:
        # Menu interativo
        print(f"Arquivos disponíveis na pasta '{pasta}':")
        try:
            arquivos = sorted([f for f in os.listdir(pasta) if f.endswith('.npz')])
            if not arquivos:
                print("  Nenhum arquivo .npz encontrado.")
                return
            for i, f in enumerate(arquivos):
                print(f"  {i + 1}. {f}")
        except FileNotFoundError:
            print(f"Erro: Pasta '{pasta}' não encontrada.")
            return

        escolha = input("\nDigite o nome do arquivo (ou o número): ").strip()

        arquivo_alvo = ""
        try:
            idx = int(escolha) - 1
            if 0 <= idx < len(arquivos):
                arquivo_alvo = arquivos[idx]
            else:
                print("Índice fora do intervalo.")
                return
        except ValueError:
            arquivo_alvo = escolha

        caminho_npz = os.path.join(pasta, arquivo_alvo)

    if not os.path.exists(caminho_npz):
        print(f"Arquivo não encontrado: {caminho_npz}")
        return

    nome_arquivo = os.path.basename(caminho_npz)
    print(f"Carregando {nome_arquivo}...")

    # ------------------------------------------------------------------
    # Carrega dados
    # ------------------------------------------------------------------
    try:
        dados = np.load(caminho_npz, allow_pickle=True)
        hist_fit = dados['hist_fit']  # Shape: [Épocas, PopSize]
        config = str(dados.get('config', 'N/A'))
        seed_usado = str(dados.get('seed_usado', 'N/A'))
    except Exception as e:
        print(f"Erro ao ler arquivo: {e}")
        return

    if isinstance(hist_fit, list):
        hist_fit = np.array(hist_fit, dtype=float)

    if hist_fit.ndim == 1:
        hist_fit = hist_fit.reshape(-1, 1)

    n_epocas, n_particulas = hist_fit.shape
    print(f"  Shape: {hist_fit.shape}  |  Épocas: {n_epocas}  |  Partículas: {n_particulas}")
    print(f"  Seed usado: {seed_usado}")
    print(f"  Config: {config}")

    # ------------------------------------------------------------------
    # Processamento (nanmin/nanmean para lidar com NaN padding)
    # ------------------------------------------------------------------
    melhor_historico = np.nanmin(hist_fit, axis=1)
    media_historico = np.nanmean(hist_fit, axis=1)
    pior_historico = np.nanmax(hist_fit, axis=1)
    desvio_historico = np.nanstd(hist_fit, axis=1)

    # Melhor acumulado (running best)
    melhor_acumulado = np.minimum.accumulate(melhor_historico)

    epocas = np.arange(n_epocas)

    # ------------------------------------------------------------------
    # Plotagem
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    # --- Gráfico 1: Convergência Global ---
    ax1 = axes[0]
    ax1.plot(epocas, melhor_acumulado, 'r-', linewidth=2.5, label='Melhor Acumulado')
    ax1.plot(epocas, melhor_historico, 'g--', linewidth=1, alpha=0.7, label='Melhor da Época')
    ax1.plot(epocas, media_historico, 'b--', alpha=0.5, label='Média do Enxame')
    ax1.fill_between(
        epocas,
        media_historico - desvio_historico,
        media_historico + desvio_historico,
        alpha=0.15, color='blue', label='± 1σ'
    )
    ax1.set_ylabel('Custo ($)')
    titulo = f'Convergência: {nome_arquivo}\n{config}\nSeed: {seed_usado}'
    ax1.set_title(titulo, fontsize=10)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Gráfico 2: Trajetória de Todas as Partículas ---
    ax2 = axes[1]
    passo = 1 if n_particulas <= 100 else max(1, n_particulas // 80)

    for i in range(0, n_particulas, passo):
        coluna = hist_fit[:, i]
        # Não plotar se for toda NaN
        if np.all(np.isnan(coluna)):
            continue
        ax2.plot(epocas, coluna, color='gray', alpha=0.25, linewidth=0.6)

    ax2.plot(epocas, melhor_acumulado, 'r-', linewidth=1.5, label='Melhor Acumulado')
    ax2.set_ylabel('Custo ($)')
    ax2.set_title('Trajetória de Todas as Partículas', fontsize=10)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # --- Gráfico 3: Taxa de Melhoria ---
    ax3 = axes[2]
    if n_epocas > 1:
        melhoria = np.diff(melhor_acumulado)
        melhoria_pct = np.where(
            melhor_acumulado[:-1] != 0,
            (melhoria / np.abs(melhor_acumulado[:-1])) * 100,
            0
        )
        cores = ['green' if v < 0 else 'red' for v in melhoria_pct]
        ax3.bar(epocas[1:], melhoria_pct, color=cores, alpha=0.7, width=0.8)
        ax3.axhline(y=0, color='black', linewidth=0.5)
        ax3.set_ylabel('Melhoria (%)')
        ax3.set_xlabel('Época (Iteração)')
        ax3.set_title('Taxa de Melhoria por Época', fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.6)
    else:
        ax3.text(0.5, 0.5, 'Dados insuficientes (1 época)',
                 ha='center', va='center', transform=ax3.transAxes)

    # Escala logarítmica se houver penalidades altas
    valor_max = np.nanmax(hist_fit)
    if valor_max > 1e8:
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        print("⚠️  Escala logarítmica ativada (valores > 1e8 detectados).")

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Resumo textual
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 RESUMO ESTATÍSTICO")
    print("=" * 60)
    print(f"  Melhor custo final    : ${melhor_acumulado[-1]:,.2f}")
    print(f"  Média final do enxame : ${media_historico[-1]:,.2f}")
    print(f"  Pior final do enxame  : ${pior_historico[-1]:,.2f}")
    print(f"  Desvio final          : ${desvio_historico[-1]:,.2f}")
    if n_epocas > 1:
        melhoria_total = melhor_historico[0] - melhor_acumulado[-1]
        print(f"  Melhoria total        : ${melhoria_total:,.2f}")
        # Época onde ocorreu a última melhoria significativa
        diffs = np.diff(melhor_acumulado)
        epocas_melhoria = np.where(diffs < -1e-6)[0]
        if len(epocas_melhoria) > 0:
            print(f"  Última melhoria       : Época {epocas_melhoria[-1] + 1}")
            print(f"  Épocas com melhoria   : {len(epocas_melhoria)}/{n_epocas}")
        else:
            print(f"  Nenhuma melhoria detectada entre épocas.")
    print(f"  Seed para reprodução  : {seed_usado}")
    print("=" * 60)


if __name__ == "__main__":
    visualizar_npz()