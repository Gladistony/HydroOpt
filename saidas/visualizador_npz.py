import numpy as np
import matplotlib.pyplot as plt
import os

def visualizar_npz():
    # 1. Pede o nome do arquivo
    pasta = "logs_detalhados_hydroopt"
    print(f"Arquivos disponíveis na pasta '{pasta}':")
    try:
        arquivos = [f for f in os.listdir(pasta) if f.endswith('.npz')]
        for i, f in enumerate(arquivos):
            print(f"  {i+1}. {f}")
    except FileNotFoundError:
        print(f"Erro: Pasta '{pasta}' não encontrada.")
        return

    escolha = input("\nDigite o nome do arquivo (ou o número) que deseja visualizar: ")
    
    # Lógica para aceitar número ou nome
    arquivo_alvo = ""
    try:
        idx = int(escolha) - 1
        if 0 <= idx < len(arquivos):
            arquivo_alvo = arquivos[idx]
    except ValueError:
        arquivo_alvo = escolha

    caminho = os.path.join(pasta, arquivo_alvo)
    
    if not os.path.exists(caminho):
        print("Arquivo não encontrado!")
        return

    print(f"Carregando {arquivo_alvo}...")
    
    # 2. Carrega os dados
    try:
        dados = np.load(caminho, allow_pickle=True)
        hist_fit = dados['hist_fit'] # Shape esperada: [Épocas, Tamanho_População]
        config = str(dados['config'])
    except Exception as e:
        print(f"Erro ao ler arquivo: {e}")
        return

    # Garante que é array numpy
    if isinstance(hist_fit, list):
        hist_fit = np.array(hist_fit)

    # 3. Processamento dos Dados
    # Melhor custo histórico (o mínimo de toda a população a cada geração)
    melhor_historico = np.min(hist_fit, axis=1)
    # Média da população (para ver a tendência geral)
    media_historico = np.mean(hist_fit, axis=1)
    
    epocas = range(len(melhor_historico))

    # 4. Plotagem
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Gráfico 1: Melhor Resultado vs Média
    ax1.plot(epocas, melhor_historico, 'r-', linewidth=2, label='Melhor Global (Líder)')
    ax1.plot(epocas, media_historico, 'b--', alpha=0.5, label='Média do Enxame')
    ax1.set_ylabel('Custo ($)')
    ax1.set_title(f'Convergência: {arquivo_alvo}\n{config}')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Gráfico 2: Todas as Partículas (O "Tubo" de Convergência)
    # Plota cada partícula como uma linha fina transparente
    num_particulas = hist_fit.shape[1]
    # Se houver muitas partículas, plota apenas uma amostra para não travar
    passo = 1 if num_particulas < 100 else 2 
    
    for i in range(0, num_particulas, passo):
        ax2.plot(epocas, hist_fit[:, i], color='gray', alpha=0.3, linewidth=0.8)
    
    # Re-plota o melhor em destaque por cima
    ax2.plot(epocas, melhor_historico, 'r-', linewidth=1.5, label='Melhor Global')
    
    ax2.set_ylabel('Custo ($)')
    ax2.set_xlabel('Época (Iteração)')
    ax2.set_title('Trajetória de Todas as Partículas')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Ajuste de escala logarítmica se os valores explodirem (penalidades)
    if np.max(hist_fit) > 1e8:
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        print("Aviso: Escala Logarítmica ativada devido a valores altos (penalidades).")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualizar_npz()