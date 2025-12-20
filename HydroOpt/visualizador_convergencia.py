"""
Módulo para visualizar e analisar a convergência de otimizações.

Permite plotar gráficos de convergência, comparar múltiplas otimizações,
e analisar a evolução do fitness ao longo das iterações.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd


class VisualizadorConvergencia:
    """
    Visualiza a convergência de otimizações.
    
    Permite gerar gráficos que mostram como o melhor fitness evolui
    ao longo das iterações (épocas) da otimização.
    
    Exemplo:
        >>> viz = VisualizadorConvergencia()
        >>> viz.adicionar_convergencia(historico, label='PSO c1=2.0')
        >>> viz.adicionar_convergencia(historico2, label='PSO c1=2.5')
        >>> viz.plotar()
    """
    
    def __init__(self, verbose=True, titulo_padrao="Convergência de Otimização"):
        """
        Inicializa o visualizador.
        
        Args:
            verbose (bool): Exibir informações
            titulo_padrao (str): Título padrão para os gráficos
        """
        self.verbose = verbose
        self.titulo_padrao = titulo_padrao
        self.convergencias = []  # Lista de {label, historico, dados}
        self.figsize = (12, 6)
        self.dpi = 100
    
    def adicionar_convergencia(self, historico, label, dados_adicionais=None):
        """
        Adiciona um histórico de convergência para visualização.
        
        Args:
            historico (list ou np.ndarray): Array com melhor fitness de cada iteração
            label (str): Rótulo para a curva (ex: "PSO c1=2.0, c2=2.0")
            dados_adicionais (dict, optional): Informações extras {'custo_real': 1000, 'pressao': 30.5}
        """
        historico = np.asarray(historico, dtype=float)
        
        if historico.ndim != 1:
            raise ValueError(f"Histórico deve ser 1D, recebido forma {historico.shape}")
        
        if len(historico) == 0:
            raise ValueError("Histórico não pode estar vazio")
        
        # Verificar se há NaN/Inf
        historico_limpo = np.nan_to_num(historico, nan=np.inf, posinf=np.inf, neginf=np.inf)
        
        self.convergencias.append({
            'label': label,
            'historico': historico_limpo,
            'iteracoes': len(historico_limpo),
            'melhor_fitness': float(np.nanmin(historico_limpo)) if not np.isinf(historico_limpo).all() else np.nan,
            'dados_adicionais': dados_adicionais or {}
        })
        
        if self.verbose:
            print(f"✓ Adicionado: {label} ({len(historico_limpo)} iterações)")
    
    def plotar(self, titulo=None, xlabel="Iteração (Época)", ylabel="Melhor Fitness",
               salvar_em=None, escala_y='linear', mostrar=True):
        """
        Plota todas as convergências adicionadas.
        
        Args:
            titulo (str, optional): Título do gráfico
            xlabel (str): Rótulo do eixo X
            ylabel (str): Rótulo do eixo Y
            salvar_em (str, optional): Caminho para salvar a figura
            escala_y (str): 'linear' ou 'log'
            mostrar (bool): Exibir o gráfico
        """
        if not self.convergencias:
            raise ValueError("Nenhuma convergência foi adicionada. Use adicionar_convergencia() primeiro.")
        
        titulo = titulo or self.titulo_padrao
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Cores e estilos
        cores = plt.cm.tab10(np.linspace(0, 1, len(self.convergencias)))
        estilos = ['-', '--', '-.', ':']
        
        for idx, conv in enumerate(self.convergencias):
            historico = conv['historico']
            iteracoes = np.arange(1, len(historico) + 1)  # Começar do 1, não 0
            
            # Filtrar infinitos para visualização
            historico_viz = np.where(np.isinf(historico), np.nan, historico)
            
            cor = cores[idx % len(cores)]
            estilo = estilos[idx % len(estilos)]
            
            ax.plot(iteracoes, historico_viz, 
                   label=conv['label'],
                   color=cor,
                   linestyle=estilo,
                   linewidth=2,
                   marker='o',
                   markersize=4,
                   alpha=0.7)
        
        # Formatação
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(titulo, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_yscale(escala_y)
        ax.legend(loc='best', fontsize=10)
        
        # Layout tight
        plt.tight_layout()
        
        # Salvar se especificado
        if salvar_em:
            Path(salvar_em).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(salvar_em, dpi=self.dpi, bbox_inches='tight')
            if self.verbose:
                print(f"✓ Gráfico salvo em: {salvar_em}")
        
        # Mostrar
        if mostrar:
            plt.show()
        
        return fig, ax
    
    def plotar_multiplos(self, grupos_convergencias, titulo=None, salvar_em=None):
        """
        Plota múltiplos gráficos lado a lado (um por algoritmo/método).
        
        Args:
            grupos_convergencias (dict): {nome_grupo: [convergencias]}
            titulo (str, optional): Título geral
            salvar_em (str, optional): Caminho para salvar
        
        Exemplo:
            grupos = {
                'PSO': [conv1, conv2, conv3],
                'WOA': [conv4, conv5],
                'GWO': [conv6]
            }
            viz.plotar_multiplos(grupos)
        """
        num_grupos = len(grupos_convergencias)
        fig, axes = plt.subplots(1, num_grupos, figsize=(6*num_grupos, 5), dpi=self.dpi)
        
        if num_grupos == 1:
            axes = [axes]
        
        titulo_geral = titulo or self.titulo_padrao
        fig.suptitle(titulo_geral, fontsize=16, fontweight='bold', y=1.02)
        
        cores = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for idx_grupo, (nome_grupo, convergencias) in enumerate(grupos_convergencias.items()):
            ax = axes[idx_grupo]
            
            for idx_conv, conv in enumerate(convergencias):
                historico = conv['historico']
                iteracoes = np.arange(1, len(historico) + 1)
                historico_viz = np.where(np.isinf(historico), np.nan, historico)
                
                ax.plot(iteracoes, historico_viz,
                       label=conv['label'],
                       color=cores[idx_conv % len(cores)],
                       linewidth=2,
                       marker='o',
                       markersize=4,
                       alpha=0.7)
            
            ax.set_xlabel("Iteração (Época)", fontsize=11, fontweight='bold')
            ax.set_ylabel("Melhor Fitness", fontsize=11, fontweight='bold')
            ax.set_title(nome_grupo, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=9)
        
        plt.tight_layout()
        
        if salvar_em:
            Path(salvar_em).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(salvar_em, dpi=self.dpi, bbox_inches='tight')
            if self.verbose:
                print(f"✓ Gráficos salvos em: {salvar_em}")
        
        plt.show()
        return fig, axes
    
    def obter_resumo_convergencia(self):
        """
        Retorna resumo estatístico de todas as convergências.
        
        Returns:
            pd.DataFrame: Tabela com estatísticas de cada convergência
        """
        resumos = []
        
        for conv in self.convergencias:
            historico = conv['historico']
            # Remover infinitos para cálculo
            historico_valido = historico[~np.isinf(historico)]
            
            resumo = {
                'Label': conv['label'],
                'Iterações': conv['iteracoes'],
                'Melhor Fitness': float(np.nanmin(historico_valido)) if len(historico_valido) > 0 else np.nan,
                'Fitness Inicial': float(historico_valido[0]) if len(historico_valido) > 0 else np.nan,
                'Melhoria (%)': float((historico_valido[0] - np.nanmin(historico_valido)) / historico_valido[0] * 100) if len(historico_valido) > 0 and historico_valido[0] != 0 else 0,
                'Variância': float(np.nanvar(historico_valido)) if len(historico_valido) > 0 else np.nan,
            }
            
            # Adicionar dados adicionais
            if conv['dados_adicionais']:
                for chave, valor in conv['dados_adicionais'].items():
                    resumo[chave] = valor
            
            resumos.append(resumo)
        
        return pd.DataFrame(resumos)
    
    def exibir_resumo(self):
        """Exibe resumo formatado da convergência."""
        resumo = self.obter_resumo_convergencia()
        
        print("\n" + "="*100)
        print("RESUMO DE CONVERGÊNCIA")
        print("="*100)
        print(resumo.to_string(index=False))
        print("="*100 + "\n")
    
    def analisar_convergencia(self, threshold_melhoria=0.01):
        """
        Analisa quando cada convergência parou de melhorar significativamente.
        
        Args:
            threshold_melhoria (float): Threshold de melhoria relativa para considerar "convergido"
        
        Returns:
            dict: {label: iteracao_convergencia}
        """
        analise = {}
        
        for conv in self.convergencias:
            historico = conv['historico']
            historico_valido = historico[~np.isinf(historico)]
            
            if len(historico_valido) < 2:
                analise[conv['label']] = 1
                continue
            
            # Calcular melhoria relativa iteração a iteração
            melhorias = []
            for i in range(1, len(historico_valido)):
                if historico_valido[i-1] != 0:
                    melhoria_rel = abs(historico_valido[i] - historico_valido[i-1]) / abs(historico_valido[i-1])
                    melhorias.append(melhoria_rel)
                else:
                    melhorias.append(0)
            
            # Encontrar iteração onde melhoria fica abaixo do threshold
            iteracao_convergencia = len(historico_valido)
            for i, melhoria in enumerate(melhorias):
                if melhoria < threshold_melhoria:
                    iteracao_convergencia = i + 2  # +1 para índice, +1 porque iteração começa em 1
                    break
            
            analise[conv['label']] = iteracao_convergencia
        
        return analise
    
    def exibir_analise_convergencia(self, threshold_melhoria=0.01):
        """Exibe análise de convergência formatada."""
        analise = self.analisar_convergencia(threshold_melhoria)
        
        print("\n" + "="*80)
        print(f"ANÁLISE DE CONVERGÊNCIA (Threshold: {threshold_melhoria*100:.2f}% melhoria)")
        print("="*80)
        
        for label, iteracao in sorted(analise.items(), key=lambda x: x[1]):
            print(f"  {label:<50} → Iteração {iteracao}")
        
        print("="*80 + "\n")
    
    def limpar(self):
        """Limpa todas as convergências adicionadas."""
        self.convergencias = []
        if self.verbose:
            print("✓ Visualizador limpo")


class ConvergenciaTracker:
    """
    Helper para rastrear convergência durante a otimização.
    
    Mantém histórico do melhor fitness encontrado em cada iteração, o
    fitness bruto por avaliação e, opcionalmente, o custo real (só dos
    diâmetros) por avaliação.
    """
    
    def __init__(self):
        """Inicializa o tracker."""
        self.historico = []              # melhor fitness acumulado (best-so-far)
        self.historico_bruto = []        # fitness bruto por avaliação
        self.historico_custo_real = []   # custo real (somente diâmetros) por avaliação (opcional)
        self.melhor_fitness = float('inf')
        self.iteracao_atual = 0
    
    def adicionar(self, fitness, custo_real=None):
        """
        Adiciona um novo fitness ao histórico.
        
        Args:
            fitness (float): Fitness encontrado nesta avaliação/iteração
            custo_real (float, optional): Custo real dos diâmetros nesta avaliação
        """
        self.iteracao_atual += 1
        
        # Armazenar fitness bruto
        self.historico_bruto.append(float(fitness))
        
        # Atualizar melhor fitness
        if fitness < self.melhor_fitness:
            self.melhor_fitness = fitness
        
        # Armazenar melhor fitness até agora (convergência)
        self.historico.append(self.melhor_fitness)
        
        # Armazenar custo real se informado
        if custo_real is not None:
            self.historico_custo_real.append(float(custo_real))
    
    def obter_historico(self):
        """Retorna o histórico de convergência."""
        return np.asarray(self.historico)
    
    def obter_historico_bruto(self):
        """Retorna o histórico de fitness bruto por avaliação."""
        return np.asarray(self.historico_bruto)
    
    def obter_historico_custo_real(self):
        """Retorna o histórico de custo real por avaliação (se disponível)."""
        return np.asarray(self.historico_custo_real)
    
    def obter_melhor_fitness(self):
        """Retorna o melhor fitness encontrado."""
        return self.melhor_fitness
    
    def limpar(self):
        """Reseta o tracker."""
        self.historico = []
        self.historico_bruto = []
        self.historico_custo_real = []
        self.melhor_fitness = float('inf')
        self.iteracao_atual = 0

    # -------------------------
    # Utilitários de visualização
    # -------------------------
    def acumular_melhor_custo_real(self):
        """
        Retorna a sequência best-so-far para custo real, se disponível.
        """
        import numpy as np
        if not self.historico_custo_real:
            return np.array([])
        arr = np.asarray(self.historico_custo_real, dtype=float)
        return np.minimum.accumulate(arr)

