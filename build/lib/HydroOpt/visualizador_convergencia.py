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
import json


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
    
    def adicionar_tracker(self, tracker, label, dados_adicionais=None):
        """
        Adiciona dados de um ConvergenciaTracker diretamente.
        
        Args:
            tracker (ConvergenciaTracker): Tracker com dados de convergência
            label (str): Rótulo para a curva
            dados_adicionais (dict, optional): Informações extras
        """
        historico = tracker.obter_historico()
        
        if len(historico) == 0:
            raise ValueError("Tracker não contém dados")
        
        stats = tracker.obter_estatisticas()
        
        # Mesclar estatísticas com dados adicionais
        dados = dados_adicionais.copy() if dados_adicionais else {}
        dados.update(stats)
        
        self.adicionar_convergencia(historico, label, dados_adicionais=dados)
        
        # Guardar referência ao tracker completo para plotar_detalhado
        self.convergencias[-1]['tracker'] = tracker
    
    def plotar_detalhado(self, tracker=None, titulo=None, salvar_em=None, mostrar=True):
        """
        Gera gráfico multi-painel com análise detalhada de convergência.
        
        Painéis:
        1. Fitness bruto (dispersão) + best-so-far (linha)
        2. Custo real dos diâmetros (evolução)
        3. Pressão mínima (evolução + linha de referência)
        4. Percentual de soluções viáveis (acumulado)
        
        Args:
            tracker (ConvergenciaTracker, optional): Tracker a plotar. Se None,
                usa o último tracker adicionado via adicionar_tracker()
            titulo (str, optional): Título geral
            salvar_em (str, optional): Caminho para salvar a figura
            mostrar (bool): Exibir o gráfico
        
        Returns:
            tuple: (fig, axes) - Figura e array de eixos matplotlib
        """
        # Obter tracker
        if tracker is None:
            for conv in reversed(self.convergencias):
                if 'tracker' in conv:
                    tracker = conv['tracker']
                    break
            if tracker is None:
                raise ValueError(
                    "Nenhum tracker disponível. Passe um ConvergenciaTracker como argumento "
                    "ou use adicionar_tracker() primeiro."
                )
        
        titulo = titulo or "Análise Detalhada de Convergência"
        
        # Obter dados
        fitness_bruto = tracker.obter_historico_bruto()
        fitness_bsf = tracker.obter_historico()
        custo_real = tracker.obter_historico_custo_real()
        custo_real_bsf = tracker.acumular_melhor_custo_real()
        pressao_min = tracker.obter_historico_pressao_min()
        viavel = tracker.obter_historico_viavel()
        
        n = len(fitness_bruto)
        avaliacoes = np.arange(1, n + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=self.dpi)
        fig.suptitle(titulo, fontsize=16, fontweight='bold')
        
        # --- Painel 1: Fitness ---
        ax1 = axes[0, 0]
        cores_viavel = np.where(viavel, '#2ca02c', '#d62728')
        ax1.scatter(avaliacoes, fitness_bruto, c=cores_viavel, s=8, alpha=0.3, label='Avaliações')
        ax1.plot(avaliacoes, fitness_bsf, color='#1f77b4', linewidth=2, label='Melhor acumulado')
        ax1.set_xlabel('Avaliação', fontsize=11)
        ax1.set_ylabel('Fitness', fontsize=11)
        ax1.set_title('Evolução do Fitness', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # --- Painel 2: Custo Real ---
        ax2 = axes[0, 1]
        mask_custo = ~np.isnan(custo_real)
        if mask_custo.any():
            ax2.scatter(avaliacoes[mask_custo], custo_real[mask_custo], s=10, alpha=0.3,
                        color='#ff7f0e', label='Custo por avaliação')
            mask_bsf = ~np.isnan(custo_real_bsf)
            if mask_bsf.any():
                ax2.plot(avaliacoes[mask_bsf], custo_real_bsf[mask_bsf], color='#d62728',
                        linewidth=2, label='Melhor custo acumulado')
        ax2.set_xlabel('Avaliação', fontsize=11)
        ax2.set_ylabel('Custo Real (R$)', fontsize=11)
        ax2.set_title('Evolução do Custo Real', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # --- Painel 3: Pressão Mínima ---
        ax3 = axes[1, 0]
        mask_pressao = ~np.isnan(pressao_min)
        if mask_pressao.any():
            cores_p = np.where(viavel[mask_pressao], '#2ca02c', '#d62728')
            ax3.scatter(avaliacoes[mask_pressao], pressao_min[mask_pressao],
                       c=cores_p, s=10, alpha=0.3, label='Pressão por avaliação')
        # Linha de referência (pressão desejada)
        ax3.axhline(y=10.0, color='red', linestyle='--', linewidth=1.5,
                     alpha=0.7, label='Pressão mínima desejada')
        ax3.set_xlabel('Avaliação', fontsize=11)
        ax3.set_ylabel('Pressão Mínima (m)', fontsize=11)
        ax3.set_title('Evolução da Pressão Mínima', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # --- Painel 4: Viabilidade ---
        ax4 = axes[1, 1]
        viavel_cumsum = np.cumsum(viavel)
        percentual_viavel = viavel_cumsum / avaliacoes * 100
        ax4.plot(avaliacoes, percentual_viavel, color='#2ca02c', linewidth=2)
        ax4.fill_between(avaliacoes, 0, percentual_viavel, alpha=0.2, color='#2ca02c')
        ax4.set_xlabel('Avaliação', fontsize=11)
        ax4.set_ylabel('Soluções Viáveis (%)', fontsize=11)
        ax4.set_title('Percentual de Soluções Viáveis', fontsize=12, fontweight='bold')
        ax4.set_ylim(0, 105)
        ax4.grid(True, alpha=0.3, linestyle='--')
        
        # Texto com estatísticas
        stats = tracker.obter_estatisticas()
        textstr = (f"Total: {stats['total_avaliacoes']}\n"
                   f"Viáveis: {stats['avaliacoes_viaveis']} ({stats['percentual_viaveis']:.1f}%)\n"
                   f"Melhor fitness: {stats['melhor_fitness']:.2f}")
        if 'melhor_custo_real' in stats:
            textstr += f"\nMelhor custo: R$ {stats['melhor_custo_real']:,.2f}"
        ax4.text(0.98, 0.02, textstr, transform=ax4.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if salvar_em:
            Path(salvar_em).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(salvar_em, dpi=self.dpi, bbox_inches='tight')
            if self.verbose:
                print(f"✓ Gráfico detalhado salvo em: {salvar_em}")
        
        if mostrar:
            plt.show()
        
        return fig, axes
    
    def plotar_comparativo_trackers(self, trackers_dict, titulo=None, salvar_em=None, mostrar=True):
        """
        Compara múltiplos trackers em gráficos sobrepostos.
        
        Args:
            trackers_dict (dict): {label: ConvergenciaTracker}
            titulo (str, optional): Título do gráfico
            salvar_em (str, optional): Caminho para salvar
            mostrar (bool): Exibir o gráfico
        
        Returns:
            tuple: (fig, axes)
        """
        titulo = titulo or "Comparação de Otimizações"
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=self.dpi)
        fig.suptitle(titulo, fontsize=16, fontweight='bold')
        
        cores = plt.cm.tab10(np.linspace(0, 1, max(len(trackers_dict), 1)))
        
        for idx, (label, tracker) in enumerate(trackers_dict.items()):
            cor = cores[idx]
            
            fitness_bsf = tracker.obter_historico()
            custo_bsf = tracker.acumular_melhor_custo_real()
            pressao = tracker.obter_historico_pressao_min()
            n = len(fitness_bsf)
            avals = np.arange(1, n + 1)
            
            # Fitness best-so-far
            axes[0].plot(avals, fitness_bsf, color=cor, linewidth=2, label=label, alpha=0.8)
            
            # Custo real best-so-far
            mask = ~np.isnan(custo_bsf)
            if mask.any():
                axes[1].plot(avals[mask], custo_bsf[mask], color=cor, linewidth=2, label=label, alpha=0.8)
            
            # Pressão mínima
            mask_p = ~np.isnan(pressao)
            if mask_p.any():
                axes[2].scatter(avals[mask_p], pressao[mask_p], color=cor, s=6, alpha=0.2, label=label)
        
        axes[0].set_title('Fitness (Best-so-far)', fontweight='bold')
        axes[0].set_xlabel('Avaliação')
        axes[0].set_ylabel('Fitness')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        
        axes[1].set_title('Custo Real (Best-so-far)', fontweight='bold')
        axes[1].set_xlabel('Avaliação')
        axes[1].set_ylabel('Custo Real (R$)')
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        
        axes[2].set_title('Pressão Mínima', fontweight='bold')
        axes[2].set_xlabel('Avaliação')
        axes[2].set_ylabel('Pressão (m)')
        axes[2].axhline(y=10.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        axes[2].grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if salvar_em:
            Path(salvar_em).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(salvar_em, dpi=self.dpi, bbox_inches='tight')
            if self.verbose:
                print(f"✓ Gráfico comparativo salvo em: {salvar_em}")
        
        if mostrar:
            plt.show()
        
        return fig, axes
    
    def limpar(self):
        """Limpa todas as convergências adicionadas."""
        self.convergencias = []
        if self.verbose:
            print("✓ Visualizador limpo")


class ConvergenciaTracker:
    """
    Rastreador completo de convergência durante a otimização.
    
    Mantém histórico detalhado de cada avaliação incluindo:
    - Fitness bruto e melhor acumulado (best-so-far)
    - Custo real (somente diâmetros)
    - Pressão mínima da rede
    - Viabilidade da solução
    - Solução completa (opcional, configurável)
    
    Todos os dados são mantidos em memória durante a otimização e podem
    ser exportados para DataFrame, CSV ou JSON ao final.
    
    Exemplo:
        >>> tracker = ConvergenciaTracker(salvar_solucoes=True)
        >>> # Durante otimização (automático):
        >>> tracker.adicionar(fitness=1500.0, custo_real=1200.0, pressao_min=12.5, viavel=True)
        >>> # Após otimização:
        >>> df = tracker.to_dataframe()
        >>> tracker.exportar_csv('convergencia.csv')
        >>> tracker.exportar_json('convergencia.json')
        >>> stats = tracker.obter_estatisticas()
    """
    
    def __init__(self, salvar_solucoes=False):
        """
        Inicializa o tracker.
        
        Args:
            salvar_solucoes (bool): Se True, salva a solução completa (vetor de diâmetros)
                                    a cada avaliação. Consome mais memória mas permite
                                    análise detalhada de como as soluções evoluíram.
        """
        self.salvar_solucoes = salvar_solucoes
        
        # Dados por avaliação
        self.historico_bruto = []          # fitness bruto por avaliação
        self.historico = []                # melhor fitness acumulado (best-so-far)
        self.historico_custo_real = []     # custo real (diâmetros) por avaliação
        self.historico_pressao_min = []    # pressão mínima por avaliação
        self.historico_viavel = []         # viabilidade por avaliação
        self.historico_solucoes = []       # soluções completas (se salvar_solucoes=True)
        
        # Estado
        self.melhor_fitness = float('inf')
        self.melhor_custo_real = float('inf')
        self.melhor_solucao = None
        self.iteracao_atual = 0
    
    def adicionar(self, fitness, custo_real=None, pressao_min=None, viavel=False, solucao=None):
        """
        Registra dados de uma avaliação.
        
        Args:
            fitness (float): Valor da função objetivo nesta avaliação
            custo_real (float, optional): Custo real dos diâmetros (sem penalidades)
            pressao_min (float, optional): Pressão mínima da rede nesta avaliação
            viavel (bool): Se a solução atende à restrição de pressão mínima
            solucao (array-like, optional): Vetor solução (ignorado se salvar_solucoes=False)
        """
        self.iteracao_atual += 1
        
        # Fitness bruto
        self.historico_bruto.append(float(fitness))
        
        # Atualizar melhor fitness (best-so-far)
        if fitness < self.melhor_fitness:
            self.melhor_fitness = fitness
        self.historico.append(self.melhor_fitness)
        
        # Custo real (np.nan quando não disponível)
        if custo_real is not None:
            self.historico_custo_real.append(float(custo_real))
            if viavel and custo_real < self.melhor_custo_real:
                self.melhor_custo_real = custo_real
        else:
            self.historico_custo_real.append(np.nan)
        
        # Pressão mínima
        if pressao_min is not None:
            self.historico_pressao_min.append(float(pressao_min))
        else:
            self.historico_pressao_min.append(np.nan)
        
        # Viabilidade
        self.historico_viavel.append(bool(viavel))
        
        # Solução completa (se configurado)
        if self.salvar_solucoes and solucao is not None:
            self.historico_solucoes.append(np.asarray(solucao, dtype=float).tolist())
        elif self.salvar_solucoes:
            self.historico_solucoes.append(None)
        
        # Atualizar melhor solução viável
        if viavel and solucao is not None:
            if custo_real is not None and custo_real <= self.melhor_custo_real:
                self.melhor_solucao = np.asarray(solucao, dtype=float).copy()
    
    # -----------------------------------------------------------
    # Acesso aos dados
    # -----------------------------------------------------------
    def obter_historico(self):
        """Retorna histórico best-so-far de fitness."""
        return np.asarray(self.historico, dtype=float)
    
    def obter_historico_bruto(self):
        """Retorna histórico de fitness bruto por avaliação."""
        return np.asarray(self.historico_bruto, dtype=float)
    
    def obter_historico_custo_real(self):
        """Retorna histórico de custo real por avaliação."""
        return np.asarray(self.historico_custo_real, dtype=float)
    
    def obter_historico_pressao_min(self):
        """Retorna histórico de pressão mínima por avaliação."""
        return np.asarray(self.historico_pressao_min, dtype=float)
    
    def obter_historico_viavel(self):
        """Retorna histórico de viabilidade por avaliação."""
        return np.asarray(self.historico_viavel, dtype=bool)
    
    def obter_melhor_fitness(self):
        """Retorna o melhor fitness encontrado."""
        return self.melhor_fitness
    
    def acumular_melhor_custo_real(self):
        """
        Retorna a sequência best-so-far para custo real, alinhada às avaliações.
        Apenas soluções viáveis são consideradas.
        """
        if not self.historico_custo_real:
            return np.array([])
        arr = np.asarray(self.historico_custo_real, dtype=float)
        viavel_arr = np.asarray(self.historico_viavel, dtype=bool)
        best = np.full(arr.shape, np.nan)
        current_best = np.nan
        for i in range(len(arr)):
            if viavel_arr[i] and not np.isnan(arr[i]):
                current_best = arr[i] if np.isnan(current_best) else min(current_best, arr[i])
            best[i] = current_best
        return best
    
    def acumular_melhor_pressao_min(self):
        """
        Retorna a sequência best-so-far para pressão mínima (somente viáveis).
        """
        if not self.historico_pressao_min:
            return np.array([])
        arr = np.asarray(self.historico_pressao_min, dtype=float)
        viavel_arr = np.asarray(self.historico_viavel, dtype=bool)
        best = np.full(arr.shape, np.nan)
        current_best = np.nan
        for i in range(len(arr)):
            if viavel_arr[i] and not np.isnan(arr[i]):
                current_best = arr[i] if np.isnan(current_best) else max(current_best, arr[i])
            best[i] = current_best
        return best
    
    # -----------------------------------------------------------
    # Exportação
    # -----------------------------------------------------------
    def to_dataframe(self):
        """
        Exporta todos os dados de convergência para um DataFrame pandas.
        
        Returns:
            pd.DataFrame: DataFrame com colunas:
                - avaliacao: número sequencial da avaliação
                - fitness_bruto: valor da função objetivo
                - fitness_melhor: melhor acumulado (best-so-far)
                - custo_real: custo real dos diâmetros
                - custo_real_melhor: melhor custo real acumulado
                - pressao_min: pressão mínima da rede
                - viavel: se a solução é viável
        """
        n = len(self.historico_bruto)
        if n == 0:
            return pd.DataFrame()
        
        dados = {
            'avaliacao': list(range(1, n + 1)),
            'fitness_bruto': self.historico_bruto,
            'fitness_melhor': self.historico,
            'custo_real': self.historico_custo_real,
            'custo_real_melhor': self.acumular_melhor_custo_real().tolist(),
            'pressao_min': self.historico_pressao_min,
            'viavel': self.historico_viavel,
        }
        
        df = pd.DataFrame(dados)
        
        # Adicionar soluções se disponíveis
        if self.salvar_solucoes and self.historico_solucoes:
            df['solucao'] = self.historico_solucoes
        
        return df
    
    def exportar_csv(self, caminho):
        """
        Exporta dados de convergência para CSV.
        
        Args:
            caminho (str): Caminho do arquivo CSV de saída
        
        Returns:
            str: Caminho do arquivo salvo
        """
        df = self.to_dataframe()
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(caminho, index=False)
        print(f"✓ Dados de convergência exportados para: {caminho}")
        return caminho
    
    def exportar_json(self, caminho):
        """
        Exporta dados de convergência para JSON com estrutura completa.
        
        Args:
            caminho (str): Caminho do arquivo JSON de saída
        
        Returns:
            str: Caminho do arquivo salvo
        """
        dados = {
            'total_avaliacoes': self.iteracao_atual,
            'melhor_fitness': float(self.melhor_fitness),
            'melhor_custo_real': float(self.melhor_custo_real) if self.melhor_custo_real != float('inf') else None,
            'avaliacoes': []
        }
        
        for i in range(len(self.historico_bruto)):
            avaliacao = {
                'id': i + 1,
                'fitness_bruto': self.historico_bruto[i],
                'fitness_melhor': self.historico[i],
                'custo_real': self.historico_custo_real[i] if not np.isnan(self.historico_custo_real[i]) else None,
                'pressao_min': self.historico_pressao_min[i] if i < len(self.historico_pressao_min) and not np.isnan(self.historico_pressao_min[i]) else None,
                'viavel': self.historico_viavel[i] if i < len(self.historico_viavel) else None,
            }
            if self.salvar_solucoes and i < len(self.historico_solucoes):
                avaliacao['solucao'] = self.historico_solucoes[i]
            dados['avaliacoes'].append(avaliacao)
        
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        with open(caminho, 'w', encoding='utf-8') as f:
            json.dump(dados, f, indent=2, ensure_ascii=False)
        print(f"✓ Dados de convergência exportados para: {caminho}")
        return caminho
    
    # -----------------------------------------------------------
    # Estatísticas resumidas
    # -----------------------------------------------------------
    def obter_estatisticas(self):
        """
        Retorna dicionário com estatísticas de convergência.
        
        Returns:
            dict: Estatísticas incluindo total de avaliações, viáveis, melhor fitness, etc.
        """
        n = len(self.historico_bruto)
        if n == 0:
            return {'total_avaliacoes': 0}
        
        arr_fitness = np.asarray(self.historico_bruto, dtype=float)
        arr_viavel = np.asarray(self.historico_viavel, dtype=bool)
        arr_custo = np.asarray(self.historico_custo_real, dtype=float)
        arr_pressao = np.asarray(self.historico_pressao_min, dtype=float)
        
        n_viaveis = int(arr_viavel.sum())
        custos_viaveis = arr_custo[arr_viavel & ~np.isnan(arr_custo)]
        pressoes_viaveis = arr_pressao[arr_viavel & ~np.isnan(arr_pressao)]
        
        stats = {
            'total_avaliacoes': n,
            'avaliacoes_viaveis': n_viaveis,
            'percentual_viaveis': (n_viaveis / n * 100) if n > 0 else 0,
            'melhor_fitness': float(self.melhor_fitness),
            'fitness_medio': float(np.nanmean(arr_fitness)),
            'fitness_desvio': float(np.nanstd(arr_fitness)),
        }
        
        if len(custos_viaveis) > 0:
            stats['melhor_custo_real'] = float(np.nanmin(custos_viaveis))
            stats['custo_real_medio'] = float(np.nanmean(custos_viaveis))
            stats['custo_real_desvio'] = float(np.nanstd(custos_viaveis))
        
        if len(pressoes_viaveis) > 0:
            stats['pressao_min_melhor_viavel'] = float(np.nanmin(pressoes_viaveis))
            stats['pressao_max_melhor_viavel'] = float(np.nanmax(pressoes_viaveis))
            stats['pressao_media_viavel'] = float(np.nanmean(pressoes_viaveis))
        
        return stats
    
    def exibir_estatisticas(self):
        """Exibe estatísticas formatadas de convergência."""
        stats = self.obter_estatisticas()
        
        print("\n" + "="*70)
        print("ESTATÍSTICAS DE CONVERGÊNCIA")
        print("="*70)
        print(f"  Total de avaliações:     {stats.get('total_avaliacoes', 0)}")
        print(f"  Avaliações viáveis:      {stats.get('avaliacoes_viaveis', 0)} ({stats.get('percentual_viaveis', 0):.1f}%)")
        print(f"  Melhor fitness:          {stats.get('melhor_fitness', 'N/A'):.6f}")
        print(f"  Fitness médio:           {stats.get('fitness_medio', 'N/A'):.2f}")
        print(f"  Desvio fitness:          {stats.get('fitness_desvio', 'N/A'):.2f}")
        
        if 'melhor_custo_real' in stats:
            print(f"\n  💰 Melhor custo real:    R$ {stats['melhor_custo_real']:,.2f}")
            print(f"  Custo real médio:        R$ {stats.get('custo_real_medio', 0):,.2f}")
        
        if 'pressao_min_melhor_viavel' in stats:
            print(f"\n  Pressão mín (viáveis):   {stats['pressao_min_melhor_viavel']:.2f} m")
            print(f"  Pressão máx (viáveis):   {stats['pressao_max_melhor_viavel']:.2f} m")
            print(f"  Pressão média (viáveis): {stats['pressao_media_viavel']:.2f} m")
        
        print("="*70 + "\n")
    
    def limpar(self):
        """Reseta o tracker."""
        self.historico = []
        self.historico_bruto = []
        self.historico_custo_real = []
        self.historico_pressao_min = []
        self.historico_viavel = []
        self.historico_solucoes = []
        self.melhor_fitness = float('inf')
        self.melhor_custo_real = float('inf')
        self.melhor_solucao = None
        self.iteracao_atual = 0

