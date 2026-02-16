"""
Módulo para análise estatística pós-otimização.

Processa os dados coletados pelo ConvergenciaTracker (ou logs brutos)
e calcula métricas detalhadas sobre o comportamento do enxame/população.

Uso típico:
    >>> resultado = otimizador.otimizar(metodo='WOA', rastrear_convergencia=True)
    >>> analise = AnalisadorEstatistico(resultado)
    >>> analise.exibir_resumo()
    >>> df = analise.to_dataframe()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any


class AnalisadorEstatistico:
    """
    Análise estatística detalhada de uma execução de otimização.

    Opera sobre os dados retornados por ``Otimizador.otimizar()``
    (ou sobre um ``ConvergenciaTracker`` diretamente) e calcula:

    * Erro médio / mediano / desvio padrão de cada partícula em relação ao melhor
    * Taxa de melhoria por época
    * Diversidade do enxame (desvio padrão entre indivíduos)
    * Detecção de estagnação
    * Percentual de soluções viáveis por época
    * Distribuição de fitness por quartis
    * Ranking de épocas mais produtivas (maior melhoria)

    Exemplo:
        >>> resultado = otimizador.otimizar(metodo='PSO', rastrear_convergencia=True)
        >>> analise = AnalisadorEstatistico(resultado)
        >>> analise.exibir_resumo()
        >>> analise.exibir_analise_particulas()
        >>> df = analise.to_dataframe_epocas()
    """

    def __init__(self, resultado: Dict[str, Any] = None, tracker=None):
        """
        Inicializa o analisador.

        Aceita **um** dos dois:
        * ``resultado`` — dicionário retornado por ``Otimizador.otimizar()``
        * ``tracker``   — instância de ``ConvergenciaTracker``

        Args:
            resultado (dict, optional): Dicionário com chaves geradas pelo otimizador.
            tracker (ConvergenciaTracker, optional): Tracker direto.
        """
        if resultado is None and tracker is None:
            raise ValueError(
                "Forneça 'resultado' (dict do otimizador) ou 'tracker' (ConvergenciaTracker)."
            )

        # Extrair tracker do resultado se necessário
        if tracker is not None:
            self.tracker = tracker
        else:
            if 'tracker' not in resultado:
                raise KeyError(
                    "O dicionário de resultado não contém 'tracker'. "
                    "Certifique-se de ter usado rastrear_convergencia=True."
                )
            self.tracker = resultado['tracker']

        # Cache de dados brutos
        self._fitness_bruto = self.tracker.obter_historico_bruto()
        self._fitness_bsf = self.tracker.obter_historico()
        self._viavel = self.tracker.obter_historico_viavel()
        self._custo_real = self.tracker.obter_historico_custo_real()
        self._pressao_min = self.tracker.obter_historico_pressao_min()
        self._epocas = self.tracker.epocas  # list of dicts por época

        # Dados adicionais do resultado
        self._seed_usado = None
        self._metodo = None
        self._melhor_custo = None
        self._custo_real_final = None
        if resultado is not None:
            self._seed_usado = resultado.get('seed_usado')
            self._melhor_custo = resultado.get('melhor_custo')
            self._custo_real_final = resultado.get('custo_real')

        # Cache de métricas calculadas
        self._metricas_cache = None

    # ==================================================================
    # Cálculo principal
    # ==================================================================
    def calcular(self) -> Dict[str, Any]:
        """
        Calcula todas as métricas estatísticas.

        Returns:
            dict: Dicionário completo com todas as métricas.
        """
        if self._metricas_cache is not None:
            return self._metricas_cache

        m: Dict[str, Any] = {}

        n_total = len(self._fitness_bruto)
        n_viaveis = int(self._viavel.sum()) if len(self._viavel) > 0 else 0
        n_epocas = len(self._epocas)

        # ----- Métricas globais -----
        m['total_avaliacoes'] = n_total
        m['total_epocas'] = n_epocas
        m['avaliacoes_viaveis'] = n_viaveis
        m['percentual_viaveis'] = (n_viaveis / n_total * 100) if n_total > 0 else 0.0
        m['melhor_fitness'] = float(self._fitness_bsf[-1]) if n_total > 0 else np.nan
        m['pior_fitness'] = float(np.max(self._fitness_bruto)) if n_total > 0 else np.nan
        m['fitness_medio_global'] = float(np.mean(self._fitness_bruto)) if n_total > 0 else np.nan
        m['fitness_mediano_global'] = float(np.median(self._fitness_bruto)) if n_total > 0 else np.nan
        m['fitness_desvio_global'] = float(np.std(self._fitness_bruto)) if n_total > 0 else np.nan
        m['seed_usado'] = self._seed_usado

        # ----- Erro das partículas em relação ao melhor -----
        if n_total > 0:
            melhor = m['melhor_fitness']
            erros = self._fitness_bruto - melhor
            m['erro_medio_particulas'] = float(np.mean(erros))
            m['erro_mediano_particulas'] = float(np.median(erros))
            m['erro_desvio_particulas'] = float(np.std(erros))
            m['erro_maximo_particulas'] = float(np.max(erros))
            m['erro_minimo_particulas'] = float(np.min(erros))

            # Quartis do erro
            q25, q50, q75 = np.percentile(erros, [25, 50, 75])
            m['erro_q25'] = float(q25)
            m['erro_q50'] = float(q50)
            m['erro_q75'] = float(q75)
            m['erro_iqr'] = float(q75 - q25)
        else:
            for k in ['erro_medio_particulas', 'erro_mediano_particulas',
                       'erro_desvio_particulas', 'erro_maximo_particulas',
                       'erro_minimo_particulas', 'erro_q25', 'erro_q50',
                       'erro_q75', 'erro_iqr']:
                m[k] = np.nan

        # ----- Análise por época -----
        if n_epocas > 0:
            m['epocas_detalhes'] = self._analisar_epocas()

            # Taxa de melhoria entre épocas
            melhores_por_epoca = np.array([e['melhor'] for e in self._epocas])
            bsf_por_epoca = np.minimum.accumulate(melhores_por_epoca)
            taxas_melhoria = np.zeros(n_epocas)
            for i in range(1, n_epocas):
                if bsf_por_epoca[i - 1] != 0:
                    taxas_melhoria[i] = (bsf_por_epoca[i - 1] - bsf_por_epoca[i]) / abs(bsf_por_epoca[i - 1])
            m['taxa_melhoria_media'] = float(np.mean(taxas_melhoria[1:]))
            m['taxa_melhoria_max'] = float(np.max(taxas_melhoria))

            # Diversidade do enxame (desvio padrão médio por época)
            desvios = np.array([e['desvio'] for e in self._epocas])
            m['diversidade_media'] = float(np.mean(desvios))
            m['diversidade_inicial'] = float(desvios[0]) if len(desvios) > 0 else np.nan
            m['diversidade_final'] = float(desvios[-1]) if len(desvios) > 0 else np.nan
            m['perda_diversidade_pct'] = (
                float((desvios[0] - desvios[-1]) / desvios[0] * 100)
                if len(desvios) > 0 and desvios[0] != 0 else 0.0
            )

            # Detecção de estagnação
            m['estagnacao'] = self._detectar_estagnacao(bsf_por_epoca)

            # Época mais produtiva (maior melhoria)
            if len(taxas_melhoria) > 1:
                idx_melhor_epoca = int(np.argmax(taxas_melhoria[1:]) + 1)
                m['epoca_mais_produtiva'] = idx_melhor_epoca
                m['melhoria_na_melhor_epoca'] = float(taxas_melhoria[idx_melhor_epoca])
            else:
                m['epoca_mais_produtiva'] = 0
                m['melhoria_na_melhor_epoca'] = 0.0

            # Viabilidade por época
            m['viabilidade_por_epoca'] = self._viabilidade_por_epoca()
        else:
            m['epocas_detalhes'] = []
            m['taxa_melhoria_media'] = np.nan
            m['taxa_melhoria_max'] = np.nan
            m['diversidade_media'] = np.nan
            m['diversidade_inicial'] = np.nan
            m['diversidade_final'] = np.nan
            m['perda_diversidade_pct'] = np.nan
            m['estagnacao'] = {}
            m['epoca_mais_produtiva'] = 0
            m['melhoria_na_melhor_epoca'] = 0.0
            m['viabilidade_por_epoca'] = []

        # ----- Custo real (somente viáveis) -----
        custos_viaveis = self._custo_real[self._viavel & ~np.isnan(self._custo_real)]
        if len(custos_viaveis) > 0:
            m['custo_real_melhor'] = float(np.min(custos_viaveis))
            m['custo_real_medio'] = float(np.mean(custos_viaveis))
            m['custo_real_mediano'] = float(np.median(custos_viaveis))
            m['custo_real_desvio'] = float(np.std(custos_viaveis))
        else:
            m['custo_real_melhor'] = np.nan
            m['custo_real_medio'] = np.nan
            m['custo_real_mediano'] = np.nan
            m['custo_real_desvio'] = np.nan

        # ----- Pressão mínima (somente viáveis) -----
        pressoes_viaveis = self._pressao_min[self._viavel & ~np.isnan(self._pressao_min)]
        if len(pressoes_viaveis) > 0:
            m['pressao_min_minima'] = float(np.min(pressoes_viaveis))
            m['pressao_min_media'] = float(np.mean(pressoes_viaveis))
            m['pressao_min_maxima'] = float(np.max(pressoes_viaveis))
        else:
            m['pressao_min_minima'] = np.nan
            m['pressao_min_media'] = np.nan
            m['pressao_min_maxima'] = np.nan

        self._metricas_cache = m
        return m

    # ==================================================================
    # Helpers de cálculo
    # ==================================================================
    def _analisar_epocas(self) -> List[Dict]:
        """Retorna lista com métricas detalhadas de cada época."""
        resultado = []
        bsf = float('inf')
        for e in self._epocas:
            todos = np.array(e['todos'])
            bsf = min(bsf, e['melhor'])
            resultado.append({
                'epoca': e['epoca'],
                'melhor': e['melhor'],
                'bsf': bsf,
                'media': e['media'],
                'mediana': float(np.median(todos)),
                'pior': e['pior'],
                'desvio': e['desvio'],
                'q25': float(np.percentile(todos, 25)),
                'q75': float(np.percentile(todos, 75)),
                'n_particulas': len(todos),
            })
        return resultado

    def _detectar_estagnacao(self, bsf_por_epoca: np.ndarray, janela: int = 5,
                              threshold: float = 1e-6) -> Dict:
        """
        Detecta estagnação no best-so-far.

        Args:
            bsf_por_epoca: Sequência best-so-far por época.
            janela: Número mínimo de épocas sem melhoria para considerar estagnação.
            threshold: Melhoria relativa mínima para considerar "movimento".

        Returns:
            dict: {estagnado, epoca_inicio, duracao, epocas_sem_melhoria}
        """
        n = len(bsf_por_epoca)
        if n < 2:
            return {'estagnado': False, 'epoca_inicio': None,
                    'duracao': 0, 'epocas_sem_melhoria': 0}

        sem_melhoria = 0
        max_sem_melhoria = 0
        epoca_inicio_estagnacao = None
        melhor_estag_inicio = None

        for i in range(1, n):
            melhoria_rel = abs(bsf_por_epoca[i] - bsf_por_epoca[i - 1])
            if bsf_por_epoca[i - 1] != 0:
                melhoria_rel /= abs(bsf_por_epoca[i - 1])
            if melhoria_rel < threshold:
                sem_melhoria += 1
                if sem_melhoria == 1:
                    melhor_estag_inicio = i
            else:
                if sem_melhoria > max_sem_melhoria:
                    max_sem_melhoria = sem_melhoria
                    epoca_inicio_estagnacao = melhor_estag_inicio
                sem_melhoria = 0

        # Checar o bloco final
        if sem_melhoria > max_sem_melhoria:
            max_sem_melhoria = sem_melhoria
            epoca_inicio_estagnacao = melhor_estag_inicio

        return {
            'estagnado': max_sem_melhoria >= janela,
            'epoca_inicio': epoca_inicio_estagnacao,
            'duracao': max_sem_melhoria,
            'epocas_sem_melhoria': max_sem_melhoria,
        }

    def _viabilidade_por_epoca(self) -> List[Dict]:
        """Calcula percentual de soluções viáveis por época."""
        pop_size = self.tracker.pop_size or 1
        n = len(self._viavel)
        resultado = []
        for i, e in enumerate(self._epocas):
            inicio = i * pop_size
            fim = min(inicio + pop_size, n)
            bloco = self._viavel[inicio:fim]
            total = len(bloco)
            viaveis = int(bloco.sum()) if total > 0 else 0
            resultado.append({
                'epoca': e['epoca'],
                'total': total,
                'viaveis': viaveis,
                'percentual': (viaveis / total * 100) if total > 0 else 0.0,
            })
        return resultado

    # ==================================================================
    # Exportação
    # ==================================================================
    def to_dataframe_epocas(self) -> pd.DataFrame:
        """
        Exporta métricas por época como DataFrame.

        Returns:
            pd.DataFrame: Uma linha por época com melhor, média, desvio, quartis, etc.
        """
        m = self.calcular()
        detalhes = m.get('epocas_detalhes', [])
        if not detalhes:
            return pd.DataFrame()
        return pd.DataFrame(detalhes)

    def to_dataframe_global(self) -> pd.DataFrame:
        """
        Exporta as métricas globais como um DataFrame de uma linha.

        Returns:
            pd.DataFrame: Uma única linha com todas as métricas escalares.
        """
        m = self.calcular()
        # Filtrar somente chaves escalares
        escalares = {k: v for k, v in m.items()
                     if not isinstance(v, (list, dict, np.ndarray))}
        return pd.DataFrame([escalares])

    def to_dict(self) -> Dict[str, Any]:
        """Retorna todas as métricas como dicionário."""
        return self.calcular()

    def exportar_json(self, caminho: str):
        """
        Exporta métricas para JSON.

        Args:
            caminho: Caminho do arquivo de saída.
        """
        import json
        from pathlib import Path
        m = self.calcular()

        # Converter arrays numpy para listas
        def _converter(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _converter(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_converter(v) for v in obj]
            return obj

        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        with open(caminho, 'w', encoding='utf-8') as f:
            json.dump(_converter(m), f, indent=2, ensure_ascii=False)
        print(f"✓ Análise estatística exportada para: {caminho}")

    # ==================================================================
    # Exibição formatada
    # ==================================================================
    def exibir_resumo(self):
        """Exibe resumo completo das estatísticas no terminal."""
        m = self.calcular()

        print("\n" + "=" * 80)
        print("ANÁLISE ESTATÍSTICA PÓS-OTIMIZAÇÃO")
        print("=" * 80)

        if m.get('seed_usado') is not None:
            print(f"  🔑 Seed utilizada: {m['seed_usado']}")

        print(f"\n📊 VISÃO GERAL")
        print(f"  Total de avaliações:       {m['total_avaliacoes']}")
        print(f"  Total de épocas:           {m['total_epocas']}")
        print(f"  Avaliações viáveis:        {m['avaliacoes_viaveis']} ({m['percentual_viaveis']:.1f}%)")
        print(f"  Melhor fitness:            {m['melhor_fitness']:.6f}")
        print(f"  Pior fitness:              {m['pior_fitness']:.2f}")
        print(f"  Fitness médio global:      {m['fitness_medio_global']:.2f}")
        print(f"  Fitness mediano global:    {m['fitness_mediano_global']:.2f}")
        print(f"  Desvio padrão global:      {m['fitness_desvio_global']:.2f}")

        print(f"\n📈 ERRO DAS PARTÍCULAS (em relação ao melhor)")
        print(f"  Erro médio:                {m['erro_medio_particulas']:.2f}")
        print(f"  Erro mediano:              {m['erro_mediano_particulas']:.2f}")
        print(f"  Desvio padrão do erro:     {m['erro_desvio_particulas']:.2f}")
        print(f"  Erro máximo:               {m['erro_maximo_particulas']:.2f}")
        print(f"  Quartil 25%:               {m['erro_q25']:.2f}")
        print(f"  Quartil 75%:               {m['erro_q75']:.2f}")
        print(f"  IQR (amplitude inter-q.):  {m['erro_iqr']:.2f}")

        if m['total_epocas'] > 0:
            print(f"\n🔄 CONVERGÊNCIA")
            print(f"  Taxa de melhoria média:    {m['taxa_melhoria_media'] * 100:.4f}%")
            print(f"  Taxa de melhoria máxima:   {m['taxa_melhoria_max'] * 100:.4f}%")
            print(f"  Época mais produtiva:      {m['epoca_mais_produtiva']}")
            print(f"  Melhoria nessa época:      {m['melhoria_na_melhor_epoca'] * 100:.4f}%")

            print(f"\n🌀 DIVERSIDADE DO ENXAME")
            print(f"  Diversidade inicial:       {m['diversidade_inicial']:.2f}")
            print(f"  Diversidade final:         {m['diversidade_final']:.2f}")
            print(f"  Diversidade média:         {m['diversidade_media']:.2f}")
            print(f"  Perda de diversidade:      {m['perda_diversidade_pct']:.1f}%")

            estag = m['estagnacao']
            if estag.get('estagnado'):
                print(f"\n⚠️  ESTAGNAÇÃO DETECTADA")
                print(f"  Início na época:           {estag['epoca_inicio']}")
                print(f"  Duração:                   {estag['duracao']} épocas")
            else:
                print(f"\n✓  Sem estagnação significativa detectada")

        if not np.isnan(m.get('custo_real_melhor', np.nan)):
            print(f"\n💰 CUSTO REAL (somente soluções viáveis)")
            print(f"  Melhor custo real:         R$ {m['custo_real_melhor']:,.2f}")
            print(f"  Custo real médio:          R$ {m['custo_real_medio']:,.2f}")
            print(f"  Custo real mediano:        R$ {m['custo_real_mediano']:,.2f}")
            print(f"  Desvio custo real:         R$ {m['custo_real_desvio']:,.2f}")

        if not np.isnan(m.get('pressao_min_minima', np.nan)):
            print(f"\n💧 PRESSÃO MÍNIMA (viáveis)")
            print(f"  Mínima encontrada:         {m['pressao_min_minima']:.2f} m")
            print(f"  Média:                     {m['pressao_min_media']:.2f} m")
            print(f"  Máxima:                    {m['pressao_min_maxima']:.2f} m")

        print("\n" + "=" * 80 + "\n")

    def exibir_analise_particulas(self, top_n: int = 5):
        """
        Exibe análise detalhada das épocas com mais e menos variação.

        Args:
            top_n: Quantas épocas exibir de cada ranking.
        """
        m = self.calcular()
        detalhes = m.get('epocas_detalhes', [])
        if not detalhes:
            print("⚠️  Sem dados de épocas para analisar.")
            return

        print("\n" + "=" * 80)
        print("ANÁLISE DETALHADA POR ÉPOCA")
        print("=" * 80)

        # Top épocas com maior diversidade
        por_desvio = sorted(detalhes, key=lambda e: e['desvio'], reverse=True)
        print(f"\n🔝 TOP {top_n} ÉPOCAS COM MAIOR DIVERSIDADE:")
        print(f"  {'Época':<8} {'Melhor':<16} {'Média':<16} {'Desvio':<16} {'Pior':<16}")
        print(f"  {'-'*72}")
        for e in por_desvio[:top_n]:
            print(f"  {e['epoca']:<8} {e['melhor']:<16.2f} {e['media']:<16.2f} "
                  f"{e['desvio']:<16.2f} {e['pior']:<16.2f}")

        # Top épocas com menor diversidade (convergidas)
        print(f"\n🔻 TOP {top_n} ÉPOCAS MAIS CONVERGIDAS:")
        por_desvio_asc = sorted(detalhes, key=lambda e: e['desvio'])
        print(f"  {'Época':<8} {'Melhor':<16} {'Média':<16} {'Desvio':<16} {'Pior':<16}")
        print(f"  {'-'*72}")
        for e in por_desvio_asc[:top_n]:
            print(f"  {e['epoca']:<8} {e['melhor']:<16.2f} {e['media']:<16.2f} "
                  f"{e['desvio']:<16.2f} {e['pior']:<16.2f}")

        # Viabilidade por época
        viab = m.get('viabilidade_por_epoca', [])
        if viab:
            print(f"\n📋 VIABILIDADE POR ÉPOCA:")
            print(f"  {'Época':<8} {'Viáveis':<12} {'Total':<10} {'%':<10}")
            print(f"  {'-'*40}")
            for v in viab:
                print(f"  {v['epoca']:<8} {v['viaveis']:<12} {v['total']:<10} {v['percentual']:<10.1f}")

        print("\n" + "=" * 80 + "\n")

    def exibir_ranking_epocas(self, criterio: str = 'melhor', top_n: int = 10):
        """
        Exibe ranking de épocas por critério.

        Args:
            criterio: 'melhor', 'media', 'desvio', 'pior'
            top_n: Quantas épocas listar
        """
        m = self.calcular()
        detalhes = m.get('epocas_detalhes', [])
        if not detalhes:
            print("⚠️  Sem dados de épocas.")
            return

        ascendente = criterio in ('melhor', 'media', 'desvio')
        ordenado = sorted(detalhes, key=lambda e: e.get(criterio, 0),
                          reverse=not ascendente)

        print(f"\n{'='*70}")
        print(f"RANKING POR ÉPOCA (critério: {criterio}, top {top_n})")
        print(f"{'='*70}")
        print(f"  {'#':<4} {'Época':<8} {'BSF':<16} {'Melhor':<16} {'Média':<16} {'Desvio':<14}")
        print(f"  {'-'*74}")
        for i, e in enumerate(ordenado[:top_n], 1):
            print(f"  {i:<4} {e['epoca']:<8} {e['bsf']:<16.2f} {e['melhor']:<16.2f} "
                  f"{e['media']:<16.2f} {e['desvio']:<14.2f}")
        print(f"{'='*70}\n")
