import copy
import json
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm


class VariadorDeParametros:
    """
    Classe para realizar varredura (sweep) de parâmetros do otimizador.
    
    Permite executar múltiplas otimizações com diferentes combinações de parâmetros,
    salvando e comparando resultados para identificar as melhores configurações.
    
    Exemplo:
        >>> variador = VariadorDeParametros(meu_otimizador)
        >>> variador.definir_parametro('c1', inicial=1.5, final=3.0, passo=0.5)
        >>> variador.definir_parametro('c2', inicial=1.5, final=3.0, passo=0.5)
        >>> variador.definir_condicoes_iniciais(populacao_inicial=minha_populacao)
        >>> resultados = variador.executar(metodo='PSO', diretorio_saida='resultados_pso')
        >>> df_resumo = variador.obter_resumo()
    """
    
    def __init__(self, otimizador, verbose=True):
        """
        Inicializa o variador de parâmetros.
        
        Args:
            otimizador (Otimizador): Instância do otimizador a ser variado
            verbose (bool): Se True, exibe informações durante execução
        """
        from .otimizador import Otimizador
        
        if not isinstance(otimizador, Otimizador):
            raise TypeError("O parâmetro 'otimizador' deve ser uma instância da classe Otimizador.")
        
        self.otimizador = otimizador
        self.verbose = verbose
        
        # Armazenar ranges de parâmetros: {nome_parametro: {'inicial': x, 'final': y, 'passo': z}}
        self.ranges_parametros = {}
        
        # Condições iniciais
        self.populacao_inicial = None
        self.solucao_inicial = None
        self.verbose_otimizacao = False
        
        # Resultados
        self.resultados = []
        self.dataframe_resultados = None
        self._timestamp_execucao = None
        
        if self.verbose:
            print("✓ VariadorDeParametros inicializado")
    
    def definir_parametro(self, nome_parametro, inicial, final, passo):
        """
        Define um parâmetro para varredura.
        
        Args:
            nome_parametro (str): Nome do parâmetro (ex: 'c1', 'c2', 'w', 'b', 'pa')
            inicial (float): Valor inicial do intervalo
            final (float): Valor final do intervalo (inclusive)
            passo (float): Incremento a cada iteração
        
        Raises:
            ValueError: Se passo <= 0 ou inicial > final
        """
        if passo <= 0:
            raise ValueError(f"Passo deve ser positivo, recebido: {passo}")
        if inicial > final:
            raise ValueError(f"Inicial ({inicial}) não pode ser maior que final ({final})")
        
        self.ranges_parametros[nome_parametro] = {
            'inicial': inicial,
            'final': final,
            'passo': passo
        }
        
        # Calcular quantos valores
        num_valores = int(np.ceil((final - inicial) / passo)) + 1
        
        if self.verbose:
            print(f"✓ Parâmetro '{nome_parametro}' definido: [{inicial}, {final}] com passo {passo} ({num_valores} valores)")
    
    def definir_condicoes_iniciais(self, populacao_inicial=None, solucao_inicial=None, verbose_otimizacao=False):
        """
        Define as condições iniciais para as otimizações.
        
        Args:
            populacao_inicial (list, optional): População inicial (lista de soluções)
            solucao_inicial (list, optional): Uma solução inicial a ser usada
            verbose_otimizacao (bool): Se True, exibe output da otimização a cada run
        """
        self.populacao_inicial = populacao_inicial
        self.solucao_inicial = solucao_inicial
        self.verbose_otimizacao = verbose_otimizacao
        
        if self.verbose:
            if populacao_inicial is not None:
                print(f"✓ População inicial definida ({len(populacao_inicial)} indivíduos)")
            if solucao_inicial is not None:
                print(f"✓ Solução inicial definida")
    
    def _gerar_combinacoes(self):
        """
        Gera todas as combinações de parâmetros definidos.
        
        Returns:
            list: Lista de dicionários {parametro: valor} para cada combinação
        """
        if not self.ranges_parametros:
            raise ValueError("Nenhum parâmetro foi definido. Use definir_parametro() primeiro.")
        
        # Gerar valores para cada parâmetro
        valores_por_parametro = {}
        for nome, config in self.ranges_parametros.items():
            valores = np.arange(
                config['inicial'],
                config['final'] + config['passo'] / 2,  # +passo/2 para incluir 'final'
                config['passo']
            )
            # Limitar precisão para evitar erros de ponto flutuante
            valores = np.round(valores, 10)
            valores_por_parametro[nome] = valores
        
        # Gerar produto cartesiano (todas as combinações)
        import itertools
        nomes_param = list(valores_por_parametro.keys())
        valores_lista = [valores_por_parametro[nome] for nome in nomes_param]
        
        combinacoes = []
        for valores_combo in itertools.product(*valores_lista):
            combo_dict = {nome: valor for nome, valor in zip(nomes_param, valores_combo)}
            combinacoes.append(combo_dict)
        
        return combinacoes
    
    def executar(self, metodo='PSO', diretorio_saida=None, salvar_json=True):
        """
        Executa a varredura de parâmetros.
        
        Args:
            metodo (str): Algoritmo de otimização (PSO, GWO, etc)
            diretorio_saida (str, optional): Diretório para salvar resultados
            salvar_json (bool): Se True, salva resultados em JSON
        
        Returns:
            pd.DataFrame: DataFrame com resumo de todos os resultados
        """
        combinacoes = self._gerar_combinacoes()
        num_combos = len(combinacoes)
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"INICIANDO VARREDURA DE PARÂMETROS")
            print(f"{'='*70}")
            print(f"Algoritmo: {metodo}")
            print(f"Combinações a executar: {num_combos}")
            print(f"Parâmetros variados: {list(self.ranges_parametros.keys())}")
            print(f"{'='*70}\n")
        
        self.resultados = []
        self._timestamp_execucao = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Criar diretório se especificado
        if diretorio_saida:
            Path(diretorio_saida).mkdir(parents=True, exist_ok=True)
        
        # Executar cada combinação
        with tqdm(total=num_combos, desc="Varredura de parâmetros", 
                  disable=not self.verbose, ncols=80) as pbar:
            for i, combo in enumerate(combinacoes):
                # Configurar parâmetros do otimizador
                self.otimizador.definir_parametros(metodo, **combo)
                
                # Executar otimização
                try:
                    resultado_opt = self.otimizador.otimizar(
                        metodo=metodo,
                        verbose=self.verbose_otimizacao,
                        solucao_inicial=self.populacao_inicial or self.solucao_inicial
                    )
                    
                    # Aplicar solução para obter dados adicionais
                    dados_solucao = self.otimizador.aplicar_solucao(
                        resultado_opt['melhor_solucao'],
                        simular=True
                    )
                    
                    # Montar resultado completo
                    resultado_completo = {
                        'combinacao_id': i,
                        'parametros': combo.copy(),
                        'melhor_custo_fitness': float(resultado_opt['melhor_custo']),
                        'custo_real': float(dados_solucao['custo_total']),
                        'pressao_minima': float(dados_solucao.get('pressao_minima', np.nan)),
                        'no_pressao_minima': str(dados_solucao.get('no_pressao_minima', 'N/A')),
                        'sucesso': True,
                        'melhor_solucao': resultado_opt['melhor_solucao'].tolist() if isinstance(resultado_opt['melhor_solucao'], np.ndarray) else resultado_opt['melhor_solucao']
                    }
                    
                except Exception as e:
                    # Registrar falha
                    resultado_completo = {
                        'combinacao_id': i,
                        'parametros': combo.copy(),
                        'melhor_custo_fitness': np.nan,
                        'custo_real': np.nan,
                        'pressao_minima': np.nan,
                        'no_pressao_minima': 'ERRO',
                        'sucesso': False,
                        'erro': str(e),
                        'melhor_solucao': None
                    }
                
                self.resultados.append(resultado_completo)
                pbar.update(1)
        
        # Criar DataFrame
        self._processar_resultados()
        
        # Salvar em JSON se solicitado
        if salvar_json and diretorio_saida:
            self._salvar_resultados_json(diretorio_saida)
        
        # Salvar DataFrame em CSV
        if diretorio_saida:
            caminho_csv = os.path.join(diretorio_saida, f"resumo_{self._timestamp_execucao}.csv")
            self.dataframe_resultados.to_csv(caminho_csv, index=False)
            if self.verbose:
                print(f"\n✓ Resultados salvos em: {caminho_csv}")
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"✓ Varredura concluída com sucesso!")
            print(f"{'='*70}\n")
        
        return self.dataframe_resultados
    
    def _processar_resultados(self):
        """
        Processa resultados e cria DataFrame com colunas expandidas.
        """
        # Extrair parâmetros para colunas separadas
        linhas = []
        for res in self.resultados:
            linha = {
                'combinacao_id': res['combinacao_id'],
                **res['parametros'],  # Expande parâmetros como colunas
                'fitness': res['melhor_custo_fitness'],
                'custo_real_R$': res['custo_real'],
                'pressao_minima_m': res['pressao_minima'],
                'no_pressao_minima': res['no_pressao_minima'],
                'sucesso': res['sucesso']
            }
            linhas.append(linha)
        
        self.dataframe_resultados = pd.DataFrame(linhas)
        
        # Ordenar por melhor custo real
        if 'custo_real_R$' in self.dataframe_resultados.columns:
            self.dataframe_resultados = self.dataframe_resultados.sort_values(
                'custo_real_R$',
                ascending=True,
                na_position='last'
            ).reset_index(drop=True)
    
    def _salvar_resultados_json(self, diretorio_saida):
        """
        Salva todos os resultados em arquivo JSON (inclui soluções).
        """
        caminho_json = os.path.join(diretorio_saida, f"resultados_completos_{self._timestamp_execucao}.json")
        
        # Preparar dados para JSON (converter numpy arrays)
        resultados_json = []
        for res in self.resultados:
            res_copia = res.copy()
            res_copia['melhor_solucao'] = res_copia['melhor_solucao']  # Já é lista
            if not res_copia['sucesso']:
                res_copia.pop('erro', None)  # Remover mensagens de erro muito longas
            resultados_json.append(res_copia)
        
        with open(caminho_json, 'w') as f:
            json.dump(resultados_json, f, indent=2)
        
        if self.verbose:
            print(f"✓ Resultados completos salvos em: {caminho_json}")
    
    def obter_resumo(self):
        """
        Retorna DataFrame com resumo dos resultados.
        
        Returns:
            pd.DataFrame: DataFrame com todos os resultados
        """
        if self.dataframe_resultados is None:
            raise ValueError("Nenhuma varredura foi executada ainda. Use executar() primeiro.")
        return self.dataframe_resultados.copy()
    
    def obter_melhor_resultado(self):
        """
        Retorna a melhor configuração encontrada (menor custo real).
        
        Returns:
            dict: {'parametros': dict, 'custo': float, 'fitness': float, 'solucao': list}
        """
        if not self.resultados:
            raise ValueError("Nenhuma varredura foi executada ainda.")
        
        # Filtrar apenas os bem-sucedidos
        sucessos = [r for r in self.resultados if r['sucesso']]
        if not sucessos:
            raise ValueError("Nenhuma otimização bem-sucedida foi encontrada.")
        
        # Encontrar o de menor custo real
        melhor = min(sucessos, key=lambda r: r['custo_real'])
        
        return {
            'parametros': melhor['parametros'],
            'custo_real': melhor['custo_real'],
            'fitness': melhor['melhor_custo_fitness'],
            'pressao_minima': melhor['pressao_minima'],
            'solucao': melhor['melhor_solucao']
        }
    
    def exibir_resumo(self, top_n=10):
        """
        Exibe um resumo formatado dos melhores resultados.
        
        Args:
            top_n (int): Número de top resultados a exibir
        """
        if self.dataframe_resultados is None:
            raise ValueError("Nenhuma varredura foi executada ainda.")
        
        df = self.dataframe_resultados.head(top_n)
        
        print("\n" + "="*90)
        print(f"TOP {min(top_n, len(df))} MELHORES RESULTADOS (Ordenado por Custo Real)")
        print("="*90)
        
        # Selecionar colunas para exibição
        colunas_exibir = ['combinacao_id', 'custo_real_R$', 'fitness', 'pressao_minima_m', 'sucesso']
        colunas_parametros = list(self.ranges_parametros.keys())
        
        # Montar DataFrame para exibição
        df_exibir = df[colunas_parametros + colunas_exibir].copy()
        
        print(df_exibir.to_string(index=False))
        print("="*90 + "\n")
    
    def comparar_parametros(self, nome_parametro1, nome_parametro2=None):
        """
        Gera análise comparativa entre dois parâmetros.
        
        Args:
            nome_parametro1 (str): Primeiro parâmetro para análise
            nome_parametro2 (str, optional): Segundo parâmetro. Se None, análise univariada
        
        Returns:
            pd.DataFrame ou pd.pivot_table: Tabela de comparação
        """
        if self.dataframe_resultados is None:
            raise ValueError("Nenhuma varredura foi executada ainda.")
        
        if nome_parametro2 is None:
            # Análise univariada
            df_analise = self.dataframe_resultados.groupby(nome_parametro1).agg({
                'custo_real_R$': ['min', 'mean', 'max'],
                'fitness': ['min', 'mean', 'max'],
                'pressao_minima_m': ['min', 'mean', 'max'],
                'combinacao_id': 'count'
            }).round(4)
            df_analise.columns = ['_'.join(col).strip() for col in df_analise.columns.values]
            return df_analise
        else:
            # Análise bivariada (pivot table)
            pivot = self.dataframe_resultados.pivot_table(
                values='custo_real_R$',
                index=nome_parametro1,
                columns=nome_parametro2,
                aggfunc='min'
            )
            return pivot
    
    def exibir_comparacao(self, nome_parametro1, nome_parametro2=None):
        """
        Exibe análise comparativa de forma formatada.
        
        Args:
            nome_parametro1 (str): Primeiro parâmetro para análise
            nome_parametro2 (str, optional): Segundo parâmetro. Se None, análise univariada
        """
        comparacao = self.comparar_parametros(nome_parametro1, nome_parametro2)
        
        print("\n" + "="*90)
        if nome_parametro2 is None:
            print(f"ANÁLISE UNIVARIADA: {nome_parametro1}")
        else:
            print(f"ANÁLISE BIVARIADA: {nome_parametro1} vs {nome_parametro2}")
        print("="*90)
        print(comparacao.to_string())
        print("="*90 + "\n")
    
    def obter_informacoes(self):
        """
        Retorna informações sobre a varredura realizada.
        
        Returns:
            dict: Informações da varredura
        """
        if not self.resultados:
            return {'status': 'Nenhuma varredura executada ainda'}
        
        sucessos = sum(1 for r in self.resultados if r['sucesso'])
        falhas = len(self.resultados) - sucessos
        
        custos_validos = [r['custo_real'] for r in self.resultados if r['sucesso']]
        
        return {
            'total_combinacoes': len(self.resultados),
            'sucessos': sucessos,
            'falhas': falhas,
            'parametros_variados': list(self.ranges_parametros.keys()),
            'melhor_custo': min(custos_validos) if custos_validos else None,
            'pior_custo': max(custos_validos) if custos_validos else None,
            'custo_medio': np.mean(custos_validos) if custos_validos else None,
            'timestamp': self._timestamp_execucao
        }
