import wntr
import os
import tempfile
import copy


class Rede:
    """
    Classe para gerenciar redes hidráulicas EPANET.
    
    Permite carregar arquivos .inp existentes ou gerar uma rede de teste aleatória.
    """
    
    def __init__(self, arquivo_inp=None):
        """
        Inicializa a rede hidráulica.
        
        Args:
            arquivo_inp (str, optional): Caminho para o arquivo .inp da rede EPANET.
                                        Se None, gera uma rede de teste aleatória.
                                        Se "hanoiFIM" ou "hanoiFIM.inp", carrega o
                                        arquivo de exemplo empacotado.
        """
        self._arquivo_original = None  # Armazenar o caminho para resets futuros
        self._copia_rede = None  # Cópia da rede para resets rápidos
        
        if arquivo_inp is None:
            print("Nenhum arquivo fornecido. Gerando rede de teste aleatória...")
            self.wn = self._gerar_rede_teste()
            self.nome = "Rede_Teste_Aleatoria"
        else:
            caminho_resolvido = arquivo_inp
            nome_normalizado = str(arquivo_inp).lower().replace('.inp', '')

            # Mapear atalhos conhecidos para arquivos empacotados
            if nome_normalizado == 'hanoifim':
                base_dir = os.path.abspath(os.path.dirname(__file__))
                candidatos = [
                    os.path.join(base_dir, 'redes', 'hanoiFIM.inp'),  # dentro do pacote
                    os.path.join(os.getcwd(), 'hanoiFIM.inp'),         # diretório de trabalho
                ]
                caminho_resolvido = next((c for c in candidatos if os.path.exists(c)), None)

            if not caminho_resolvido or not os.path.exists(caminho_resolvido):
                raise FileNotFoundError(f"Arquivo não encontrado: {arquivo_inp}")
            
            # Armazenar caminho original para resets
            self._arquivo_original = caminho_resolvido
            
            print(f"Carregando rede do arquivo: {caminho_resolvido}")
            self.wn = wntr.network.WaterNetworkModel(caminho_resolvido)
            self.nome = os.path.basename(caminho_resolvido).replace('.inp', '')
        
        # Criar cópia da rede para resets rápidos (sem recarregar do disco)
        self._copia_rede = copy.deepcopy(self.wn)
        
        self.resultados = None
        print(f"Rede '{self.nome}' carregada com sucesso!")
        self._exibir_informacoes()
    
    def _gerar_rede_teste(self):
        """
        Gera uma rede de teste aleatória com configuração básica.
        
        Returns:
            WaterNetworkModel: Rede hidráulica de teste
        """
        wn = wntr.network.WaterNetworkModel()
        
        # Adicionar reservatório
        wn.add_reservoir('Reservatorio1', base_head=100.0)
        
        # Adicionar nós de junção
        wn.add_junction('No1', base_demand=0.01, elevation=50.0)
        wn.add_junction('No2', base_demand=0.015, elevation=45.0)
        wn.add_junction('No3', base_demand=0.012, elevation=40.0)
        
        # Adicionar tubulações
        wn.add_pipe('Tubo1', 'Reservatorio1', 'No1', length=1000.0, 
                    diameter=0.3, roughness=100, minor_loss=0.0)
        wn.add_pipe('Tubo2', 'No1', 'No2', length=800.0, 
                    diameter=0.25, roughness=100, minor_loss=0.0)
        wn.add_pipe('Tubo3', 'No2', 'No3', length=600.0, 
                    diameter=0.2, roughness=100, minor_loss=0.0)
        
        # Configurar opções de simulação
        wn.options.time.duration = 3600  # 1 hora
        wn.options.time.hydraulic_timestep = 3600
        wn.options.time.pattern_timestep = 3600
        
        return wn
    
    def _exibir_informacoes(self):
        """Exibe informações básicas sobre a rede carregada."""
        num_nos = len(self.wn.junction_name_list)
        num_reservatorios = len(self.wn.reservoir_name_list)
        num_tanques = len(self.wn.tank_name_list)
        num_tubos = len(self.wn.pipe_name_list)
        num_bombas = len(self.wn.pump_name_list)
        num_valvulas = len(self.wn.valve_name_list)
        
        print(f"\nInformações da rede:")
        print(f"  - Nós de junção: {num_nos}")
        print(f"  - Reservatórios: {num_reservatorios}")
        print(f"  - Tanques: {num_tanques}")
        print(f"  - Tubulações: {num_tubos}")
        print(f"  - Bombas: {num_bombas}")
        print(f"  - Válvulas: {num_valvulas}")
    
    def simular(self, verbose=None):
        """
        Executa a simulação hidráulica da rede.
        
        Args:
            verbose (bool or str, optional): Controla o nível de saída
                - None: usa o padrão
                - False: nenhuma saída
                - True: apenas informações essenciais
                - 'detalhado': saída completa (padrão original)
        
        Returns:
            dict: Dicionário com resumo dos resultados da simulação
        """
        # Aplicar verbose padrão se não especificado
        if verbose is None:
            verbose = True  # Por padrão, mostra apenas essencial (não detalhado)
        
        if verbose:
            print(f"\nIniciando simulação da rede '{self.nome}'...")
        
        try:
            # Executar simulação
            sim = wntr.sim.EpanetSimulator(self.wn)
            self.resultados = sim.run_sim()
            
            # Processar resultados
            pressoes = self.resultados.node['pressure']
            vazoes = self.resultados.link['flowrate']
            
            # Calcular estatísticas
            resumo = {
                'sucesso': True,
                'pressao_minima': pressoes.min().min(),
                'pressao_maxima': pressoes.max().max(),
                'pressao_media': pressoes.mean().mean(),
                'vazao_minima': vazoes.min().min(),
                'vazao_maxima': vazoes.max().max(),
                'vazao_media': vazoes.mean().mean(),
                'nos_com_pressao_baixa': (pressoes < 20.0).any(axis=0).sum()
            }
            
            # Imprimir detalhes apenas se verbose='detalhado'
            if verbose == 'detalhado':
                print("\n✓ Simulação concluída com sucesso!")
                print(f"\nResumo dos resultados:")
                print(f"  - Pressão mínima: {resumo['pressao_minima']:.2f} m")
                print(f"  - Pressão máxima: {resumo['pressao_maxima']:.2f} m")
                print(f"  - Pressão média: {resumo['pressao_media']:.2f} m")
                print(f"  - Vazão mínima: {resumo['vazao_minima']:.4f} m³/s")
                print(f"  - Vazão máxima: {resumo['vazao_maxima']:.4f} m³/s")
                print(f"  - Nós com pressão < 20m: {resumo['nos_com_pressao_baixa']}")
            
            return resumo
            
        except Exception as e:
            if verbose:
                print(f"\n✗ Erro durante a simulação: {str(e)}")
            return {'sucesso': False, 'erro': str(e)}
    
    def obter_pressoes(self):
        """
        Retorna as pressões de todos os nós da rede.
        
        Returns:
            pandas.DataFrame: DataFrame com as pressões em cada nó ao longo do tempo.
                             Índice: timestamps, Colunas: nomes dos nós
        
        Raises:
            ValueError: Se a simulação ainda não foi executada
        """
        if self.resultados is None:
            raise ValueError("A simulação deve ser executada antes de obter as pressões. Execute rede.simular() primeiro.")
        
        return self.resultados.node['pressure']
    
    def obter_pressao_minima(self, excluir_reservatorios=True, verbose=True):
        """
        Retorna a pressão mínima da rede.
        
        Args:
            excluir_reservatorios (bool): Se True, exclui reservatórios e tanques do cálculo.
                                         Default: True
        
        Returns:
            dict: Dicionário contendo:
                - 'valor': pressão mínima (float)
                - 'no': nome do nó com pressão mínima (str)
                - 'tempo': timestamp quando ocorreu a pressão mínima (str)
        
        Raises:
            ValueError: Se a simulação ainda não foi executada
        """
        if self.resultados is None:
            raise ValueError("A simulação deve ser executada antes de obter as pressões. Execute rede.simular() primeiro.")
        
        # Obter pressões
        pressoes = self.resultados.node['pressure']
        
        # Validar se DataFrame não está vazio
        if pressoes.empty or pressoes.size == 0:
            return {
                'valor': float('inf'),
                'no': 'N/A',
                'tempo': 'N/A'
            }
        
        if excluir_reservatorios:
            # Obter lista de nós de junção (excluindo reservatórios e tanques)
            nos_juncao = self.wn.junction_name_list
            
            # Filtrar apenas nós de junção
            if nos_juncao:
                pressoes = pressoes[nos_juncao]
            
            # Validar novamente após filtro
            if pressoes.empty or pressoes.size == 0:
                return {
                    'valor': float('inf'),
                    'no': 'N/A',
                    'tempo': 'N/A'
                }
        
        # Encontrar o valor mínimo global
        valor_minimo = pressoes.min().min()
        
        # Encontrar em qual nó ocorreu
        no_minimo = pressoes.min().idxmin()
        
        # Encontrar em qual tempo ocorreu
        tempo_minimo = pressoes[no_minimo].idxmin()
        
        resultado = {
            'valor': valor_minimo,
            'no': no_minimo,
            'tempo': str(tempo_minimo)
        }
        
        if verbose:
            print(f"\nPressão mínima da rede:")
            print(f"  - Valor: {resultado['valor']:.2f} m")
            print(f"  - Nó: {resultado['no']}")
            print(f"  - Tempo: {resultado['tempo']}")
        
        return resultado
    
    def salvar(self, caminho_saida=None):
        """
        Salva a rede em um arquivo .inp
        
        Args:
            caminho_saida (str, optional): Caminho para salvar o arquivo. 
                                          Se None, salva como '[nome_rede].inp'
        """
        if caminho_saida is None:
            caminho_saida = f"{self.nome}.inp"
        
        self.wn.write_inpfile(caminho_saida)
        print(f"\nRede salva em: {caminho_saida}")
        return caminho_saida
    
    def obter_nos_e_tubos(self):
        """
        Retorna DataFrames com informações detalhadas dos nós e tubulações.
        
        Returns:
            tuple: (df_nos, df_tubos) - DataFrames com pandas
                  df_nos: ID, Cota, Demanda, Pressão
                  df_tubos: ID, De, Para, Comprimento, Diâmetro, Vazão
        
        Raises:
            ValueError: Se a simulação ainda não foi executada
        """
        import pandas as pd
        
        if self.resultados is None:
            raise ValueError("A simulação deve ser executada antes. Execute rede.simular() primeiro.")
        
        # ==============================================================
        # PARTE A: Nós (Pressão)
        # ==============================================================
        
        dados_nos = []
        
        # Iterar sobre todas as junctions
        for node_name in self.wn.junction_name_list:
            node = self.wn.get_node(node_name)
            
            # Obter a pressão do nó (média ao longo do tempo)
            if node_name in self.resultados.node['pressure'].columns:
                pressao = self.resultados.node['pressure'][node_name].mean()
            else:
                pressao = None
            
            dados_nos.append({
                "ID do Nó": node_name,
                "Cota (m)": round(node.elevation, 2),
                "Demanda (m³/s)": round(node.base_demand, 4),
                "Pressão (mca)": round(pressao, 2) if pressao is not None else None
            })
        
        # Adicionar reservatórios
        for reservoir_name in self.wn.reservoir_name_list:
            reservoir = self.wn.get_node(reservoir_name)
            
            if reservoir_name in self.resultados.node['pressure'].columns:
                pressao = self.resultados.node['pressure'][reservoir_name].mean()
            else:
                pressao = None
            
            # Reservatório pode não ter elevation, usar getattr com padrão 0.0
            cota = getattr(reservoir, 'elevation', 0.0)
            
            dados_nos.append({
                "ID do Nó": reservoir_name,
                "Cota (m)": round(cota, 2),
                "Demanda (m³/s)": 0.0,
                "Pressão (mca)": round(pressao, 2) if pressao is not None else None
            })
        
        df_nos = pd.DataFrame(dados_nos)
        
        # ==============================================================
        # PARTE B: Tubulações (Vazão)
        # ==============================================================
        
        dados_tubos = []
        
        # Iterar sobre todas as tubulações
        for pipe_name in self.wn.pipe_name_list:
            pipe = self.wn.get_link(pipe_name)
            
            # Obter a vazão (média ao longo do tempo)
            if pipe_name in self.resultados.link['flowrate'].columns:
                vazao_m3_s = self.resultados.link['flowrate'][pipe_name].mean()
                vazao_l_s = vazao_m3_s * 1000  # Converter m³/s para L/s
            else:
                vazao_l_s = None
            
            dados_tubos.append({
                "ID Tubo": pipe_name,
                "De (Nó)": pipe.start_node,
                "Para (Nó)": pipe.end_node,
                "Comprimento (m)": round(pipe.length, 2),
                "Diâmetro (mm)": round(pipe.diameter * 1000, 2),
                "Vazão (L/s)": round(vazao_l_s, 2) if vazao_l_s is not None else None
            })
        
        df_tubos = pd.DataFrame(dados_tubos)
        
        return df_nos, df_tubos
    
    def exibir_nos_e_tubos(self):
        """
        Exibe em formato tabular os dados dos nós e tubulações após simulação.
        
        Raises:
            ValueError: Se a simulação ainda não foi executada
        """
        if self.resultados is None:
            raise ValueError("A simulação deve ser executada antes. Execute rede.simular() primeiro.")
        
        df_nos, df_tubos = self.obter_nos_e_tubos()
        
        print("\n" + "="*80)
        print("DADOS DOS NÓS")
        print("="*80)
        print(df_nos.to_string(index=False))
        print(f"\nTotal de nós: {len(df_nos)}")
        
        print("\n" + "-"*80 + "\n")
        
        print("DADOS DAS TUBULAÇÕES")
        print("="*80)
        print(df_tubos.to_string(index=False))
        print(f"\nTotal de tubulações: {len(df_tubos)}")
        
        # Estatísticas
        print("\n" + "="*80)
        print("ESTATÍSTICAS")
        print("="*80)
        
        print("\nNÓS:")
        print(f"  Pressão mínima: {df_nos['Pressão (mca)'].min():.2f} mca")
        print(f"  Pressão máxima: {df_nos['Pressão (mca)'].max():.2f} mca")
        print(f"  Pressão média: {df_nos['Pressão (mca)'].mean():.2f} mca")
        print(f"  Demanda total: {df_nos['Demanda (m³/s)'].sum():.4f} m³/s ({df_nos['Demanda (m³/s)'].sum()*1000:.2f} L/s)")
        
        print("\nTUBULAÇÕES:")
        print(f"  Vazão mínima: {df_tubos['Vazão (L/s)'].min():.2f} L/s")
        print(f"  Vazão máxima: {df_tubos['Vazão (L/s)'].max():.2f} L/s")
        print(f"  Vazão média: {df_tubos['Vazão (L/s)'].mean():.2f} L/s")
        print(f"  Comprimento total: {df_tubos['Comprimento (m)'].sum():.2f} m")
        
        print("\n" + "="*80 + "\n")
    
    def criar_copia_com_diametros(self, diametros_dict, nome_copia=None):
        """
        Cria uma NOVA REDE com os diâmetros personalizados.
        
        Args:
            diametros_dict (dict): Dicionário {nome_tubo: diametro_em_metros}
                                   Exemplo: {'Pipe-1': 0.05, 'Pipe-2': 0.10}
            nome_copia (str, optional): Nome para a nova rede. 
                                        Se None, usa nome_original_otimizada
        
        Returns:
            Rede: Nova instância de Rede com os diâmetros aplicados
        
        Exemplo:
            >>> rede_original = Rede('hanoiFIM')
            >>> dados = otimizador.aplicar_solucao(solucao)
            >>> rede_otimizada = rede_original.criar_copia_com_diametros(
            ...     dados['diametros'], 
            ...     nome_copia='hanoiFIM_otimizada'
            ... )
            >>> rede_otimizada.simular()
        """
        # Criar cópia profunda da rede atual
        rede_nova = Rede.__new__(Rede)
        rede_nova.wn = copy.deepcopy(self.wn)
        rede_nova._arquivo_original = self._arquivo_original
        rede_nova._copia_rede = copy.deepcopy(rede_nova.wn)
        rede_nova.resultados = None
        
        # Definir nome
        if nome_copia is None:
            rede_nova.nome = f"{self.nome}_otimizada"
        else:
            rede_nova.nome = nome_copia
        
        # Aplicar diâmetros
        print(f"\nAplicando diâmetros otimizados à rede '{rede_nova.nome}'...")
        diametros_aplicados = 0
        diametros_nao_encontrados = []
        
        for tubo, novo_diametro in diametros_dict.items():
            try:
                # Validar diâmetro
                if not isinstance(novo_diametro, (int, float)) or novo_diametro <= 0:
                    raise ValueError(f"Diâmetro inválido para {tubo}: {novo_diametro}")
                
                # Obter o pipe (tubo)
                if tubo not in rede_nova.wn.pipe_name_list:
                    diametros_nao_encontrados.append(tubo)
                    continue
                
                pipe = rede_nova.wn.get_link(tubo)
                pipe.diameter = novo_diametro
                diametros_aplicados += 1
                
            except Exception as e:
                print(f"⚠️  Erro ao aplicar diâmetro para {tubo}: {e}")
        
        print(f"✓ {diametros_aplicados} diâmetros aplicados com sucesso")
        
        if diametros_nao_encontrados:
            print(f"⚠️  {len(diametros_nao_encontrados)} tubos não encontrados na rede")
            print(f"   (Talvez nomes diferentes: {diametros_nao_encontrados[:3]}...)")
        
        # Atualizar cópia
        rede_nova._copia_rede = copy.deepcopy(rede_nova.wn)
        
        print(f"✓ Rede '{rede_nova.nome}' criada com sucesso!")
        
        return rede_nova