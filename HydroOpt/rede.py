import wntr
import os
import tempfile


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
        """
        if arquivo_inp is None:
            print("Nenhum arquivo fornecido. Gerando rede de teste aleatória...")
            self.wn = self._gerar_rede_teste()
            self.nome = "Rede_Teste_Aleatoria"
        else:
            if not os.path.exists(arquivo_inp):
                raise FileNotFoundError(f"Arquivo não encontrado: {arquivo_inp}")
            
            print(f"Carregando rede do arquivo: {arquivo_inp}")
            self.wn = wntr.network.WaterNetworkModel(arquivo_inp)
            self.nome = os.path.basename(arquivo_inp).replace('.inp', '')
        
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
    
    def simular(self):
        """
        Executa a simulação hidráulica da rede.
        
        Returns:
            dict: Dicionário com resumo dos resultados da simulação
        """
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
    
    def obter_pressao_minima(self, excluir_reservatorios=True):
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
        
        if excluir_reservatorios:
            # Obter lista de nós de junção (excluindo reservatórios e tanques)
            nos_juncao = self.wn.junction_name_list
            
            # Filtrar apenas nós de junção
            pressoes = pressoes[nos_juncao]
        
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