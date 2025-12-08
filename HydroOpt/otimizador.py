import copy
import logging
from tqdm import tqdm
from mealpy.utils.space import FloatVar
from mealpy.utils.problem import Problem

# Suprimir warnings do WNTR durante otimização
logging.getLogger("wntr").setLevel(logging.ERROR)
logging.getLogger("wntr.epanet.io").setLevel(logging.ERROR)
logging.getLogger("wntr.epanet.toolkit").setLevel(logging.ERROR)


class Otimizador:
    """
    Classe para otimização de redes hidráulicas usando algoritmos de enxame.
    
    Detecta disponibilidade de GPU e permite ativá-la ou desativá-la manualmente.
    """
    
    def __init__(self, rede, usar_gpu=None, verbose=True, pressao_min_desejada=10.0, epoch=50, pop_size=30, diametros=None, usar_paralelismo=True, n_workers=None):
        """
        Inicializa o Otimizador com uma rede hidráulica.
        
        Args:
            rede (Rede): Instância da classe Rede a ser otimizada
            usar_gpu (bool, optional): Se True força uso de GPU, False força CPU, None detecta automaticamente
            verbose (bool): Se True, exibe informações sobre configuração
        """
        from .rede import Rede
        
        # Validar rede
        if not isinstance(rede, Rede):
            raise TypeError("O parâmetro 'rede' deve ser uma instância da classe Rede.")
        
        self.rede = rede
        self.verbose = verbose
        self.pressao_min_desejada = pressao_min_desejada
        self.epoch = epoch
        self.pop_size = pop_size
        self.diametros = diametros
        self.usar_paralelismo = usar_paralelismo
        self.n_workers = n_workers
        self._parametros_padrao = self._criar_parametros_padrao()
        self.parametros = copy.deepcopy(self._parametros_padrao)
        
        # Detectar GPU disponível
        self.gpu_disponivel = self._detectar_gpu()
        
        # Definir modo de uso
        if usar_gpu is None:
            # Usar GPU se disponível
            self.usar_gpu = self.gpu_disponivel
        else:
            # Forçar modo especificado
            if usar_gpu and not self.gpu_disponivel:
                if self.verbose:
                    print("⚠️  GPU solicitada mas não disponível. Usando CPU.")
                self.usar_gpu = False
            else:
                self.usar_gpu = usar_gpu
        
        if self.verbose:
            self._exibir_configuracao()

    def _criar_parametros_padrao(self):
        """
        Define os parâmetros padrão para cada algoritmo suportado.

        Retorna:
            dict: Dicionário {metodo: {parametros}}
        """
        return {
            # Big 4
            'PSO': {'c1': 2.05, 'c2': 2.05, 'w': 0.4},
            'GWO': {},  # Parameter-free
            'WOA': {'b': 1.0},
            'ABC': {'limit': 100},

            # Pássaros e Insetos
            'CS': {'pa': 0.25},
            'BA': {'loudness': 1.0, 'pulse_rate': 0.5},
            'FA': {'alpha': 0.5, 'beta': 0.2, 'gamma': 1.0},
            'HHO': {},  # Parameter-free

            # Evolutivos
            'DE': {'wf': 0.8, 'cr': 0.9},
            'GA': {'pc': 0.9, 'pm': 0.01},
        }
    
    def _detectar_gpu(self):
        """
        Detecta a disponibilidade de GPU no sistema.
        
        Returns:
            bool: True se GPU está disponível, False caso contrário
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            pass
        
        try:
            import cupy as cp
            cp.cuda.Device()
            return True
        except (ImportError, RuntimeError):
            pass
        
        return False
    
    def _exibir_configuracao(self):
        """Exibe informações sobre a configuração do otimizador."""
        print("\n" + "="*60)
        print("CONFIGURAÇÃO DO OTIMIZADOR")
        print("="*60)
        print(f"\nRede: {self.rede.nome}")
        print(f"GPU Disponível: {'Sim ✓' if self.gpu_disponivel else 'Não ✗'}")
        print(f"GPU em Uso: {'Sim ✓' if self.usar_gpu else 'Não (CPU)'}")
        print(f"Pressão mínima desejada: {self.pressao_min_desejada} m")
        print(f"Épocas: {self.epoch} | População: {self.pop_size}")
        print("\n" + "="*60 + "\n")
    
    def obter_status_gpu(self):
        """
        Retorna informações sobre o status da GPU.
        
        Returns:
            dict: Dicionário com status {'disponivel': bool, 'em_uso': bool}
        """
        return {
            'disponivel': self.gpu_disponivel,
            'em_uso': self.usar_gpu
        }

    # ------------------------------------------------------------------
    # Avaliação de solução / objetivo
    # ------------------------------------------------------------------
    def _penalidade_base(self):
        """Retorna penalidade base muito alta para forçar respeito às restrições."""
        if self.diametros is not None:
            try:
                # Multiplica por 1e6 para ter penalidade de bilhões quando violado
                return self.diametros.obter_penalidade() * 1e6
            except Exception:
                return 1e9
        return 1e9

    def _resetar_rede(self):
        """
        Reseta a rede para os diâmetros padrão originais.
        Necessário a cada iteração para começar com estado limpo.
        
        Usa uma cópia em memória da rede (rápido) em vez de recarregar do disco.
        """
        import copy
        
        # Se a rede tem uma cópia em memória, usar dela (muito mais rápido)
        if hasattr(self.rede, '_copia_rede') and self.rede._copia_rede is not None:
            self.rede.wn = copy.deepcopy(self.rede._copia_rede)
        # Fallback: recarregar do arquivo original (mais lento)
        elif hasattr(self.rede, '_arquivo_original') and self.rede._arquivo_original:
            import wntr
            self.rede.wn = wntr.network.WaterNetworkModel(self.rede._arquivo_original)
    
    def _atualizar_diametros_rede(self, solution):
        """
        Atualiza os diâmetros da rede baseado na solução (valores [0,1]).
        
        Args:
            solution (list): Lista de valores [0,1] para mapear aos diâmetros disponíveis.
        
        Returns:
            float: Custo total dos diâmetros aplicados
        """
        custo_diametros = 0.0
        
        if solution is None or self.diametros is None:
            return custo_diametros
        
        diametros_disponiveis = self.diametros.obter_diametros()
        
        # Mapear valores [0,1] para índices de diâmetros
        for i, pipe_name in enumerate(self.rede.wn.pipe_name_list):
            if i < len(solution):
                # Converter valor [0,1] para índice de diâmetro
                idx = int(solution[i] * (len(diametros_disponiveis) - 1))
                idx = min(max(0, idx), len(diametros_disponiveis) - 1)
                
                diametro_escolhido = diametros_disponiveis[idx]
                pipe = self.rede.wn.get_link(pipe_name)
                
                # Verificação de segurança
                if type(pipe).__name__ != 'Pipe':
                    continue
                
                pipe.diameter = diametro_escolhido
                custo_diametros += self.diametros.obter_valor(diametro_escolhido) * pipe.length
        
        return custo_diametros
    
    def _calcular_erro_quadrado(self, pressoes_reais):
        """
        Calcula erro quadrado médio da pressão em relação à pressão desejada.
        Quanto mais longe da pressão desejada, maior o erro.
        
        Args:
            pressoes_reais (pandas.Series): Pressões dos nós de junção
        
        Returns:
            float: Erro quadrado médio normalizado
        """
        import numpy as np
        
        if pressoes_reais is None or len(pressoes_reais) == 0:
            return float('inf')
        
        # Calcular erro quadrado para cada nó
        erros_quadrados = (pressoes_reais - self.pressao_min_desejada) ** 2
        
        # Retornar média dos erros quadrados
        return np.mean(erros_quadrados)
    
    def _avaliar_rede(self, solution=None, verbose=False):
        """
        Simula a rede e calcula custo com penalidade.
        Usa uma mistura de: custo dos diâmetros + erro quadrado + penalidade de pressão.
        
        Args:
            solution (list): Lista de valores [0,1] para mapear aos diâmetros disponíveis.
        
        Returns:
            float: Custo total (custo dos diâmetros + erro quadrado + penalidade de pressão)
        """
        penalidade_base = self._penalidade_base()
        
        # IMPORTANTE: Resetar a rede a cada iteração para garantir estado limpo
        self._resetar_rede()
        
        # Aplicar diâmetros da solução aos tubos
        custo_diametros = self._atualizar_diametros_rede(solution)
        
        # Simular rede com novos diâmetros (sem prints durante otimização)
        resultado = self.rede.simular(verbose=False)

        if not resultado.get('sucesso', False):
            return penalidade_base + custo_diametros

        # Obter pressões reais
        pressao_info = self.rede.obter_pressao_minima(excluir_reservatorios=True, verbose=verbose)
        pressao_min = pressao_info['valor']
        
        # Se pressão é inválida (inf ou nan), retornar penalidade máxima
        if pressao_min == float('inf') or pressao_min != pressao_min:  # NaN check
            return penalidade_base + custo_diametros

        # Calcular erro quadrado das pressões
        pressoes_node = self.rede.obter_pressoes()
        if pressoes_node is not None and not pressoes_node.empty:
            nos_juncao = self.rede.wn.junction_name_list
            pressoes_juncao = pressoes_node[nos_juncao].iloc[0]  # Primeira linha (tempo 0)
            erro_quadrado = self._calcular_erro_quadrado(pressoes_juncao)
        else:
            erro_quadrado = 0.0

        # Penalidade se pressão mínima não atende ao requisito
        penalidade_pressao = 0.0
        if pressao_min < self.pressao_min_desejada:
            deficiencia = self.pressao_min_desejada - pressao_min
            
            # Mistura: Penalidade Fixa (punição) + Linear (direção) + Quadrática (severidade)
            # Isso cria uma "rampa" suave para o lobo subir em direção à solução viável
            penalidade_pressao = 1e5 + (1e6 * deficiencia) + (1e7 * (deficiencia ** 2))

        # Função objetivo: mistura de custo (60%) + erro quadrado (40%)
        # Ambos penalizados se pressão for insuficiente
        peso_custo = 0.6
        peso_erro = 0.4
        
        custo_final = (peso_custo * custo_diametros + 
                      peso_erro * erro_quadrado + 
                      penalidade_pressao)

        return custo_final

    # ------------------------------------------------------------------
    # Gerenciamento de parâmetros de algoritmos (MealPy)
    # ------------------------------------------------------------------
    def listar_metodos(self):
        """Lista os métodos de otimização suportados."""
        return sorted(self.parametros.keys())

    def obter_parametros(self, metodo):
        """
        Retorna os parâmetros atuais de um método.

        Args:
            metodo (str): Nome do método (ex.: 'PSO', 'GWO')

        Returns:
            dict: Parâmetros configurados para o método
        """
        metodo = metodo.upper()
        if metodo not in self.parametros:
            raise KeyError(f"Método '{metodo}' não suportado. Disponíveis: {self.listar_metodos()}")
        return copy.deepcopy(self.parametros[metodo])

    def definir_parametros(self, metodo, **novos_parametros):
        """
        Atualiza/define parâmetros de um método específico.

        Args:
            metodo (str): Nome do método
            **novos_parametros: Parâmetros a serem atualizados
        """
        metodo = metodo.upper()
        if metodo not in self.parametros:
            raise KeyError(f"Método '{metodo}' não suportado. Disponíveis: {self.listar_metodos()}")

        # Atualiza mantendo parâmetros existentes
        self.parametros[metodo].update(novos_parametros)

        if self.verbose:
            print(f"✓ Parâmetros do método {metodo} atualizados: {self.parametros[metodo]}")

    def resetar_parametros(self, metodo=None):
        """
        Restaura parâmetros padrão.

        Args:
            metodo (str, optional): Se None, reseta todos. Caso contrário, reseta apenas o método indicado.
        """
        if metodo is None:
            self.parametros = copy.deepcopy(self._parametros_padrao)
            if self.verbose:
                print("✓ Todos os parâmetros foram restaurados para os padrões.")
            return

        metodo = metodo.upper()
        if metodo not in self.parametros:
            raise KeyError(f"Método '{metodo}' não suportado. Disponíveis: {self.listar_metodos()}")

        self.parametros[metodo] = copy.deepcopy(self._parametros_padrao[metodo])
        if self.verbose:
            print(f"✓ Parâmetros do método {metodo} restaurados para os padrões: {self.parametros[metodo]}")
    
    def ativar_gpu(self):
        """
        Ativa o uso de GPU se estiver disponível.
        
        Returns:
            bool: True se GPU foi ativada, False se não disponível
        """
        if self.gpu_disponivel:
            self.usar_gpu = True
            if self.verbose:
                print("✓ GPU ativada com sucesso!")
            return True
        else:
            if self.verbose:
                print("⚠️  GPU não está disponível no sistema.")
            return False
    
    def desativar_gpu(self):
        """
        Desativa o uso de GPU (força execução em CPU).
        """
        self.usar_gpu = False
        if self.verbose:
            print("✓ GPU desativada. Usando CPU para cálculos.")
    
    def alternar_gpu(self):
        """
        Alterna entre usar GPU e CPU.
        
        Returns:
            bool: Estado atual (True = usando GPU, False = usando CPU)
        """
        if self.gpu_disponivel:
            self.usar_gpu = not self.usar_gpu
            status = "ativada" if self.usar_gpu else "desativada"
            if self.verbose:
                print(f"✓ GPU {status}.")
            return self.usar_gpu
        else:
            if self.verbose:
                print("⚠️  GPU não está disponível. Continuando com CPU.")
            return False
    
    def obter_rede(self):
        """
        Retorna a rede associada ao otimizador.
        
        Returns:
            Rede: Instância da rede
        """
        return self.rede
    
    def simular_rede(self):
        """
        Executa uma simulação da rede associada.
        
        Returns:
            dict: Resultado da simulação
        """
        if self.verbose:
            modo = "GPU" if self.usar_gpu else "CPU"
            print(f"\nExecutando simulação em {modo}...")
        
        return self.rede.simular()
    
    def obter_informacoes(self):
        """
        Retorna informações detalhadas do otimizador.
        
        Returns:
            dict: Dicionário com informações
        """
        return {
            'rede': self.rede.nome,
            'gpu_disponivel': self.gpu_disponivel,
            'gpu_em_uso': self.usar_gpu,
            'modo': 'GPU' if self.usar_gpu else 'CPU',
            'pressao_min_desejada': self.pressao_min_desejada,
            'epoch': self.epoch,
            'pop_size': self.pop_size,
            'usar_paralelismo': self.usar_paralelismo,
            'n_workers': self.n_workers or 'auto'
        }
    
    def exibir_configuracao(self):
        """
        Exibe as configurações atuais do otimizador de forma formatada.
        Função pública para visualizar os parâmetros.
        """
        info = self.obter_informacoes()
        
        print("\n" + "="*70)
        print("CONFIGURAÇÃO ATUAL DO OTIMIZADOR")
        print("="*70)
        print(f"\n📊 Rede Hidráulica:")
        print(f"    Nome: {info['rede']}")
        print(f"    Tubos: {len(self.rede.wn.pipe_name_list)}")
        print(f"    Nós de junção: {len(self.rede.wn.junction_name_list)}")
        
        print(f"\n⚙️  Parâmetros de Otimização:")
        print(f"    Pressão mínima desejada: {info['pressao_min_desejada']} m")
        print(f"    Épocas: {info['epoch']}")
        print(f"    População: {info['pop_size']}")
        print(f"    Total de avaliações: {info['epoch'] * info['pop_size']}")
        
        print(f"\n💻 Computação:")
        print(f"    GPU disponível: {'Sim ✓' if info['gpu_disponivel'] else 'Não ✗'}")
        print(f"    GPU em uso: {'Sim ✓' if info['gpu_em_uso'] else 'Não (CPU)'}")
        print(f"    Modo: {info['modo']}")
        print(f"    Paralelismo: {'Ativado' if info['usar_paralelismo'] else 'Desativado'}")
        print(f"    Workers: {info['n_workers']}")
        
        print(f"\n📋 Algoritmos Disponíveis:")
        metodos = self.listar_metodos()
        print(f"    Quantidade: {len(metodos)}")
        print(f"    Métodos: {', '.join(metodos)}")
        
        if self.diametros is not None:
            print(f"\n📏 Diâmetros Configurados:")
            diams = self.diametros.obter_diametros()
            print(f"    Quantidade: {len(diams)} diâmetros")
            print(f"    Intervalo: {diams[0]:.4f}m a {diams[-1]:.4f}m")
            print(f"    Penalidade base: {self._penalidade_base():.2e}")
        else:
            print(f"\n⚠️  Diâmetros: Nenhum configurado")
        
        print("\n" + "="*70 + "\n")

    # ------------------------------------------------------------------
    # Execução de otimização (MealPy)
    # ------------------------------------------------------------------
    def otimizar(self, metodo='PSO', verbose=False):
        """
        Executa otimização usando MealPy com penalização de pressão mínima.

        Args:
            metodo (str): Algoritmo a usar (PSO, GWO, WOA, ABC, CS, BA, FA, HHO, DE, GA)

        Returns:
            dict: {'melhor_custo': float, 'melhor_solucao': list, 'historico': list}
        """
        metodo = metodo.upper()
        if metodo not in self.parametros:
            raise KeyError(f"Método '{metodo}' não suportado. Disponíveis: {self.listar_metodos()}")

        # Tentar importar mealpy
        try:
            from mealpy import swarm_based, evolutionary_based
        except ImportError:
            raise ImportError("MealPy não está instalado. Adicione 'mealpy' às dependências.")

        # Criar classe derivada de Problem para MealPy 3.0+
        optimizer_instance = self
        n_tubos = len(self.rede.wn.pipe_name_list)

        # Estimar total de avaliações (épocas * população)
        total_evals = max(1, int(self.epoch) * int(self.pop_size))

        class HydroNetworkProblem(Problem):
            """Problema de otimização de rede hidráulica para MealPy 3.0+"""
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
            
            def obj_func(self, solution):
                """Função objetivo que simula a rede hidráulica

                Atualiza a barra de progresso por avaliação (1 avaliação = 1 chamada).
                """
                value = optimizer_instance._avaliar_rede(solution, verbose=verbose)

                # Atualizar barra de progresso se estiver definida
                try:
                    pbar = getattr(optimizer_instance, '_pbar', None)
                    if pbar is not None:
                        pbar.update(1)
                except Exception:
                    # Não queremos interromper a avaliação por um erro de UI
                    pass

                return [value]

        # Problema para MealPy 3.0+: Uma variável [0,1] para cada tubo
        problem = HydroNetworkProblem(
            bounds=[FloatVar(lb=0, ub=1) for _ in range(n_tubos)],
            minmax='min',
            log_to=None,
        )

        modelo = self._instanciar_modelo(metodo, swarm_based, evolutionary_based)

        workers = self._definir_workers()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"INICIANDO OTIMIZAÇÃO: {metodo}")
            print(f"{'='*60}")
            print(f"Épocas: {self.epoch} | População: {self.pop_size} | Workers: {workers}")
            print(f"{'='*60}\n")

        # Criar barra de progresso com tqdm (conta avaliações: épocas * população)
        with tqdm(total=total_evals, desc=f"Otimizando com {metodo}", 
                  unit="avaliação", disable=not self.verbose, ncols=80) as pbar:
            # Expor a barra para o obj_func via instância do otimizador
            optimizer_instance._pbar = pbar

            # Rodar otimização (MealPy 3.0+)
            # Usar 'single' para evitar problemas de memória com WNTR em multithread/multiprocess
            agent = modelo.solve(
                problem,
                mode='single',
                n_workers=1,
            )
            
            # Extrair resultados do agent retornado
            melhor_solucao = agent.solution
            melhor_custo = agent.target.objectives[0]
            
            # Remover referência à barra
            optimizer_instance._pbar = None

        self._resetar_rede()
        # Calculamos o custo financeiro puro (sem penalidades de pressão)
        custo_real_investimento = self._atualizar_diametros_rede(melhor_solucao)
        # -----------------------------------------------------

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"✓ Otimização concluída com {metodo}")
            print(f"{'='*60}")
            # Exibe o Fitness (Score matemático)
            print(f"  🔹 Melhor Fitness (Score):   {melhor_custo:.6f}")
            # Exibe o Dinheiro (O que importa para o engenheiro)
            print(f"  💰 Custo Real Estimado:      R$ {custo_real_investimento:,.2f}")
            print(f"{'='*60}\n")

        return {
            'melhor_custo': melhor_custo,
            'melhor_solucao': melhor_solucao,
            'historico': [melhor_custo],  # MealPy 3.0 não retorna histórico
        }

    def aplicar_solucao(self, solucao, simular=True):
        """
        Aplica uma solução otimizada à rede e opcionalmente simula.
        
        Args:
            solucao (list): Array de valores [0,1] dos diâmetros (saída de otimizar())
            simular (bool): Se True, executa simulação e retorna dados da rede
        
        Returns:
            dict: {
                'diametros': dict com diâmetro de cada tubo,
                'custo_total': custo da solução,
                'resultado_simulacao': dados da simulação (se simular=True),
                'pressoes': DataFrame com pressões de cada nó (se simular=True),
                'pressao_minima': valor mínimo de pressão (se simular=True)
            }
        """
        # Resetar rede
        self._resetar_rede()
        
        # Aplicar diâmetros
        custo = self._atualizar_diametros_rede(solucao)
        
        # Extrair diâmetros de cada tubo
        diametros_dict = {}
        lista_diametros = self.diametros.obter_diametros()
        
        for i, tubo in enumerate(self.rede.wn.pipe_name_list):
            # Mapear valor [0,1] para índice de diâmetro (mesma lógica usada em _atualizar_diametros_rede)
            indice = int(solucao[i] * (len(lista_diametros) - 1))
            indice = min(max(0, indice), len(lista_diametros) - 1)
            diametro_selecionado = lista_diametros[indice]
            diametros_dict[tubo] = diametro_selecionado
        
        resultado = {
            'diametros': diametros_dict,
            'custo_total': custo,
        }
        
        # Simular se solicitado
        if simular:
            resultado_sim = self.rede.simular(verbose='detalhado')
            resultado['resultado_simulacao'] = resultado_sim
            resultado['pressoes'] = self.rede.obter_pressoes()
            
            pressao_info = self.rede.obter_pressao_minima(excluir_reservatorios=True)
            resultado['pressao_minima'] = pressao_info['valor']
            resultado['no_pressao_minima'] = pressao_info['no']
        
        return resultado
    
    def exibir_diametros(self, diametros_dict):
        """
        Exibe os diâmetros de forma formatada.
        
        Args:
            diametros_dict (dict): Dicionário {tubo: diametro}
        """
        print("\n" + "="*70)
        print("DIÂMETROS DA SOLUÇÃO OTIMIZADA")
        print("="*70)
        print(f"{'Tubo':<20} {'Diâmetro (m)':<15} {'Diâmetro (mm)':<15}")
        print("-"*70)
        
        for tubo, diametro in sorted(diametros_dict.items()):
            diametro_mm = diametro * 1000
            print(f"{tubo:<20} {diametro:<15.6f} {diametro_mm:<15.2f}")
        
        print("="*70 + "\n")

    def _definir_workers(self):
        """
        Define número de workers para CPU paralela quando permitido.
        
        Estratégia: Deixa um núcleo livre para o SO não travar.
        """
        if self.usar_gpu:
            return 1
        if not self.usar_paralelismo:
            return 1
        try:
            import os
            if self.n_workers is None:
                cpu_count = os.cpu_count() or 1
                # Deixa um núcleo livre para o SO não travar
                workers = max(1, cpu_count - 1)
                if self.verbose:
                    print(f"📊 Paralelismo: {workers} workers (de {cpu_count} núcleos disponíveis)")
                return workers
            return max(1, int(self.n_workers))
        except Exception:
            return 1

    def _instanciar_modelo(self, metodo, swarm_based, evolutionary_based):
        """Instancia o modelo MealPy correspondente ao método escolhido."""
        params = self.parametros[metodo]

        if metodo == 'PSO':
            return swarm_based.PSO.OriginalPSO(epoch=self.epoch, pop_size=self.pop_size, c1=params['c1'], c2=params['c2'], w=params['w'])
        if metodo == 'GWO':
            return swarm_based.GWO.OriginalGWO(epoch=self.epoch, pop_size=self.pop_size)
        if metodo == 'WOA':
            return swarm_based.WOA.OriginalWOA(epoch=self.epoch, pop_size=self.pop_size, b=params['b'])
        if metodo == 'ABC':
            return swarm_based.ABC.OriginalABC(epoch=self.epoch, pop_size=self.pop_size, limit=params['limit'])
        if metodo == 'CS':
            return swarm_based.CS.OriginalCS(epoch=self.epoch, pop_size=self.pop_size, pa=params['pa'])
        if metodo == 'BA':
            return swarm_based.BA.OriginalBA(epoch=self.epoch, pop_size=self.pop_size, A=params['loudness'], r=params['pulse_rate'])
        if metodo == 'FA':
            return swarm_based.FA.OriginalFA(epoch=self.epoch, pop_size=self.pop_size, alpha=params['alpha'], beta=params['beta'], gamma=params['gamma'])
        if metodo == 'HHO':
            return swarm_based.HHO.OriginalHHO(epoch=self.epoch, pop_size=self.pop_size)
        if metodo == 'DE':
            return evolutionary_based.DE.OriginalDE(epoch=self.epoch, pop_size=self.pop_size, wf=params['wf'], cr=params['cr'])
        if metodo == 'GA':
            return evolutionary_based.GA.BaseGA(epoch=self.epoch, pop_size=self.pop_size, pc=params['pc'], pm=params['pm'])

        raise KeyError(f"Método '{metodo}' não suportado.")
