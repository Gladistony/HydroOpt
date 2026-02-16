import copy
import logging
from tqdm import tqdm
from mealpy.utils.space import FloatVar, IntegerVar
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
        Atualiza os diâmetros da rede baseado na solução (índices inteiros de diâmetros).
        
        Args:
            solution (list): Lista de índices inteiros [0, n_diâmetros-1] dos diâmetros.
        
        Returns:
            float: Custo total dos diâmetros aplicados
        """
        custo_diametros = 0.0
        
        if solution is None or self.diametros is None:
            return custo_diametros
        
        diametros_disponiveis = self.diametros.obter_diametros()
        n_diametros = len(diametros_disponiveis)
        
        # Mapear índices inteiros para diâmetros disponíveis
        for i, pipe_name in enumerate(self.rede.wn.pipe_name_list):
            if i < len(solution):
                # Índice direto do diâmetro (IntegerVar)
                idx = int(round(float(solution[i])))
                idx = min(max(0, idx), n_diametros - 1)
                
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
        # Disponibilizar custo real (somente diâmetros) para o rastreador
        self._ultimo_custo_diametros = float(custo_diametros)
        # Por padrão, considerar inviável até provar o contrário
        self._ultima_viavel = False
        self._ultima_pressao_min = None
        
        # Simular rede com novos diâmetros (sem prints durante otimização)
        resultado = self.rede.simular(verbose=False)

        if not resultado.get('sucesso', False):
            # manter último custo real disponível
            return penalidade_base + custo_diametros

        # Obter pressões reais
        pressao_info = self.rede.obter_pressao_minima(excluir_reservatorios=True, verbose=verbose)
        pressao_min = pressao_info['valor']
        self._ultima_pressao_min = pressao_min
        
        # Se pressão é inválida (inf ou nan), retornar penalidade máxima
        if pressao_min == float('inf') or pressao_min != pressao_min:  # NaN check
            # manter último custo real disponível
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

        # Função objetivo: custo dos diâmetros + penalidade de pressão (erro quadrado removido)
        # Retornamos custo real quando viável; penalties tornam soluções inviáveis muito caras
        custo_final = custo_diametros + penalidade_pressao

        # Atualizar flag de viabilidade (sucesso da simulação + pressão atende)
        self._ultima_viavel = (pressao_min >= self.pressao_min_desejada)

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
    def otimizar(self, metodo='PSO', verbose=False, solucao_inicial=None, rastrear_convergencia=True, seed=None, salvar_solucoes=False):
        """
        Executa otimização usando MealPy com penalização de pressão mínima.

        Args:
            metodo (str): Algoritmo a usar (PSO, GWO, WOA, ABC, CS, BA, FA, HHO, DE, GA)
            verbose (bool): Exibir informações durante otimização
            solucao_inicial (list, optional): População ou solução inicial
            rastrear_convergencia (bool): Rastrear histórico de convergência
            seed (int, optional): Seed para reprodutibilidade
            salvar_solucoes (bool): Se True, salva a solução completa de cada avaliação
                                    no tracker (consome mais memória mas permite análise detalhada)

        Returns:
            dict: {
                'melhor_custo': float,
                'melhor_solucao': list,
                'historico': list,
                'historico_convergencia': array (best-so-far fitness por avaliação),
                'tracker': ConvergenciaTracker (objeto com todos os dados detalhados),
                'estatisticas_convergencia': dict (resumo estatístico da convergência)
            }
        """
        metodo = metodo.upper()
        if metodo not in self.parametros:
            raise KeyError(f"Método '{metodo}' não suportado. Disponíveis: {self.listar_metodos()}")

        # Configurar seed recuperável: se fornecida, usar; senão, gerar e registrar
        self._configurar_seed_interno(seed)

        # Tentar importar mealpy
        try:
            from mealpy import swarm_based, evolutionary_based
        except ImportError:
            raise ImportError("MealPy não está instalado. Adicione 'mealpy' às dependências.")

        # Criar classe derivada de Problem para MealPy 3.0+
        optimizer_instance = self
        n_tubos = len(self.rede.wn.pipe_name_list)

        # Inicializar rastreador de convergência
        if rastrear_convergencia:
            from .visualizador_convergencia import ConvergenciaTracker
            convergencia_tracker = ConvergenciaTracker(pop_size=self.pop_size, salvar_solucoes=salvar_solucoes)
        
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
                # Capturar custo real dos diâmetros desta avaliação (se disponível)
                # Somente reportar custo real quando a solução é viável
                custo_real = None
                if getattr(optimizer_instance, '_ultima_viavel', False):
                    custo_real = getattr(optimizer_instance, '_ultimo_custo_diametros', None)

                pressao_min = getattr(optimizer_instance, '_ultima_pressao_min', None)
                viavel = getattr(optimizer_instance, '_ultima_viavel', False)

                # Rastrear convergência se habilitado
                if rastrear_convergencia:
                    convergencia_tracker.adicionar(
                        value,
                        custo_real=custo_real,
                        pressao_min=pressao_min,
                        viavel=viavel,
                        solucao=solution
                    )

                # Atualizar barra de progresso se estiver definida
                try:
                    pbar = getattr(optimizer_instance, '_pbar', None)
                    if pbar is not None:
                        pbar.update(1)
                except Exception:
                    # Não queremos interromper a avaliação por um erro de UI
                    pass

                return [value]

        # Problema para MealPy 3.0+: Uma variável inteira [0, n_diâmetros-1] para cada tubo
        n_diametros = len(self.diametros.obter_diametros()) if self.diametros else 2
        problem = HydroNetworkProblem(
            bounds=[IntegerVar(lb=0, ub=n_diametros - 1) for _ in range(n_tubos)],
            minmax='min',
            log_to=None,
            save_population=True,  # Salvar população real de cada época
        )

        modelo = self._instanciar_modelo(metodo, swarm_based, evolutionary_based)

        workers = self._definir_workers()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"INICIANDO OTIMIZAÇÃO: {metodo}")
            print(f"{'='*60}")
            print(f"Épocas: {self.epoch} | População: {self.pop_size} | Workers: {workers}")
            print(f"Seed: {self.seed_usado}")
            print(f"{'='*60}\n")

        

        # Criar barra de progresso com tqdm (conta avaliações: épocas * população)
        with tqdm(total=total_evals, desc=f"Otimizando com {metodo}", 
                  unit="avaliação", disable=not self.verbose, ncols=80) as pbar:
            # Expor a barra para o obj_func via instância do otimizador
            optimizer_instance._pbar = pbar

            # Preparar argumentos para solve
            solve_kwargs = {
                'mode': 'single',
                'n_workers': 1,
            }
            
            # --- LÓGICA FLEXÍVEL DE SOLUÇÃO INICIAL ---
            if solucao_inicial is not None:
                import numpy as np

                def _normalizar_individuo(ind, idx=None):
                    arr = np.asarray(ind, dtype=float).ravel()

                    # Se veio um escalar ou vetor de tamanho 1, replicar para todos os tubos
                    if arr.size == 1:
                        arr = np.full(n_tubos, float(arr[0]))

                    # Auto-detectar formato [0,1] antigo e converter para índices inteiros
                    has_fractional = np.any(arr != np.floor(arr))
                    if has_fractional and arr.max() <= 1.0 and arr.min() >= 0.0:
                        arr = np.round(arr * (n_diametros - 1))

                    # Sanitizar NaN/Inf e limitar a [0, n_diametros-1]
                    arr = np.nan_to_num(arr, nan=0.0, posinf=float(n_diametros - 1), neginf=0.0)
                    arr = np.clip(np.round(arr), 0, n_diametros - 1)

                    if arr.size != n_tubos:
                        info_idx = f" (indivíduo {idx})" if idx is not None else ""
                        raise ValueError(
                            f"solucao_inicial{info_idx}: tamanho {arr.size}, esperado {n_tubos}. "
                            f"Forneça um índice de diâmetro [0, {n_diametros-1}] para cada tubo."
                        )
                    return arr

                # Detectar formato fornecido
                if isinstance(solucao_inicial, np.ndarray):
                    if solucao_inicial.ndim == 1:
                        # Vetor único
                        solucao_unica = _normalizar_individuo(solucao_inicial)
                        populacao_final = [solucao_unica]
                        qtd_restante = int(self.pop_size) - 1
                        if qtd_restante > 0:
                            aleatorios = np.random.randint(0, n_diametros, (qtd_restante, n_tubos)).astype(float)
                            populacao_final.extend(aleatorios)
                        if self.verbose:
                            print(f"🚀 Warm Start: Gerando {self.pop_size - 1} indivíduos aleatórios a partir da guia.")
                        solve_kwargs['starting_solutions'] = populacao_final
                    elif solucao_inicial.ndim == 2:
                        # Matriz (população completa)
                        if solucao_inicial.shape[0] != self.pop_size:
                            print(f"⚠️ AVISO: População inicial tem {solucao_inicial.shape[0]} indivíduos, mas pop_size é {self.pop_size}.")
                        populacao_np = [ _normalizar_individuo(solucao_inicial[i], idx=i) for i in range(solucao_inicial.shape[0]) ]
                        if self.verbose:
                            print(f"🚀 Usando população inicial personalizada ({len(populacao_np)} indivíduos).")
                        solve_kwargs['starting_solutions'] = populacao_np
                    else:
                        raise ValueError("solucao_inicial numpy deve ser vetor 1D ou matriz 2D")
                else:
                    # Listas/tuplas
                    eh_solucao_unica = isinstance(solucao_inicial[0], (int, float))
                    if eh_solucao_unica:
                        if self.verbose:
                            print(f"🚀 Warm Start: Gerando {self.pop_size - 1} indivíduos aleatórios a partir da guia.")
                        solucao_unica = _normalizar_individuo(solucao_inicial)
                        populacao_final = [solucao_unica]
                        qtd_restante = int(self.pop_size) - 1
                        if qtd_restante > 0:
                            aleatorios = np.random.randint(0, n_diametros, (qtd_restante, n_tubos)).astype(float)
                            populacao_final.extend(aleatorios)
                        solve_kwargs['starting_solutions'] = populacao_final
                    else:
                        if len(solucao_inicial) != self.pop_size:
                            print(f"⚠️ AVISO: População inicial tem {len(solucao_inicial)} indivíduos, mas pop_size é {self.pop_size}.")
                        populacao_np = [_normalizar_individuo(sol, idx=i) for i, sol in enumerate(solucao_inicial)]
                        if self.verbose:
                            print(f"🚀 Usando população inicial personalizada ({len(populacao_np)} indivíduos).")
                        solve_kwargs['starting_solutions'] = populacao_np

            # Rodar otimização (MealPy 3.0+)
            # Usar 'single' para evitar problemas de memória com WNTR em multithread/multiprocess
            agent = modelo.solve(problem, **solve_kwargs)
            
            # Extrair resultados do agent retornado
            melhor_solucao = agent.solution
            melhor_custo = agent.target.objectives[0]
            
            # Remover referência à barra
            optimizer_instance._pbar = None

        self._resetar_rede()
        # Calculamos o custo financeiro puro (sem penalidades de pressão)
        custo_real_investimento = self._atualizar_diametros_rede(melhor_solucao)
        # -----------------------------------------------------

        # Extrair histórico por época do MealPy (número real de épocas)
        historico_epocas_mealpy = []
        try:
            if hasattr(modelo, 'history') and modelo.history is not None:
                for ep_agent in modelo.history.list_global_best:
                    historico_epocas_mealpy.append(ep_agent.target.objectives[0])
                
                # ===== EXTRAIR POPULAÇÃO REAL DE CADA ÉPOCA =====
                # O MealPy armazena a população completa (com fitness atualizado)
                # ao final de cada época. Isso inclui entidades que NÃO foram
                # reavaliadas pelo obj_func (ex: lobos que copiaram a posição do Alpha).
                # Esses dados são a fonte REAL para análise estatística do enxame.
                if rastrear_convergencia and hasattr(modelo.history, 'list_population'):
                    populacoes = modelo.history.list_population
                    if len(populacoes) > 0:
                        convergencia_tracker.definir_populacao_por_epoca(populacoes)
                    elif len(historico_epocas_mealpy) > 0:
                        # Fallback: se list_population estiver vazio, recalcular por contagem
                        convergencia_tracker.recalcular_epocas(len(historico_epocas_mealpy))
                elif rastrear_convergencia and len(historico_epocas_mealpy) > 0:
                    convergencia_tracker.recalcular_epocas(len(historico_epocas_mealpy))
        except Exception:
            pass
        
        # Rastrear convergência final se habilitado (inclui custo real estimado)
        if rastrear_convergencia:
            convergencia_tracker.adicionar(
                melhor_custo,
                custo_real=custo_real_investimento,
                pressao_min=None,
                viavel=True,
                solucao=melhor_solucao
            )

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"✓ Otimização concluída com {metodo}")
            print(f"{'='*60}")
            # Exibe o Fitness (Score matemático)
            print(f"  🔹 Melhor Fitness (Score):   {melhor_custo:.6f}")
            # Exibe o Dinheiro (O que importa para o engenheiro)
            print(f"  💰 Custo Real Estimado:      R$ {custo_real_investimento:,.2f}")
            print(f"{'='*60}\n")

        resultado = {
            'melhor_custo': melhor_custo,
            'melhor_solucao': melhor_solucao,
            'historico': [melhor_custo],  # MealPy 3.0 não retorna histórico completo
            'seed_usado': getattr(self, 'seed_usado', None),
            'custo_real': custo_real_investimento,
        }
        
        # Adicionar histórico de convergência se rastreado
        if rastrear_convergencia:
            resultado['historico_convergencia'] = convergencia_tracker.obter_historico()
            resultado['historico_custo_real'] = convergencia_tracker.obter_historico_custo_real()
            resultado['historico_custo_real_bsf'] = convergencia_tracker.acumular_melhor_custo_real()
            resultado['historico_pressao_min'] = convergencia_tracker.obter_historico_pressao_min()
            resultado['historico_viavel'] = convergencia_tracker.obter_historico_viavel()
            resultado['historico_fitness_bruto'] = convergencia_tracker.obter_historico_bruto()
            resultado['tracker'] = convergencia_tracker
            resultado['estatisticas_convergencia'] = convergencia_tracker.obter_estatisticas()
            
            # Dados por época (para gráficos de convergência suave)
            resultado['melhor_por_epoca'] = convergencia_tracker.obter_melhor_por_epoca()
            resultado['media_por_epoca'] = convergencia_tracker.obter_media_por_epoca()
            resultado['desvio_por_epoca'] = convergencia_tracker.obter_desvio_por_epoca()
            resultado['todos_por_epoca'] = convergencia_tracker.obter_todos_por_epoca()
            resultado['epocas'] = convergencia_tracker.epocas
            
            # Histórico do MealPy (best-so-far por época via MealPy)
            if historico_epocas_mealpy:
                resultado['historico_mealpy_por_epoca'] = historico_epocas_mealpy
        
        return resultado

    # -------------------------
    # Seed recuperável
    # -------------------------
    def configurar_seed(self, seed=None):
        """
        Configura a seed de forma recuperável. Se `seed` for None,
        gera uma seed aleatória cripto-segura e registra.

        Afeta o gerador global do NumPy e do Python `random`.

        Args:
            seed (int|None): Seed desejada ou None para gerar aleatória.

        Returns:
            int: Seed efetivamente usada.
        """
        return self._configurar_seed_interno(seed)

    def _configurar_seed_interno(self, seed=None):
        import numpy as np
        import random
        import secrets

        # Se veio uma seed explícita, normalizar para int
        if seed is not None:
            try:
                seed_int = int(seed)
            except Exception:
                raise ValueError("seed deve ser conversível para inteiro")
        else:
            # Gerar seed aleatória 32-bit cripto-segura
            seed_int = secrets.randbits(32)

        # Aplicar seed nos geradores comuns
        np.random.seed(seed_int)
        random.seed(seed_int)

        # Registrar para recuperação
        self.seed_usado = int(seed_int)

        # Sempre imprimir a seed para fins de reprodução
        if self.verbose:
            if seed is not None:
                print(f"🔑 Seed fornecida: {self.seed_usado}")
            else:
                print(f"🔑 Seed gerada automaticamente: {self.seed_usado}  (guarde para reprodução)")

        return self.seed_usado

    def aplicar_solucao(self, solucao, simular=True):
        """
        Aplica uma solução otimizada à rede e opcionalmente simula.
        
        Args:
            solucao (list): Array de índices inteiros dos diâmetros (saída de otimizar())
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
            # Índice direto do diâmetro (mesma lógica usada em _atualizar_diametros_rede)
            indice = int(round(float(solucao[i])))
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
