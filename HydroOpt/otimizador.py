import copy


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
        """Retorna penalidade base derivada dos diâmetros (se fornecidos)."""
        if self.diametros is not None:
            try:
                return self.diametros.obter_penalidade()
            except Exception:
                return 1e5
        return 1e5

    def _avaliar_rede(self):
        """
        Simula a rede e calcula custo com penalidade se pressão mínima < desejada.
        """
        resultado = self.rede.simular()
        penalidade_base = self._penalidade_base()

        if not resultado.get('sucesso', False):
            return penalidade_base

        pressao_min = self.rede.obter_pressao_minima(excluir_reservatorios=True)['valor']

        if pressao_min < self.pressao_min_desejada:
            return penalidade_base * (self.pressao_min_desejada - pressao_min + 1)

        # Custo base: 0 (placeholder). Aqui poderíamos somar custos de diâmetros etc.
        return 0.0

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

    # ------------------------------------------------------------------
    # Execução de otimização (MealPy)
    # ------------------------------------------------------------------
    def otimizar(self, metodo='PSO'):
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

        # Função objetivo ignora x (placeholder) e usa simulação da rede
        def obj_func(x):
            return self._avaliar_rede()

        # Problema genérico (variável dummy só para compatibilidade)
        problem = {
            'obj_func': obj_func,
            'lb': [0],
            'ub': [1],
            'minmax': 'min',
        }

        modelo = self._instanciar_modelo(metodo, swarm_based, evolutionary_based)

        workers = self._definir_workers()

        # Rodar otimização (MealPy 2.5+ aceita paralelismo via n_workers)
        melhor_solucao, melhor_custo, historico = modelo.solve(
            problem,
            n_workers=workers,
            verbose=self.verbose,
        )

        if self.verbose:
            print(f"\n✓ Otimização concluída com {metodo}: melhor custo = {melhor_custo}")

        return {
            'melhor_custo': melhor_custo,
            'melhor_solucao': melhor_solucao,
            'historico': historico,
        }

    def _definir_workers(self):
        """Define número de workers para CPU paralela quando permitido."""
        if self.usar_gpu:
            return 1
        if not self.usar_paralelismo:
            return 1
        try:
            import os
            if self.n_workers is None:
                return max(1, os.cpu_count() or 1)
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
