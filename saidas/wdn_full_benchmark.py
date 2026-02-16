import os
import csv
import time
import numpy as np
import warnings
import itertools
from tqdm import tqdm

# ==============================================================================
# IMPORTAÇÕES DA SUA BIBLIOTECA (HYDROOPT)
# ==============================================================================
try:
    from HydroOpt.rede import Rede
    from HydroOpt.diametros import LDiametro
    from HydroOpt.otimizador import Otimizador
    from HydroOpt.core import gerar_solucao_heuristica
except ImportError as e:
    print(f"ERRO CRÍTICO: Não foi possível importar o HydroOpt. Verifique se a pasta 'HydroOpt' está no diretório.")
    raise e

# Importações do MealPy
from mealpy import swarm_based, evolutionary_based
from mealpy.utils.problem import Problem
from mealpy.utils.space import FloatVar

# ==============================================================================
# 1. CLASSE OTIMIZADOR CUSTOMIZADA
# ==============================================================================
class OtimizadorCustomizado(Otimizador):
    def _avaliar_rede(self, solution=None, verbose=False):
        self._resetar_rede()
        custo_diametros = self._atualizar_diametros_rede(solution)
        
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                sim_res = self.rede.simular(verbose=False)
                
                if len(w) > 0:
                    for warning in w:
                        msg = str(warning.message).lower()
                        if "converge" in msg or "balanced" in msg:
                            return 1e12 
                
                if not sim_res.get('sucesso', False):
                    return 1e12

            p_info = self.rede.obter_pressao_minima(excluir_reservatorios=True, verbose=False)
            pressao_min = p_info['valor']

            if np.isnan(pressao_min) or np.isinf(pressao_min):
                return 1e12

            if pressao_min < self.pressao_min_desejada:
                diferenca = self.pressao_min_desejada - pressao_min
                penalidade = 1e7 * (diferenca ** 2) + 1e8 
                return custo_diametros + penalidade
            
            self._ultima_viavel = True
            return custo_diametros

        except Exception:
            return 1e12

# ==============================================================================
# 2. HELPER: GERADOR DE COMBINAÇÕES (VARIADOR)
# ==============================================================================
class Variador:
    def __init__(self):
        self.params_grid = {}
    def definir_parametro(self, nome, inicial, final, passo):
        if isinstance(inicial, int) and isinstance(passo, int):
             valores = np.arange(inicial, final + 1, passo).tolist()
        else:
            epsilon = passo / 1000.0
            valores = np.arange(inicial, final + epsilon, passo).tolist()
            valores = [round(x, 4) for x in valores]
        self.params_grid[nome] = valores
    def gerar_combinacoes(self):
        nomes = list(self.params_grid.keys())
        valores = list(self.params_grid.values())
        return [dict(zip(nomes, combo)) for combo in itertools.product(*valores)]

# ==============================================================================
# 3. FUNÇÃO DE EXECUÇÃO
# ==============================================================================
def executar_cenario(metodo, params, rede_base, diametros, seed, log_dir):
    otimizador = OtimizadorCustomizado(
        rede=rede_base,
        diametros=diametros,
        pressao_min_desejada=30.0,
        verbose=False
    )
    
    n_vars = len(rede_base.wn.pipe_name_list)
    
    class HydroProblem(Problem):
        def obj_func(self, solution):
            return otimizador._avaliar_rede(solution, verbose=False)

    problem = HydroProblem(
        bounds=[FloatVar(lb=0, ub=1) for _ in range(n_vars)],
        minmax='min',
        log_to=None
    )

    p = params.copy()
    epoch = p.pop('epoch', 100)
    pop_size = p.pop('pop_size', 50)
    
    if metodo == 'PSO':
        model = swarm_based.PSO.OriginalPSO(epoch=epoch, pop_size=pop_size, **p)
    elif metodo == 'GWO':
        model = swarm_based.GWO.OriginalGWO(epoch=epoch, pop_size=pop_size, **p)
    elif metodo == 'WOA':
        model = swarm_based.WOA.OriginalWOA(epoch=epoch, pop_size=pop_size, **p)
    elif metodo == 'ABC':
        model = swarm_based.ABC.OriginalABC(epoch=epoch, pop_size=pop_size, **p)
    elif metodo == 'GA':
        model = evolutionary_based.GA.BaseGA(epoch=epoch, pop_size=pop_size, **p)
    else:
        raise ValueError(f"Método {metodo} desconhecido.")

    solve_kwargs = {'mode': 'single', 'n_workers': 1}
    
    if seed is not None:
        start_solutions = [np.array(seed)] 
        for _ in range(pop_size - 1):
             start_solutions.append(np.random.uniform(0, 1, n_vars))
        solve_kwargs['starting_solutions'] = start_solutions

    if hasattr(model, 'save_population'):
        model.save_population = True

    agent = model.solve(problem, **solve_kwargs)
    
    historico_populacao = [] 
    historico_fitness = []   
    
    try:
        if hasattr(model, 'history') and hasattr(model.history, 'list_population') and model.history.list_population:
            for epoch_agents in model.history.list_population:
                pop_pos = [ind.solution for ind in epoch_agents]
                pop_fit = [ind.target.fitness for ind in epoch_agents]
                historico_populacao.append(pop_pos)
                historico_fitness.append(pop_fit)
        elif hasattr(model, 'history') and hasattr(model.history, 'list_global_best') and model.history.list_global_best:
            for ind in model.history.list_global_best:
                historico_populacao.append([ind.solution]) 
                historico_fitness.append([ind.target.fitness])
        elif hasattr(model, 'history') and isinstance(model.history, list):
             historico_fitness = model.history
    except Exception:
        pass

    return {
        'melhor_custo': agent.target.fitness,
        'melhor_design': agent.solution,
        'hist_pop': np.array(historico_populacao, dtype=object),
        'hist_fit': np.array(historico_fitness, dtype=object)
    }

# ==============================================================================
# 4. LOOP PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    INP_FILE = "Hanoi.inp" 
    RESULT_CSV = "resultado_hydroopt_final.csv"
    LOG_DIR = "logs_detalhados_hydroopt"
    os.makedirs(LOG_DIR, exist_ok=True)

    print(">>> [1/3] Carregando Rede e Diâmetros...")
    try:
        rede_teste = Rede(INP_FILE)
        ld = LDiametro()
        ld.adicionar(12*0.0254, 45.73).adicionar(16*0.0254, 70.40).adicionar(20*0.0254, 98.38)
        ld.adicionar(24*0.0254, 129.30).adicionar(30*0.0254, 180.80).adicionar(40*0.0254, 278.30)
    except Exception as e:
        print(f"❌ Erro ao carregar rede: {e}")
        exit(1)

    print(">>> [2/3] Gerando Heurística Inicial (Warm Start)...")
    try:
        solucao_heuristica = gerar_solucao_heuristica(rede_teste, ld, pressao_min_desejada=30.0, verbose=False)
        print("✓ Heurística gerada com sucesso.")
    except Exception as e:
        print(f"⚠️ Falha na heurística: {e}. Usando inicialização aleatória.")
        solucao_heuristica = None

    # --- DEFINIÇÃO DOS CENÁRIOS (IGUAL AO ANTERIOR) ---
    v_pso = Variador()
    v_pso.definir_parametro('pop_size', 20, 100, 50)
    v_pso.definir_parametro('epoch', 50, 100, 50)
    v_pso.definir_parametro('w', 0.4, 0.9, 0.25)
    v_pso.definir_parametro('c1', 1.5, 2.5, 0.5)
    v_pso.definir_parametro('c2', 1.5, 2.5, 0.5)

    v_gwo = Variador()
    v_gwo.definir_parametro('pop_size', 20, 100, 50)
    v_gwo.definir_parametro('epoch', 50, 100, 50)

    v_abc = Variador()
    v_abc.definir_parametro('pop_size', 20, 100, 50)
    v_abc.definir_parametro('epoch', 50, 100, 50)
    v_abc.definir_parametro('limit', 10, 50, 20)

    v_ga = Variador()
    v_ga.definir_parametro('pop_size', 20, 100, 50)
    v_ga.definir_parametro('epoch', 50, 100, 50)
    v_ga.definir_parametro('mutation_rate', 0.05, 0.25, 0.1)

    v_woa = Variador()
    v_woa.definir_parametro('pop_size', 20, 100, 50)
    v_woa.definir_parametro('epoch', 50, 100, 50)

    TAREFAS = [
        ('PSO', v_pso), ('GWO', v_gwo), ('ABC', v_abc), ('GA', v_ga), ('WOA', v_woa)
    ]

    # Prepara lista de jobs
    jobs = []
    for nome_modelo, variador in TAREFAS:
        combinacoes = variador.gerar_combinacoes()
        for params in combinacoes:
            # Lógica: (Usar Heuristica?, Seed)
            modos = [(False, None), (True, solucao_heuristica)]
            
            for usar_heuristica, seed_atual in modos:
                # Regras de Execução:
                # 1. Se for rodar com heurística mas ela falhou (None), pula.
                if usar_heuristica and seed_atual is None: continue
                
                # 2. REGRA DO PSO: "Mesmo que antes" => PSO roda APENAS COM heurística
                if nome_modelo == 'PSO' and not usar_heuristica: continue
                
                jobs.append((nome_modelo, params, usar_heuristica, seed_atual))

    # Verifica checkpoint
    tarefas_concluidas = set()
    if os.path.exists(RESULT_CSV):
        with open(RESULT_CSV, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) > 2:
                    chave = f"{row[0]}_{row[1]}_{row[2]}" # Algo_Heuristica_Params
                    tarefas_concluidas.add(chave)
    else:
        with open(RESULT_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Algoritmo', 'Usou_Heuristica', 'Parametros', 'Melhor_Custo', 'Tempo_s', 'Arquivo_NPZ', 'Status'])

    print(f"\n>>> [3/3] Iniciando Benchmark: {len(jobs)} jobs totais.")
    
    for i, (algoritmo, params, usar_heuristica, seed) in enumerate(jobs, start=1):
        # Chave para checkpoint
        chave_atual = f"{algoritmo}_{str(usar_heuristica)}_{str(params)}"
        if chave_atual in tarefas_concluidas:
            continue

        start_time = time.time()
        label = "WARM" if usar_heuristica else "COLD"
        
        # Nome do arquivo seguro
        safe_params = str(params).replace(" ", "").replace(":", "").replace("'", "").replace("{","").replace("}","").replace(",","_")
        npz_file = f"{algoritmo}_{label}_{safe_params}.npz"
        
        print(f"[{i}/{len(jobs)}] {algoritmo} ({label}) | {params}")
        
        try:
            resultado = executar_cenario(algoritmo, params, rede_teste, ld, seed, LOG_DIR)
            duration = time.time() - start_time
            custo = resultado['melhor_custo']
            status = "VALIDO" if custo < 50000000 else "INVALIDO"
            
            # Salva NPZ se for válido
            if status == "VALIDO":
                npz_path = os.path.join(LOG_DIR, npz_file)
                np.savez_compressed(
                    npz_path,
                    hist_pop=resultado['hist_pop'],
                    hist_fit=resultado['hist_fit'],
                    config=str(params)
                )
            else:
                npz_file = "" # Não salvou arquivo

            with open(RESULT_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    algoritmo, str(usar_heuristica), str(params), 
                    f"{custo:.2f}", f"{duration:.2f}", npz_file, status
                ])
            print(f"    --> ${custo:,.2f} ({status})")

        except Exception as e:
            print(f"    --> ERRO: {e}")
            with open(RESULT_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([algoritmo, str(usar_heuristica), str(params), "0", "0", "", f"ERRO: {e}"])