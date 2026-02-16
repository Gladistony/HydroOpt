"""
Benchmark completo de redes de distribuição de água (WDN).

Usa a API atualizada do HydroOpt com IntegerVar (índices inteiros de diâmetros),
rastreamento por época e AnalisadorEstatistico.

Gera logs .npz com shape [Épocas, PopSize] e relatório CSV.
"""

import os
import csv
import time
import numpy as np
import warnings
import itertools
import secrets
from tqdm import tqdm

# ==============================================================================
# IMPORTAÇÕES DA SUA BIBLIOTECA (HYDROOPT)
# ==============================================================================
try:
    from HydroOpt.rede import Rede
    from HydroOpt.diametros import LDiametro
    from HydroOpt.otimizador import Otimizador
    from HydroOpt.core import gerar_solucao_heuristica
    from HydroOpt.analisador_estatistico import AnalisadorEstatistico
except ImportError as e:
    print(f"ERRO CRÍTICO: Não foi possível importar o HydroOpt. Verifique se a pasta 'HydroOpt' está no diretório.")
    raise e

# Importações do MealPy
import secrets
from mealpy import swarm_based, evolutionary_based
from mealpy.utils.problem import Problem
from mealpy.utils.space import IntegerVar

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
def executar_cenario(metodo, params, rede_base, diametros, seed_solucao, log_dir):
    """
    Executa uma otimização individual.

    Args:
        metodo: Nome do algoritmo (PSO, GWO, WOA, ABC, GA)
        params: Dict com parâmetros + epoch + pop_size
        rede_base: Instância de Rede
        diametros: Instância de LDiametro
        seed_solucao: Solução heurística (lista de índices inteiros) ou None
        log_dir: Pasta para salvar logs

    Returns:
        dict com resultados
    """
    p = params.copy()
    epoch = p.pop('epoch', 100)
    pop_size = p.pop('pop_size', 50)

    otimizador = Otimizador(
        rede=rede_base,
        diametros=diametros,
        pressao_min_desejada=30.0,
        epoch=epoch,
        pop_size=pop_size,
        verbose=False,
        usar_paralelismo=False,
    )

    # Configurar parâmetros do algoritmo
    if p:
        otimizador.definir_parametros(metodo, **p)

    # Gerar seed aleatória e registrar para reprodução
    seed_int = secrets.randbits(32)

    # Montar solução inicial (warm start)
    solucao_inicial = None
    if seed_solucao is not None:
        solucao_inicial = seed_solucao  # Já é lista de índices inteiros

    # Executar otimização via API do HydroOpt
    resultado = otimizador.otimizar(
        metodo=metodo,
        verbose=False,
        solucao_inicial=solucao_inicial,
        rastrear_convergencia=True,
        seed=seed_int,
        salvar_solucoes=False,
    )

    # Construir matriz hist_fit [Épocas, PopSize] a partir dos dados por época
    todos_por_epoca = resultado.get('todos_por_epoca', [])
    if todos_por_epoca and len(todos_por_epoca) > 0:
        # Padronizar tamanho (algumas épocas podem ter N diferente — ex: ABC)
        max_pop = max(len(ep) for ep in todos_por_epoca)
        hist_fit = np.full((len(todos_por_epoca), max_pop), np.nan)
        for i, ep_fits in enumerate(todos_por_epoca):
            hist_fit[i, :len(ep_fits)] = ep_fits
    else:
        # Fallback: usar histórico bruto (uma coluna só, como antes)
        hist_fit = np.array(resultado.get('historico_fitness_bruto', []))
        if hist_fit.ndim == 1:
            hist_fit = hist_fit.reshape(-1, 1)

    return {
        'melhor_custo': resultado['melhor_custo'],
        'melhor_design': resultado['melhor_solucao'],
        'custo_real': resultado.get('custo_real', np.nan),
        'seed_usado': resultado.get('seed_usado', seed_int),
        'hist_fit': hist_fit,
        'tracker': resultado.get('tracker'),
        'resultado_completo': resultado,
    }

# ==============================================================================
# 4. LOOP PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    INP_FILE = "hanoiFIM"
    RESULT_CSV = "resultado_hydroopt_final.csv"
    LOG_DIR = "logs_detalhados_hydroopt"
    
    # Limpar execuções anteriores
    print(">>> Limpando arquivos anteriores...")
    if os.path.exists(RESULT_CSV):
        os.remove(RESULT_CSV)
        print(f"  ✓ Removido {RESULT_CSV}")
    
    if os.path.exists(LOG_DIR):
        import shutil
        shutil.rmtree(LOG_DIR)
        print(f"  ✓ Removido {LOG_DIR}/")
    
    os.makedirs(LOG_DIR, exist_ok=True)

    print(">>> [1/3] Carregando Rede e Diâmetros...")
    try:
        rede_teste = Rede(INP_FILE)
        ld = LDiametro()
        ld.adicionar_polegadas(12, 45.73)
        ld.adicionar_polegadas(16, 70.40)
        ld.adicionar_polegadas(20, 98.38)
        ld.adicionar_polegadas(24, 129.30)
        ld.adicionar_polegadas(30, 180.80)
        ld.adicionar_polegadas(40, 278.30)
    except Exception as e:
        print(f"❌ Erro ao carregar rede: {e}")
        exit(1)

    print(">>> [2/3] Gerando Heurística Inicial (Warm Start)...")
    try:
        # Agora retorna índices inteiros [0, n_diametros-1]
        solucao_heuristica = gerar_solucao_heuristica(rede_teste, ld, pressao_min_desejada=30.0, verbose=False)
        print(f"✓ Heurística gerada com sucesso. (Indices: min={min(solucao_heuristica)}, max={max(solucao_heuristica)})")
    except Exception as e:
        print(f"⚠️ Falha na heurística: {e}. Usando inicialização aleatória.")
        solucao_heuristica = None

    # --- DEFINIÇÃO DOS CENÁRIOS ---
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
            # (Usar Heuristica?, seed_solucao)
            modos = [(False, None), (True, solucao_heuristica)]

            for usar_heuristica, seed_atual in modos:
                if usar_heuristica and seed_atual is None:
                    continue
                # PSO roda APENAS com heurística
                if nome_modelo == 'PSO' and not usar_heuristica:
                    continue

                jobs.append((nome_modelo, params, usar_heuristica, seed_atual))

    # Criar arquivo CSV com cabeçalhos
    with open(RESULT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Algoritmo', 'Usou_Heuristica', 'Parametros',
            'Melhor_Custo', 'Custo_Real', 'Seed_Usado',
            'Tempo_s', 'Arquivo_NPZ', 'Status'
        ])

    print(f"\n>>> [3/3] Iniciando Benchmark: {len(jobs)} jobs totais.")

    for i, (algoritmo, params, usar_heuristica, seed) in enumerate(jobs, start=1):
        start_time = time.time()
        label = "WARM" if usar_heuristica else "COLD"

        safe_params = (str(params).replace(" ", "").replace(":", "")
                       .replace("'", "").replace("{", "").replace("}", "")
                       .replace(",", "_"))
        npz_file = f"{algoritmo}_{label}_{safe_params}.npz"

        print(f"[{i}/{len(jobs)}] {algoritmo} ({label}) | {params}")

        try:
            resultado = executar_cenario(algoritmo, params, rede_teste, ld, seed, LOG_DIR)
            duration = time.time() - start_time
            custo = resultado['melhor_custo']
            custo_real = resultado.get('custo_real', np.nan)
            seed_usado = resultado.get('seed_usado', 'N/A')
            status = "VALIDO" if custo < 50_000_000 else "INVALIDO"

            # Salva NPZ (agora com shape [Épocas, PopSize])
            if status == "VALIDO":
                npz_path = os.path.join(LOG_DIR, npz_file)
                np.savez_compressed(
                    npz_path,
                    hist_fit=resultado['hist_fit'],
                    config=str(params),
                    seed_usado=seed_usado,
                )
            else:
                npz_file = ""

            # Análise estatística automática (se tracker disponível)
            if resultado.get('tracker') is not None:
                try:
                    analise = AnalisadorEstatistico(resultado=resultado['resultado_completo'])
                    metricas = analise.calcular()
                    err_medio = metricas.get('erro_medio_particulas', np.nan)
                    div = metricas.get('diversidade_media', np.nan)
                    print(f"    📊 Erro médio partículas: {err_medio:.2f} | Diversidade média: {div:.2f}")
                except Exception:
                    pass

            with open(RESULT_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    algoritmo, str(usar_heuristica), str(params),
                    f"{custo:.2f}", f"{custo_real:.2f}", str(seed_usado),
                    f"{duration:.2f}", npz_file, status
                ])
            print(f"    --> ${custo:,.2f} (real: ${custo_real:,.2f}) ({status}) [seed={seed_usado}]")

        except Exception as e:
            print(f"    --> ERRO: {e}")
            with open(RESULT_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    algoritmo, str(usar_heuristica), str(params),
                    "0", "0", "N/A", "0", "", f"ERRO: {e}"
                ])