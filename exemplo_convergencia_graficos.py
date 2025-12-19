"""
Exemplo de Convergência e Análise Gráfica

Demonstra como rastrear e visualizar a convergência de otimizações,
comparando diferentes parâmetros e algoritmos.
"""

import HydroOpt
print(f"Versão HydroOpt: {HydroOpt.__version__}")

from HydroOpt import Rede, Otimizador, LDiametro, VariadorDeParametros, VisualizadorConvergencia
from HydroOpt.core import gerar_solucao_heuristica
import numpy as np

# --- CONFIGURAÇÃO INICIAL ---

print("\n" + "="*70)
print("EXEMPLO: VISUALIZAÇÃO DE CONVERGÊNCIA")
print("="*70)

# 1. Criar e simular rede
minha_rede = Rede('hanoiFIM')
print("\nSimulando estado inicial...")
minha_rede.simular()
pressao_inicial = minha_rede.obter_pressao_minima(excluir_reservatorios=True)
print(f"   Pressão mínima inicial: {pressao_inicial['valor']:.2f} m")

# 2. Configurar diâmetros disponíveis
lista_diametros = LDiametro()
lista_diametros.adicionar_polegadas(12, 45.73)   # 12"
lista_diametros.adicionar_polegadas(16, 70.40)   # 16"
lista_diametros.adicionar_polegadas(20, 98.38)   # 20"
lista_diametros.adicionar_polegadas(24, 129.30)  # 24"
lista_diametros.adicionar_polegadas(30, 180.80)  # 30"
lista_diametros.adicionar_polegadas(40, 278.30)  # 40"

# 3. Criar otimizador com parâmetros fixos
populacao_tamanho = 50
pressao_alvo = 30.0

meu_otimizador = Otimizador(
    rede=minha_rede,
    diametros=lista_diametros,
    epoch=50,  # Número de épocas (iterações)
    pop_size=populacao_tamanho,
    usar_paralelismo=False,  # Desativar para evitar problemas
    verbose=False
)

meu_otimizador.pressao_min_desejada = pressao_alvo

# 4. Gerar população inicial (sempre a mesma para comparação justa)
print("\nGerando população inicial...")
solucao_guia = gerar_solucao_heuristica(minha_rede, lista_diametros, pressao_min_desejada=pressao_alvo)
num_tubos = len(minha_rede.wn.pipe_name_list)
qtd_aleatorios = populacao_tamanho - 1
populacao_aleatoria = np.random.uniform(0.0, 1.0, (qtd_aleatorios, num_tubos)).tolist()
minha_populacao_inicial = [solucao_guia] + populacao_aleatoria

# --- EXEMPLO 1: Comparar diferentes valores de c1 no PSO ---

print("\n" + "="*70)
print("TESTE 1: IMPACTO DO PARÂMETRO c1 (Coeficiente Cognitivo)")
print("="*70)

# Criar visualizador
viz_convergencia = VisualizadorConvergencia(verbose=True, 
                                           titulo_padrao="Convergência PSO - Variando c1")

# Testar diferentes valores de c1
valores_c1 = [1.5, 2.0, 2.5]

for c1 in valores_c1:
    print(f"\n→ Executando PSO com c1={c1}...")
    
    # Configurar parâmetros
    meu_otimizador.definir_parametros('PSO', 
        c1=c1, 
        c2=2.0,  # Manter c2 fixo
        w=0.4    # Manter w fixo
    )
    
    # Executar com rastreamento de convergência
    resultado = meu_otimizador.otimizar(
        metodo='PSO',
        solucao_inicial=minha_populacao_inicial,
        rastrear_convergencia=True  # ← Ativar rastreamento
    )
    
    # Adicionar ao visualizador
    if 'historico_convergencia' in resultado:
        viz_convergencia.adicionar_convergencia(
            resultado['historico_convergencia'],
            label=f"c1={c1}",
            dados_adicionais={'custo_real': resultado['melhor_custo']}
        )

# Plotar comparação
print("\nGerando gráfico...")
viz_convergencia.plotar(
    titulo="Convergência PSO: Impacto do Parâmetro c1",
    xlabel="Iteração",
    ylabel="Melhor Fitness",
    salvar_em='grafico_c1_convergencia.png'
)

# Exibir resumo
viz_convergencia.exibir_resumo()

# Análise de convergência
viz_convergencia.exibir_analise_convergencia(threshold_melhoria=0.01)

# --- EXEMPLO 2: Comparar diferentes algoritmos ---

print("\n" + "="*70)
print("TESTE 2: COMPARAÇÃO DE ALGORITMOS")
print("="*70)

# Criar novo visualizador para comparação de algoritmos
viz_algoritmos = VisualizadorConvergencia(verbose=True,
                                         titulo_padrao="Convergência: PSO vs GWO vs WOA")

algoritmos_testar = [
    ('PSO', {'c1': 2.05, 'c2': 2.05, 'w': 0.4}),
    ('GWO', {}),
    ('WOA', {'b': 1.0})
]

for metodo, params in algoritmos_testar:
    print(f"\n→ Executando {metodo}...")
    
    # Configurar parâmetros
    if params:
        meu_otimizador.definir_parametros(metodo, **params)
    
    # Executar
    resultado = meu_otimizador.otimizar(
        metodo=metodo,
        solucao_inicial=minha_populacao_inicial,
        rastrear_convergencia=True
    )
    
    # Adicionar ao visualizador
    if 'historico_convergencia' in resultado:
        viz_algoritmos.adicionar_convergencia(
            resultado['historico_convergencia'],
            label=metodo,
            dados_adicionais={'fitness_final': resultado['melhor_custo']}
        )

# Plotar comparação
print("\nGerando gráfico comparativo...")
viz_algoritmos.plotar(
    titulo="Comparação de Convergência: PSO vs GWO vs WOA",
    xlabel="Iteração",
    ylabel="Melhor Fitness",
    salvar_em='grafico_algoritmos_convergencia.png'
)

viz_algoritmos.exibir_resumo()

# --- EXEMPLO 3: Variador de Parâmetros com Rastreamento de Convergência ---

print("\n" + "="*70)
print("TESTE 3: VARREDURA DE PARÂMETROS COM GRÁFICOS")
print("="*70)

# Criar variador
variador = VariadorDeParametros(meu_otimizador, verbose=True)

# Definir ranges (pequeno para rapidez)
variador.definir_parametro('c1', inicial=1.5, final=2.5, passo=0.5)
variador.definir_parametro('c2', inicial=1.5, final=2.5, passo=0.5)

variador.definir_condicoes_iniciais(populacao_inicial=minha_populacao_inicial)

print("\nExecutando varredura...")
df_resultados = variador.executar(
    metodo='PSO',
    diretorio_saida='resultados_convergencia',
    salvar_json=False
)

# Exibir melhores resultados
print("\n✓ Top 5 melhores configurações:")
variador.exibir_resumo(top_n=5)

# --- EXEMPLO 4: Gráfico com Escala Logarítmica ---

print("\n" + "="*70)
print("TESTE 4: GRÁFICO COM ESCALA LOGARÍTMICA")
print("="*70)

viz_log = VisualizadorConvergencia(verbose=True,
                                   titulo_padrao="Convergência (Escala Log)")

# Executar uma otimização
print("\n→ Executando PSO para gráfico logarítmico...")
meu_otimizador.definir_parametros('PSO', c1=2.05, c2=2.05, w=0.4)
resultado_log = meu_otimizador.otimizar(
    metodo='PSO',
    solucao_inicial=minha_populacao_inicial,
    rastrear_convergencia=True
)

viz_log.adicionar_convergencia(
    resultado_log['historico_convergencia'],
    label='PSO'
)

# Plotar com escala logarítmica
print("\nGerando gráfico com escala logarítmica...")
viz_log.plotar(
    titulo="Convergência PSO (Escala Logarítmica)",
    xlabel="Iteração",
    ylabel="Melhor Fitness (escala log)",
    escala_y='log',
    salvar_em='grafico_log_convergencia.png'
)

# --- RESUMO FINAL ---

print("\n" + "="*70)
print("EXEMPLOS CONCLUÍDOS COM SUCESSO!")
print("="*70)
print("\n✓ Arquivos gerados:")
print("  - grafico_c1_convergencia.png")
print("  - grafico_algoritmos_convergencia.png")
print("  - grafico_log_convergencia.png")
print("  - resultados_convergencia/ (dados CSV e JSON)")
print("\n✓ O que foi demonstrado:")
print("  1. Rastreamento de convergência durante otimização")
print("  2. Comparação visual de diferentes parâmetros")
print("  3. Comparação de algoritmos")
print("  4. Análise automática de ponto de convergência")
print("  5. Gráficos com escala normal e logarítmica")
print("\n" + "="*70 + "\n")
