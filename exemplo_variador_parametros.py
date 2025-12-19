"""
Exemplo de uso da classe VariadorDeParametros

Demonstra como fazer varredura de parâmetros para encontrar as melhores
configurações de otimização para uma rede hidráulica.
"""

import HydroOpt
print(f"Versão HydroOpt: {HydroOpt.__version__}")

from HydroOpt import Rede, Otimizador, LDiametro, VariadorDeParametros
from HydroOpt.core import gerar_solucao_heuristica
import numpy as np

# --- CONFIGURAÇÃO INICIAL ---

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

# 3. Criar otimizador
populacao_tamanho = 50  # Reduzido para exemplo mais rápido
pressao_alvo = 30.0

meu_otimizador = Otimizador(
    rede=minha_rede,
    diametros=lista_diametros,
    epoch=50,  # Reduzido para exemplo
    pop_size=populacao_tamanho,
    usar_paralelismo=True
)

meu_otimizador.pressao_min_desejada = pressao_alvo
meu_otimizador._exibir_configuracao()

# 4. Gerar população inicial (solução heurística + aleatórios)
print("\nGerando população inicial...")
solucao_guia = gerar_solucao_heuristica(minha_rede, lista_diametros, pressao_min_desejada=pressao_alvo)
num_tubos = len(minha_rede.wn.pipe_name_list)
qtd_aleatorios = populacao_tamanho - 1
populacao_aleatoria = np.random.uniform(0.0, 1.0, (qtd_aleatorios, num_tubos)).tolist()
minha_populacao_inicial = [solucao_guia] + populacao_aleatoria

# --- INÍCIO DA VARREDURA DE PARÂMETROS ---

print("\n" + "="*70)
print("EXEMPLO: VARREDURA DE PARÂMETROS PSO")
print("="*70)

# Criar variador de parâmetros
variador = VariadorDeParametros(meu_otimizador, verbose=True)

# Definir ranges para os parâmetros que queremos variar
# Variando c1 e c2 (coeficientes cognitivo e social do PSO)
variador.definir_parametro('c1', inicial=1.5, final=2.5, passo=0.5)
variador.definir_parametro('c2', inicial=1.5, final=2.5, passo=0.5)

# Manter w (inércia) constante
meu_otimizador.definir_parametros('PSO', w=0.4)

# Definir condições iniciais para todas as otimizações
variador.definir_condicoes_iniciais(
    populacao_inicial=minha_populacao_inicial,
    verbose_otimizacao=False  # Não exibir output de cada otimização
)

# Executar varredura com salvamento de resultados
print("\nExecutando varredura...")
resultado_df = variador.executar(
    metodo='PSO',
    diretorio_saida='resultados_pso',
    salvar_json=True
)

# --- ANÁLISE DE RESULTADOS ---

print("\n" + "="*70)
print("ANÁLISE DE RESULTADOS")
print("="*70)

# Exibir top 5 melhores resultados
variador.exibir_resumo(top_n=5)

# Obter melhor resultado
melhor = variador.obter_melhor_resultado()
print("\n✓ MELHOR CONFIGURAÇÃO ENCONTRADA:")
print(f"   Parâmetros: {melhor['parametros']}")
print(f"   Custo Real: R$ {melhor['custo_real']:,.2f}")
print(f"   Fitness: {melhor['fitness']:.6f}")
print(f"   Pressão Mínima: {melhor['pressao_minima']:.2f} m")

# Análise univariada (impacto de cada parâmetro)
print("\n" + "="*70)
print("ANÁLISE UNIVARIADA")
print("="*70)

print("\nImpacto do parâmetro 'c1':")
variador.exibir_comparacao('c1')

print("\nImpacto do parâmetro 'c2':")
variador.exibir_comparacao('c2')

# Análise bivariada (interação entre parâmetros)
print("\n" + "="*70)
print("ANÁLISE BIVARIADA: c1 vs c2")
print("="*70)
variador.exibir_comparacao('c1', 'c2')

# Informações gerais
info = variador.obter_informacoes()
print("\n" + "="*70)
print("INFORMAÇÕES GERAIS DA VARREDURA")
print("="*70)
print(f"Total de combinações testadas: {info['total_combinacoes']}")
print(f"Sucessos: {info['sucessos']}")
print(f"Falhas: {info['falhas']}")
print(f"Melhor custo encontrado: R$ {info['melhor_custo']:,.2f}")
print(f"Pior custo encontrado: R$ {info['pior_custo']:,.2f}")
print(f"Custo médio: R$ {info['custo_medio']:,.2f}")
print(f"Timestamp: {info['timestamp']}")
print("="*70 + "\n")

# --- EXEMPLO AVANÇADO: VARREDURA COM MÚLTIPLOS PARÂMETROS ---

print("\n" + "="*70)
print("EXEMPLO AVANÇADO: VARREDURA COM MAIS PARÂMETROS")
print("="*70)

# Criar novo variador para WOA (Whale Optimization Algorithm)
variador_woa = VariadorDeParametros(meu_otimizador, verbose=True)

# WOA tem apenas um parâmetro ajustável: 'b' (controla o raio de busca)
variador_woa.definir_parametro('b', inicial=0.5, final=2.0, passo=0.5)

variador_woa.definir_condicoes_iniciais(
    populacao_inicial=minha_populacao_inicial,
    verbose_otimizacao=False
)

print("\nExecutando varredura WOA...")
resultado_woa = variador_woa.executar(
    metodo='WOA',
    diretorio_saida='resultados_woa',
    salvar_json=True
)

# Comparar resultados
melhor_pso = variador.obter_melhor_resultado()
melhor_woa = variador_woa.obter_melhor_resultado()

print("\n" + "="*70)
print("COMPARAÇÃO PSO vs WOA")
print("="*70)
print(f"\nMelhor PSO:")
print(f"  Parâmetros: {melhor_pso['parametros']}")
print(f"  Custo: R$ {melhor_pso['custo_real']:,.2f}")
print(f"  Pressão: {melhor_pso['pressao_minima']:.2f} m")

print(f"\nMelhor WOA:")
print(f"  Parâmetros: {melhor_woa['parametros']}")
print(f"  Custo: R$ {melhor_woa['custo_real']:,.2f}")
print(f"  Pressão: {melhor_woa['pressao_minima']:.2f} m")

economia = melhor_woa['custo_real'] - melhor_pso['custo_real']
print(f"\nDiferença: R$ {economia:,.2f} ({economia/melhor_woa['custo_real']*100:.1f}%)")
melhor_metodo = "PSO" if economia > 0 else "WOA"
print(f"Melhor método: {melhor_metodo}")
print("="*70 + "\n")

print("✓ Exemplos concluídos com sucesso!")
print("  Os resultados foram salvos em 'resultados_pso' e 'resultados_woa'")
