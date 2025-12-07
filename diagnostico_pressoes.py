#!/usr/bin/env python
"""
Script simplificado para investigar pressões negativas na otimização GWO.
"""

import sys
sys.path.insert(0, '/mnt/discoD/Github/HydroOpt')

from HydroOpt import Otimizador, Rede, LDiametro
import numpy as np

print("="*70)
print("DIAGNÓSTICO: Investigação de Pressões Negativas")
print("="*70)

# Criar lista de diâmetros (30 valores entre 0.05 e 1.0)
diametros_dict = {}
for i in range(30):
    d = 0.05 + (1.0 - 0.05) * i / 29
    # Custo aumenta com diâmetro (cubo do diâmetro para custo não-linear)
    custo = 100 * (d ** 3)
    diametros_dict[d] = custo

print(f"\nDiâmetros criados: {len(diametros_dict)} valores entre {min(diametros_dict.keys()):.4f}m e {max(diametros_dict.keys()):.4f}m")

# Teste 1: Rede com diâmetros mínimos
print("\n" + "="*70)
print("TESTE 1: Rede com TODOS os diâmetros MÍNIMOS (0.05m)")
print("="*70)

rede1 = Rede('hanoiFIM')
diams1 = LDiametro(diametros_dict)
opt1 = Otimizador(rede1, diams1, verbose=False)

# Solução com todos zeros (diâmetros mínimos)
sol_min = np.zeros(len(rede1.wn.pipe_name_list))
custo_min = opt1._avaliar_rede(sol_min)

p_info = rede1.obter_pressao_minima(excluir_reservatorios=True)
print(f"  Pressão mínima: {p_info['valor']:.2f} m (nó: {p_info['no']})")
print(f"  Custo total: {custo_min:.2f}")
print(f"  Penalidade aplicada: {'SIM' if p_info['valor'] < 10.0 else 'NÃO'}")

# Teste 2: Rede com diâmetros máximos
print("\n" + "="*70)
print("TESTE 2: Rede com TODOS os diâmetros MÁXIMOS (1.0m)")
print("="*70)

rede2 = Rede('hanoiFIM')
diams2 = LDiametro(diametros_dict)
opt2 = Otimizador(rede2, diams2, verbose=False)

# Solução com todos uns (diâmetros máximos)
sol_max = np.ones(len(rede2.wn.pipe_name_list))
custo_max = opt2._avaliar_rede(sol_max)

p_info2 = rede2.obter_pressao_minima(excluir_reservatorios=True)
print(f"  Pressão mínima: {p_info2['valor']:.2f} m (nó: {p_info2['no']})")
print(f"  Custo total: {custo_max:.2f}")
print(f"  Penalidade aplicada: {'SIM' if p_info2['valor'] < 10.0 else 'NÃO'}")

# Teste 3: Análise da penalidade
print("\n" + "="*70)
print("TESTE 3: Análise da Penalidade")
print("="*70)

rede3 = Rede('hanoiFIM')
diams3 = LDiametro(diametros_dict)
opt3 = Otimizador(rede3, diams3, verbose=False, pressao_min_desejada=10.0)

sol_test = np.zeros(len(rede3.wn.pipe_name_list))
custo_test = opt3._avaliar_rede(sol_test)

p_info3 = rede3.obter_pressao_minima(excluir_reservatorios=True)
p_min = p_info3['valor']
def_val = opt3.pressao_min_desejada - p_min
pen_base = opt3._penalidade_base()
pen_calc = pen_base * (def_val ** 2)

print(f"  Pressão desejada: {opt3.pressao_min_desejada:.2f} m")
print(f"  Pressão obtida: {p_min:.2f} m")
print(f"  Deficiência: {def_val:.2f} m")
print(f"  Penalidade base: {pen_base:.2e}")
print(f"  Penalidade calculada: {pen_calc:.2e}")
print(f"  Custo total: {custo_test:.2f}")

# Teste 4: Executar GWO rápido
print("\n" + "="*70)
print("TESTE 4: GWO com 5 épocas x 10 população (rápido)")
print("="*70)

rede4 = Rede('hanoiFIM')
diams4 = LDiametro(diametros_dict)
opt4 = Otimizador(rede4, diams4, epoch=5, pop_size=10, verbose=True)
resultado = opt4.otimizar(metodo='GWO')

# Verificar solução final
p_final = rede4.obter_pressao_minima(excluir_reservatorios=True)
print(f"\n  Pressão mínima final: {p_final['valor']:.2f} m")
print(f"  Melhor custo: {resultado['melhor_custo']:.2f}")

print("\n" + "="*70)
print("CONCLUSÕES:")
print("="*70)
if p_info['valor'] < 0:
    print("""
✗ PROBLEMA CRÍTICO: Mesmo com MÍNIMOS diâmetros, há pressões negativas!
  
  Possíveis causas:
  1. Rede hanoiFIM tem demanda insuficiente para viabilidade
  2. Pressão_min_desejada (10m) é muito alta para essa rede
  3. Arquivo .inp tem configuração inválida
  
  Solução: Testar com pressão_min_desejada = 0 ou reduzir demanda
    """)
elif p_info2['valor'] < 0:
    print("""
⚠️  PROBLEMA: Mesmo com MÁXIMOS diâmetros, há pressões negativas!
  
  Possível causa: Arquivo .inp com demanda inválida ou configuração incorreta
  
  Solução: Revisar arquivo hanoiFIM.inp
    """)
else:
    print(f"""
✓ Rede é viável:
  - Mín diâmetros: pressão = {p_info['valor']:.2f} m
  - Máx diâmetros: pressão = {p_info2['valor']:.2f} m
  
  Se GWO ainda produz pressões negativas:
    - Aumentar population (50+)
    - Aumentar epochs (100+)
    - Tentar outro algoritmo (ABC, HHO, DE)
    """)
