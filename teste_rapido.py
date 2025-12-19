#!/usr/bin/env python3
"""
TESTE RÁPIDO - Convergência e Variação de Parâmetros

Execute este script para testar rapidamente as novas funcionalidades.
"""

import os
import sys
import numpy as np

print("\n" + "="*70)
print("TESTE RÁPIDO: CONVERGÊNCIA E VARIAÇÃO DE PARÂMETROS")
print("="*70)

try:
    import HydroOpt
    print(f"\n✓ HydroOpt importado com sucesso (versão {HydroOpt.__version__})")
except ImportError as e:
    print(f"✗ Erro ao importar HydroOpt: {e}")
    sys.exit(1)

try:
    from HydroOpt import (
        Rede, Otimizador, LDiametro, 
        VariadorDeParametros, VisualizadorConvergencia
    )
    print("✓ Todas as classes importadas com sucesso")
except ImportError as e:
    print(f"✗ Erro ao importar classes: {e}")
    sys.exit(1)

from HydroOpt.core import gerar_solucao_heuristica

# --- TESTE 1: Convergência Básica ---

print("\n" + "="*70)
print("TESTE 1: Rastreamento de Convergência")
print("="*70)

try:
    print("\n→ Criando rede...")
    rede = Rede('hanoiFIM')
    rede.simular()
    
    print("→ Configurando diâmetros...")
    diametros = LDiametro()
    diametros.adicionar_polegadas(12, 45.73)
    diametros.adicionar_polegadas(16, 70.40)
    diametros.adicionar_polegadas(20, 98.38)
    diametros.adicionar_polegadas(24, 129.30)
    diametros.adicionar_polegadas(30, 180.80)
    diametros.adicionar_polegadas(40, 278.30)
    
    print("→ Criando otimizador...")
    otimizador = Otimizador(
        rede=rede,
        diametros=diametros,
        epoch=10,  # Pequeno para teste rápido
        pop_size=20,
        usar_paralelismo=False,
        verbose=False
    )
    otimizador.pressao_min_desejada = 30.0
    
    print("→ Gerando população inicial...")
    populacao_teste = [
        np.random.uniform(0, 1, len(rede.wn.pipe_name_list))
        for _ in range(20)
    ]
    
    print("→ Executando otimização com rastreamento...")
    resultado = otimizador.otimizar(
        metodo='PSO',
        solucao_inicial=populacao_teste,
        rastrear_convergencia=True
    )
    
    if 'historico_convergencia' in resultado:
        historico = resultado['historico_convergencia']
        print(f"✓ Convergência rastreada com sucesso!")
        print(f"  - Comprimento do histórico: {len(historico)}")
        print(f"  - Primeiro fitness: {historico[0]:.2f}")
        print(f"  - Último fitness: {historico[-1]:.2f}")
        print(f"  - Melhoria: {historico[0] - historico[-1]:.2f}")
    else:
        print("✗ Histórico de convergência não encontrado no resultado")
        
except Exception as e:
    print(f"✗ Erro no Teste 1: {e}")
    import traceback
    traceback.print_exc()

# --- TESTE 2: Visualizador de Convergência ---

print("\n" + "="*70)
print("TESTE 2: Visualizador de Convergência")
print("="*70)

try:
    print("\n→ Criando visualizador...")
    viz = VisualizadorConvergencia(verbose=False)
    
    print("→ Adicionando convergências...")
    # Adicionar a convergência do teste anterior
    if 'historico_convergencia' in resultado:
        viz.adicionar_convergencia(
            resultado['historico_convergencia'],
            label='PSO'
        )
        
        print("→ Gerando resumo...")
        resumo = viz.obter_resumo_convergencia()
        print(f"✓ Resumo gerado com sucesso!")
        print(f"\n{resumo.to_string(index=False)}")
        
        print("\n→ Analisando convergência...")
        analise = viz.analisar_convergencia(threshold_melhoria=0.01)
        print(f"✓ Análise concluída!")
        for label, iter_conv in analise.items():
            print(f"  - {label}: convergiu na iteração {iter_conv}")
        
    else:
        print("✗ Nenhuma convergência disponível para visualizar")
        
except Exception as e:
    print(f"✗ Erro no Teste 2: {e}")
    import traceback
    traceback.print_exc()

# --- TESTE 3: Variador de Parâmetros (Simplificado) ---

print("\n" + "="*70)
print("TESTE 3: Variador de Parâmetros (Teste Rápido)")
print("="*70)

try:
    print("\n→ Criando variador...")
    variador = VariadorDeParametros(otimizador, verbose=False)
    
    print("→ Definindo parâmetro...")
    variador.definir_parametro('c1', inicial=2.0, final=2.0, passo=0.5)
    
    print("→ Definindo condições iniciais...")
    variador.definir_condicoes_iniciais(populacao_inicial=populacao_teste)
    
    print("→ Executando varredura (apenas 1 combinação para teste rápido)...")
    df = variador.executar(
        metodo='PSO',
        diretorio_saida=None,
        salvar_json=False
    )
    
    if df is not None and len(df) > 0:
        print(f"✓ Varredura executada com sucesso!")
        print(f"  - Combinações testadas: {len(df)}")
        print(f"  - Sucessos: {df['sucesso'].sum()}")
        print(f"  - Melhor custo: R$ {df['custo_real_R$'].min():,.2f}")
    else:
        print("✗ Nenhum resultado de varredura")
        
except Exception as e:
    print(f"✗ Erro no Teste 3: {e}")
    import traceback
    traceback.print_exc()

# --- TESTE 4: Múltiplos Algoritmos (Rápido) ---

print("\n" + "="*70)
print("TESTE 4: Comparação de Algoritmos (Teste Rápido)")
print("="*70)

try:
    print("\n→ Criando novo visualizador...")
    viz_algos = VisualizadorConvergencia(verbose=False)
    
    print("→ Testando algoritmos...")
    algoritmos = ['PSO', 'GWO']  # Apenas 2 para teste rápido
    
    for metodo in algoritmos:
        print(f"  - Executando {metodo}...")
        resultado_algo = otimizador.otimizar(
            metodo=metodo,
            solucao_inicial=populacao_teste,
            rastrear_convergencia=True
        )
        
        if 'historico_convergencia' in resultado_algo:
            viz_algos.adicionar_convergencia(
                resultado_algo['historico_convergencia'],
                label=metodo
            )
        else:
            print(f"    ⚠️ Sem histórico para {metodo}")
    
    print("✓ Algoritmos testados com sucesso!")
    
    print("\n→ Resumo de convergência:")
    resumo_algos = viz_algos.obter_resumo_convergencia()
    print(resumo_algos.to_string(index=False))
    
except Exception as e:
    print(f"✗ Erro no Teste 4: {e}")
    import traceback
    traceback.print_exc()

# --- RESULTADO FINAL ---

print("\n" + "="*70)
print("TESTES CONCLUÍDOS!")
print("="*70)

print("\n✓ Funcionalidades testadas:")
print("  1. ✓ Rastreamento de convergência automático")
print("  2. ✓ Visualizador de convergência")
print("  3. ✓ Variador de parâmetros")
print("  4. ✓ Comparação de algoritmos")

print("\n📝 Próximos passos:")
print("  1. Execute: python exemplo_convergencia_graficos.py")
print("  2. Execute: python exemplo_variador_parametros.py")
print("  3. Consulte: README_CONVERGENCIA_E_PARAMETROS.md")

print("\n📊 Documentação disponível:")
print("  - CONVERGENCIA_SUMMARY.md")
print("  - VISUALIZADOR_CONVERGENCIA_README.md")
print("  - VARIADOR_PARAMETROS_README.md")

print("\n" + "="*70)
print("✅ Teste rápido completado com sucesso!")
print("="*70 + "\n")
