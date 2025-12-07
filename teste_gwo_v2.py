#!/usr/bin/env python
"""
Script de otimização com a nova penalidade e função objetivo híbrida.
Testa GWO com parâmetros melhorados.
"""

from HydroOpt import Rede, LDiametro, Otimizador

# Carregar rede
print("\n" + "="*70)
print("OTIMIZAÇÃO COM GWO - NOVA PENALIDADE E FUNÇÃO OBJETIVO HÍBRIDA")
print("="*70)

rede = Rede(arquivo_inp='hanoiFIM')

# Criar lista de diâmetros com custos realistas
lista = LDiametro()
lista.adicionar_polegadas(12, 45.726)
lista.adicionar_polegadas(16, 65.450)
lista.adicionar_polegadas(20, 91.44)
lista.adicionar_polegadas(24, 129.06)
lista.adicionar_polegadas(30, 191.76)
lista.adicionar_polegadas(40, 278.0)

print(f"\nDiâmetros configurados: 6")
print(f"Penalidade base: {lista.obter_penalidade() * 1e6:.2e}")

# Criar otimizador
otimizador = Otimizador(
    rede=rede,
    verbose=True,
    pressao_min_desejada=10.0,
    epoch=10,  # 10 épocas
    pop_size=20,  # População de 20
    diametros=lista
)

print("\n" + "="*70)
print("EXECUTANDO OTIMIZAÇÃO COM GWO (10 ÉPOCAS, 20 INDIVÍDUOS)")
print("="*70)

# Executar otimização
resultado = otimizador.otimizar(metodo='GWO')

# Exibir resultados
print("\n" + "="*70)
print("RESULTADOS DA OTIMIZAÇÃO")
print("="*70)
print(f"Melhor custo encontrado: {resultado['melhor_custo']:.6e}")
print(f"Número de variáveis (tubos): {len(resultado['melhor_solucao'])}")

# Aplicar melhor solução
print("\n" + "="*70)
print("SIMULANDO COM A MELHOR SOLUÇÃO ENCONTRADA")
print("="*70)

otimizador._resetar_rede()
custo_diams = otimizador._atualizar_diametros_rede(resultado['melhor_solucao'])
resultado_sim = rede.simular()

if resultado_sim.get('sucesso'):
    pressao_info = rede.obter_pressao_minima(excluir_reservatorios=True)
    pressao_min = pressao_info['valor']
    
    print(f"\n✓ Simulação bem-sucedida!")
    print(f"  Custo dos diâmetros: {custo_diams:.6e}")
    print(f"  Pressão mínima: {pressao_min:.2f} m")
    print(f"  Pressão desejada: 10.0 m")
    print(f"  Requisito atendido: {'SIM ✓' if pressao_min >= 10.0 else 'NÃO ✗'}")
    
    # Mostrar diâmetros escolhidos
    print(f"\nDiâmetros escolhidos (primeiros 10 tubos):")
    diametros_disponiveis = lista.obter_diametros()
    for i, (pipe_name, valor_solucao) in enumerate(zip(rede.wn.pipe_name_list[:10], resultado['melhor_solucao'][:10])):
        idx = int(valor_solucao * (len(diametros_disponiveis) - 1))
        idx = min(max(0, idx), len(diametros_disponiveis) - 1)
        diametro = diametros_disponiveis[idx]
        pipe = rede.wn.get_link(pipe_name)
        comprimento = pipe.length
        custo_tubo = lista.obter_valor(diametro) * comprimento
        
        # Converter para polegadas para exibição
        polegadas = diametro / 0.0254
        print(f"    {pipe_name:8s}: {diametro:.4f}m ({polegadas:.1f}\") | L={comprimento:.0f}m | Custo={custo_tubo:.0f}")
else:
    print(f"✗ Simulação falhou!")

print("\n" + "="*70)
print("RESUMO")
print("="*70)
print("""
A otimização agora usa:
1. Penalidade MUITO MAIOR (1e9 em vez de 1e5) para forçar restrições
2. Função objetivo híbrida: 60% custo + 40% erro quadrado
3. Reset automático da rede a cada avaliação
4. Erro quadrado penaliza distância da pressão desejada

Isto resulta em:
- Soluções que respeitam a pressão mínima de forma agressiva
- Otimização equilibrada entre custo e qualidade
- Convergência mais rápida para soluções viáveis
""")
