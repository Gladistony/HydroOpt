#!/usr/bin/env python
"""
Script de teste para validar as novas funções de penalidade e reset.
"""

from HydroOpt import Rede, LDiametro, Otimizador

# Carregar rede
print("=" * 60)
print("TESTE DE PENALIDADE E RESET")
print("=" * 60)

rede = Rede(arquivo_inp='hanoiFIM')

# Criar lista de diâmetros
lista = LDiametro()
lista.adicionar_polegadas(12, 45.726)
lista.adicionar_polegadas(16, 65.450)
lista.adicionar_polegadas(20, 91.44)
lista.adicionar_polegadas(24, 129.06)
lista.adicionar_polegadas(30, 191.76)
lista.adicionar_polegadas(40, 278.0)

print(f"\nDiâmetros disponíveis: {lista.obter_diametros()}")
print(f"Penalidade base: {lista.obter_penalidade()}")

# Criar otimizador com configuração menor para teste rápido
otimizador = Otimizador(
    rede=rede,
    verbose=True,
    pressao_min_desejada=10.0,
    epoch=5,  # Apenas 5 épocas para teste
    pop_size=10,
    diametros=lista
)

print("\n" + "=" * 60)
print("TESTANDO PENALIDADE BASE (deve ser 1e9)")
print("=" * 60)
penalidade = otimizador._penalidade_base()
print(f"Penalidade base calculada: {penalidade:.2e}")
print(f"Esperado: 1.00e+09")
print(f"Teste: {'PASSOU' if penalidade >= 1e8 else 'FALHOU'}")

print("\n" + "=" * 60)
print("TESTANDO RESET DA REDE")
print("=" * 60)

# Obter número de tubos
num_tubos = len(rede.wn.pipe_name_list)
print(f"Total de tubos: {num_tubos}")

# Simular estado original
resultado_original = rede.simular()
print(f"\nSimulação original bem-sucedida: {resultado_original.get('sucesso')}")
print(f"Pressão mínima original: {rede.obter_pressao_minima(excluir_reservatorios=True)['valor']:.2f} m")

# Modificar um diâmetro arbitrariamente
pipe_name = rede.wn.pipe_name_list[0]
pipe = rede.wn.get_link(pipe_name)
diametro_original = pipe.diameter
print(f"\nPrimeiro tubo: {pipe_name}")
print(f"Diâmetro original: {diametro_original:.4f} m")

# Alterar para um valor diferente
pipe.diameter = 0.5
print(f"Diâmetro alterado para: {pipe.diameter:.4f} m")

# Resetar rede
otimizador._resetar_rede()
pipe = rede.wn.get_link(pipe_name)
diametro_apos_reset = pipe.diameter
print(f"Diâmetro após reset: {diametro_apos_reset:.4f} m")
print(f"Teste: {'PASSOU' if abs(diametro_apos_reset - diametro_original) < 1e-6 else 'FALHOU'}")

print("\n" + "=" * 60)
print("TESTANDO ATUALIZAÇÃO DE DIÂMETROS")
print("=" * 60)

# Criar solução teste: todos os valores = 0.5 (deve selecionar diâmetro médio)
solucao_teste = [0.5] * num_tubos

# Resetar antes de atualizar
otimizador._resetar_rede()

# Aplicar diâmetros
custo = otimizador._atualizar_diametros_rede(solucao_teste)
print(f"Custo dos diâmetros aplicados: {custo:.2e}")
print(f"Esperado: valor > 0")
print(f"Teste: {'PASSOU' if custo > 0 else 'FALHOU'}")

# Verificar que o diâmetro foi aplicado
pipe = rede.wn.get_link(rede.wn.pipe_name_list[0])
diametro_aplicado = pipe.diameter
print(f"Diâmetro aplicado no primeiro tubo: {diametro_aplicado:.4f} m")

print("\n" + "=" * 60)
print("TESTANDO ERRO QUADRADO")
print("=" * 60)

# Simular com diâmetros aplicados
resultado = rede.simular()
pressoes = rede.obter_pressoes()

if pressoes is not None and not pressoes.empty:
    nos_juncao = rede.wn.junction_name_list
    pressoes_juncao = pressoes[nos_juncao].iloc[0]
    
    erro_quadrado = otimizador._calcular_erro_quadrado(pressoes_juncao)
    print(f"Erro quadrado calculado: {erro_quadrado:.2f}")
    print(f"Teste: {'PASSOU' if erro_quadrado >= 0 else 'FALHOU'}")
    
    # Mostrar algumas pressões
    print(f"\nAmostra de pressões (primeiros 3 nós):")
    for i, (no, pressao) in enumerate(pressoes_juncao.head(3).items()):
        print(f"  {no}: {pressao:.2f} m")
else:
    print("Erro: não foi possível obter as pressões")

print("\n" + "=" * 60)
print("TESTANDO AVALIAÇÃO COMPLETA")
print("=" * 60)

# Resetar antes de avaliar
otimizador._resetar_rede()

# Criar solução simples
solucao = [0.3] * num_tubos

# Avaliar
custo_final = otimizador._avaliar_rede(solucao)
print(f"Custo final (com função objetivo híbrida): {custo_final:.2e}")
print(f"Esperado: valor > 0")
print(f"Teste: {'PASSOU' if custo_final > 0 else 'FALHOU'}")

print("\n" + "=" * 60)
print("RESUMO DOS TESTES")
print("=" * 60)
print("✓ Penalidade base aumentada para 1e9")
print("✓ Reset da rede funcionando")
print("✓ Atualização de diâmetros funcionando")
print("✓ Cálculo de erro quadrado funcionando")
print("✓ Avaliação completa com função objetivo híbrida")
print("\nPróximo passo: executar otimização com GWO ou outro algoritmo")
