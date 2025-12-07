"""
Exemplo de uso da biblioteca HydroOpt
Demonstra como:
1. Criar uma rede hidráulica
2. Simular a rede
3. Obter informações de status
4. Otimizar com diferentes algoritmos
"""

from HydroOpt import Rede, LDiametro, Otimizador


def exemplo_1_criar_rede_simples():
    """Exemplo 1: Criar uma rede simples de teste"""
    print("\n" + "="*70)
    print("EXEMPLO 1: Criar uma Rede Simples")
    print("="*70)
    
    # Criar rede (sem arquivo .inp, gera uma de teste)
    rede = Rede()
    
    # Acessar informações da rede
    print(f"\nNome da rede: {rede.nome}")
    print(f"Total de nós: {len(rede.wn.junction_name_list)}")
    print(f"Total de tubos: {len(rede.wn.pipe_name_list)}")


def exemplo_2_simular_rede():
    """Exemplo 2: Simular a rede e obter resultados"""
    print("\n" + "="*70)
    print("EXEMPLO 2: Simular a Rede e Obter Resultados")
    print("="*70)
    
    rede = Rede()
    
    # Executar simulação
    resultado = rede.simular()
    
    # Verificar se simulação foi bem-sucedida
    if resultado['sucesso']:
        print(f"\n✓ Simulação bem-sucedida!")
        print(f"  Pressão mínima: {resultado['pressao_minima']:.2f} m")
        print(f"  Pressão máxima: {resultado['pressao_maxima']:.2f} m")
        print(f"  Pressão média: {resultado['pressao_media']:.2f} m")
        print(f"  Vazão média: {resultado['vazao_media']:.4f} m³/s")
        print(f"  Nós com pressão baixa (<20m): {resultado['nos_com_pressao_baixa']}")
    else:
        print(f"\n✗ Erro na simulação: {resultado['erro']}")


def exemplo_3_obter_pressoes():
    """Exemplo 3: Obter informações detalhadas de pressão"""
    print("\n" + "="*70)
    print("EXEMPLO 3: Obter Informações de Pressão")
    print("="*70)
    
    rede = Rede()
    rede.simular()
    
    # Obter pressão mínima com detalhes
    info_pressao = rede.obter_pressao_minima(excluir_reservatorios=True)
    
    print(f"\nPressão mínima encontrada:")
    print(f"  Valor: {info_pressao['valor']:.2f} m")
    print(f"  Nó: {info_pressao['no']}")
    print(f"  Tempo: {info_pressao['tempo']}")
    
    # Obter matriz de pressões
    pressoes = rede.obter_pressoes()
    print(f"\nMatriz de pressões:")
    print(f"  Dimensão: {pressoes.shape}")
    print(f"  Nós: {list(pressoes.columns)}")


def exemplo_4_criar_lista_diametros():
    """Exemplo 4: Criar lista de diâmetros comerciais"""
    print("\n" + "="*70)
    print("EXEMPLO 4: Criar Lista de Diâmetros")
    print("="*70)
    
    # Opção 1: Usar diâmetros padrão (recomendado)
    print("\nOpção 1: Usar diâmetros padrão")
    diametros = LDiametro.criar_padrao()
    print(f"  Total de diâmetros: {len(diametros)}")
    print(f"  Diâmetros disponíveis: {diametros.obter_diametros()}")
    
    # Opção 2: Criar do zero com valores em milímetros
    print("\nOpção 2: Criar a partir de milímetros")
    diametros_mm = LDiametro.criar_de_mm({
        50: 20.0,    # 50mm custa 20
        100: 50.0,   # 100mm custa 50
        150: 75.0,   # 150mm custa 75
        200: 100.0,  # 200mm custa 100
    })
    print(f"  Diâmetros criados: {len(diametros_mm)}")
    print(diametros_mm)
    
    # Opção 3: Criar vazio e adicionar manualmente
    print("\nOpção 3: Criar vazio e adicionar")
    diametros_custom = LDiametro()
    diametros_custom.adicionar(0.1, 50.0)
    diametros_custom.adicionar(0.15, 75.0)
    diametros_custom.adicionar(0.2, 100.0)
    print(f"  Diâmetros adicionados: {len(diametros_custom)}")


def exemplo_5_otimizar_rede():
    """Exemplo 5: Otimizar a rede com diferentes algoritmos"""
    print("\n" + "="*70)
    print("EXEMPLO 5: Otimizar a Rede")
    print("="*70)
    
    # Criar rede e diâmetros
    rede = Rede()
    diametros = LDiametro.criar_padrao()
    
    # Criar otimizador
    # Parâmetros:
    # - rede: instância da classe Rede
    # - usar_gpu: None (detecta), True (força GPU), False (força CPU)
    # - verbose: True (mostra informações), False (silencioso)
    # - pressao_min_desejada: pressão mínima em metros (padrão 10.0)
    # - epoch: número de épocas (gerações)
    # - pop_size: tamanho da população
    # - diametros: lista de diâmetros (opcional)
    # - usar_paralelismo: True (usa múltiplos núcleos)
    # - n_workers: número de workers (None = automático)
    
    otimizador = Otimizador(
        rede=rede,
        usar_gpu=False,  # Forçar CPU para este exemplo
        verbose=True,
        pressao_min_desejada=10.0,
        epoch=5,  # Poucas épocas para teste rápido
        pop_size=10,  # Pequena população para teste rápido
        diametros=diametros,
        usar_paralelismo=True,
        n_workers=None  # Automático (cpu_count - 1)
    )
    
    # Verificar status da GPU
    status_gpu = otimizador.obter_status_gpu()
    print(f"\nStatus da GPU:")
    print(f"  Disponível: {status_gpu['disponivel']}")
    print(f"  Em uso: {status_gpu['em_uso']}")
    
    # Listar métodos disponíveis
    metodos = otimizador.listar_metodos()
    print(f"\nMétodos de otimização disponíveis ({len(metodos)}):")
    for metodo in metodos:
        print(f"  - {metodo}")
    
    # Executar otimização com PSO
    print(f"\nIniciando otimização com PSO...")
    resultado = otimizador.otimizar(metodo='PSO')
    
    print(f"\nResultados da otimização:")
    print(f"  Melhor custo: {resultado['melhor_custo']:.6f}")
    print(f"  Melhor solução: {resultado['melhor_solucao']}")


def exemplo_6_gerenciar_parametros():
    """Exemplo 6: Gerenciar parâmetros dos algoritmos"""
    print("\n" + "="*70)
    print("EXEMPLO 6: Gerenciar Parâmetros dos Algoritmos")
    print("="*70)
    
    rede = Rede()
    otimizador = Otimizador(rede=rede, verbose=True, epoch=10, pop_size=20)
    
    # Obter parâmetros atuais do PSO
    print("\nParâmetros padrão do PSO:")
    params_pso = otimizador.obter_parametros('PSO')
    print(f"  {params_pso}")
    
    # Alterar parâmetros
    print("\nAlterando parâmetros do PSO...")
    otimizador.definir_parametros('PSO', c1=2.0, c2=2.0, w=0.7)
    
    params_pso_novo = otimizador.obter_parametros('PSO')
    print(f"  Novos parâmetros: {params_pso_novo}")
    
    # Restaurar parâmetros padrão
    print("\nRestaurando parâmetros padrão...")
    otimizador.resetar_parametros('PSO')
    print(f"  Restaurados: {otimizador.obter_parametros('PSO')}")


def exemplo_7_controlar_gpu():
    """Exemplo 7: Controlar GPU"""
    print("\n" + "="*70)
    print("EXEMPLO 7: Controlar GPU")
    print("="*70)
    
    rede = Rede()
    otimizador = Otimizador(rede=rede, verbose=True)
    
    print(f"\nStatus inicial:")
    print(f"  GPU disponível: {otimizador.gpu_disponivel}")
    print(f"  GPU em uso: {otimizador.usar_gpu}")
    
    # Tentar ativar GPU
    print(f"\nTentando ativar GPU...")
    sucesso = otimizador.ativar_gpu()
    
    if sucesso:
        print(f"  GPU ativada!")
    else:
        print(f"  GPU não disponível, continuando com CPU")
    
    # Desativar GPU
    print(f"\nDesativando GPU (forçar CPU)...")
    otimizador.desativar_gpu()
    
    # Alternar GPU
    print(f"\nAlternando GPU...")
    estado = otimizador.alternar_gpu()
    print(f"  GPU {'ativada' if estado else 'desativada'}")


def exemplo_8_informacoes_completas():
    """Exemplo 8: Obter informações completas do otimizador"""
    print("\n" + "="*70)
    print("EXEMPLO 8: Informações Completas do Otimizador")
    print("="*70)
    
    rede = Rede()
    otimizador = Otimizador(
        rede=rede,
        verbose=True,
        pressao_min_desejada=15.0,
        epoch=50,
        pop_size=30,
        usar_paralelismo=True
    )
    
    info = otimizador.obter_informacoes()
    
    print("\nInformações do Otimizador:")
    for chave, valor in info.items():
        print(f"  {chave}: {valor}")


def main():
    """Executar todos os exemplos"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " EXEMPLOS DE USO DA BIBLIOTECA HYDROOPT ".center(68) + "║")
    print("╚" + "="*68 + "╝")
    
    try:
        exemplo_1_criar_rede_simples()
        exemplo_2_simular_rede()
        exemplo_3_obter_pressoes()
        exemplo_4_criar_lista_diametros()
        exemplo_6_gerenciar_parametros()
        exemplo_7_controlar_gpu()
        exemplo_8_informacoes_completas()
        
        # O exemplo 5 leva mais tempo, descomente se quiser executar
        # exemplo_5_otimizar_rede()
        
        print("\n" + "="*70)
        print("✓ Todos os exemplos executados com sucesso!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
