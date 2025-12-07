"""
Teste do método Grey Wolf Optimizer (GWO)
Simulação com 3 épocas para otimização de rede hidráulica
"""

from HydroOpt import Rede, Otimizador, LDiametro


def teste_gwo_3_epocas():
    """
    Executa teste do GWO com 50 épocas.
    """
    print("\n" + "="*80)
    print("TESTE: GREY WOLF OPTIMIZER (GWO) - 10 ÉPOCAS")
    print("="*80)
    
    # 1. Criar rede
    print("\n[1] Criando rede...")
    rede = Rede(arquivo_inp='hanoiFIM.inp')
    print(f"   ✓ Rede '{rede.nome}' carregada")
    print(f"   Nós: {len(rede.wn.junction_name_list)}")
    print(f"   Tubos: {len(rede.wn.pipe_name_list)}")
    
    # 2. Simular estado inicial
    print("\n[2] Simulando estado inicial...")
    rede.simular()
    pressao_inicial = rede.obter_pressao_minima(excluir_reservatorios=True)
    print(f"   Pressão mínima inicial: {pressao_inicial['valor']:.2f} m")
    print(f"   No: {pressao_inicial['no']}")
    
    # 3. Configurar lista de diâmetros
    print("\n[3] Configurando diâmetros disponíveis...")
    lista_diametros = LDiametro()
    # Diâmetros disponíveis para a rede Hanoi (baseado no problema clássico)
    # Fonte: problema de otimização clássico de Hanoi
    # Usando polegadas (padrão americano) com custos por metro
    lista_diametros.adicionar_polegadas(12, 45.726)   # 12"
    lista_diametros.adicionar_polegadas(16, 70.406)   # 16"
    lista_diametros.adicionar_polegadas(20, 98.378)   # 20"
    lista_diametros.adicionar_polegadas(24, 129.333)  # 24"
    lista_diametros.adicionar_polegadas(30, 180.748)  # 30"
    lista_diametros.adicionar_polegadas(40, 278.280)  # 40"
    
    print(f"   ✓ {len(lista_diametros.obter_diametros())} diâmetros disponíveis")
    diams = lista_diametros.obter_diametros()
    print(f"   Faixa: {min(diams)*1000:.0f}mm a {max(diams)*1000:.0f}mm")
    
    # 4. Criar otimizador
    print("\n[4] Criando otimizador GWO...")
    otimizador = Otimizador(
        rede=rede,
        diametros=lista_diametros,
        epoch=10,
        pop_size=20
    )
    print(f"   Método: GWO (Grey Wolf Optimizer)")
    print(f"   Épocas: 10")
    print(f"   População: 20")
    
    # 5. Executar otimização
    print("\n[5] Executando otimização...")
    print("   (Aguarde, processando 10 épocas...)\n")
    
    resultado = otimizador.otimizar(metodo='GWO')
    
    # 6. Exibir resultados
    print("\n[6] RESULTADOS:")
    print("   " + "="*76)
    
    melhor_custo = resultado['melhor_custo']
    melhor_solucao = resultado['melhor_solucao']
    historico = resultado['historico']
    
    print(f"\n   📈 CUSTO FINAL")
    if melhor_custo == 0.0:
        print(f"      {melhor_custo:.6f}  ✓ Ótimo!")
    else:
        print(f"      {melhor_custo:.6f}")
    
    print(f"\n   📍 Melhor solução encontrada:")
    print(f"      {melhor_solucao}")
    
    print(f"\n   📊 Histórico de custos por época:")
    for i, custo in enumerate(historico, 1):
        print(f"      Época {i}: {custo:.6f}")
    
    # 7. Análise da progressão
    print(f"\n   📉 Progressão:")
    if len(historico) >= 2:
        melhora_total = historico[0] - historico[-1]
        percentual = (melhora_total / max(historico[0], 0.0001)) * 100
        
        print(f"      Custo inicial: {historico[0]:.6f}")
        print(f"      Custo final:   {historico[-1]:.6f}")
        print(f"      Melhora:       {melhora_total:.6f} ({percentual:.2f}%)")
        
        if melhora_total > 0:
            print(f"      ✓ Otimização bem-sucedida!")
        elif melhora_total < 0:
            print(f"      ⚠️  Custo piorou")
        else:
            print(f"      ➡️  Sem mudanças")
    
    print("\n" + "="*80)
    print("✓ Teste concluído!")
    print("="*80 + "\n")
    
    return resultado


if __name__ == "__main__":
    teste_gwo_3_epocas()
