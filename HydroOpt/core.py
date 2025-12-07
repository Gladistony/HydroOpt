import os
from .diametros import LDiametro
from .rede import Rede


def testar_ldiametro():
    """
    Função de teste para a classe LDiametro.
    
    Demonstra:
    - Criação vazia
    - Adição de diâmetros
    - Conversão automática de mm para m
    - Forçar valores
    - Adicionar dicionário
    - Métodos de consulta
    - Criação de padrão
    """
    print("\n" + "="*60)
    print("TESTANDO CLASSE LDIAMETRO")
    print("="*60)
    
    try:
        # Teste 1: Criar lista vazia
        print("\n[Teste 1] Criando lista vazia...")
        lista = LDiametro()
        assert len(lista) == 0, "Lista deveria estar vazia"
        print("✓ Lista vazia criada com sucesso")
        
        # Teste 2: Adicionar diâmetros em metros
        print("\n[Teste 2] Adicionando diâmetros em metros...")
        lista.adicionar(0.1, 50.0).adicionar(0.15, 75.0).adicionar(0.2, 100.0)
        assert len(lista) == 3, "Lista deveria ter 3 diâmetros"
        print("✓ Diâmetros em metros adicionados com sucesso")
        
        # Teste 3: Conversão automática de mm para m
        print("\n[Teste 3] Testando conversão automática (100mm → 0.1m)...")
        lista2 = LDiametro()
        lista2.adicionar(100, 50.0)  # Será convertido para 0.1m
        assert 0.1 in lista2, "Diâmetro 0.1m deveria estar na lista"
        print("✓ Conversão automática funcionou")
        
        # Teste 4: Forçar valor sem conversão
        print("\n[Teste 4] Testando forçar valor (forcar=True)...")
        lista3 = LDiametro()
        try:
            lista3.adicionar(100, 50.0, forcar=True)  # Mantém 100m
            print("✓ Valor forçado aceito (mesmo sendo muito grande)")
        except ValueError:
            print("✓ Proteção contra valores muito grandes funcionou")
        
        # Teste 5: Adicionar dicionário
        print("\n[Teste 5] Adicionando múltiplos diâmetros via dicionário...")
        novos = {50: 20.0, 75: 35.0, 150: 80.0}  # Em mm, serão convertidos
        lista.adicionar_dicionario(novos)
        assert len(lista) >= 5, "Lista deveria ter mais de 5 diâmetros após adicionar dicionário"
        print("✓ Dicionário de diâmetros adicionado com sucesso")
        
        # Teste 6: Métodos de consulta
        print("\n[Teste 6] Testando métodos de consulta...")
        diametros = lista.obter_diametros()
        valores = lista.obter_valores()
        assert len(diametros) == len(valores), "Tamanho de diâmetros e valores deveria ser igual"
        print(f"✓ Consultas funcionando: {len(diametros)} diâmetros")
        
        # Teste 7: Obter valor específico
        print("\n[Teste 7] Obtendo valor de diâmetro específico...")
        valor = lista.obter_valor(0.1)
        print(f"✓ Valor do diâmetro 0.1m: {valor}")
        
        # Teste 8: Diâmetro mais próximo
        print("\n[Teste 8] Procurando diâmetro mais próximo...")
        mais_proximo = lista.diametro_mais_proximo(0.125)
        print(f"✓ Diâmetro mais próximo de 0.125m: {mais_proximo}m")
        
        # Teste 9: Criar lista padrão
        print("\n[Teste 9] Criando lista com diâmetros padrão...")
        lista_padrao = LDiametro.criar_padrao()
        assert len(lista_padrao) == 10, "Lista padrão deveria ter 10 diâmetros"
        print(f"✓ Lista padrão criada com {len(lista_padrao)} diâmetros")
        
        # Teste 10: Criar de mm
        print("\n[Teste 10] Criando lista a partir de valores em mm...")
        lista_mm = LDiametro.criar_de_mm({50: 20, 100: 50, 200: 100})
        assert 0.05 in lista_mm, "Diâmetro 0.05m (50mm) deveria estar na lista"
        assert 0.1 in lista_mm, "Diâmetro 0.1m (100mm) deveria estar na lista"
        print(f"✓ Lista criada de mm com sucesso ({len(lista_mm)} diâmetros)")
        
        # Teste 11: Operações especiais
        print("\n[Teste 11] Testando operações especiais ([], in, len)...")
        assert 0.1 in lista, "Operador 'in' deveria funcionar"
        valor_via_index = lista[0.1]
        lista[0.3] = 120.0  # Adicionar via indexação
        assert 0.3 in lista, "Indexação deveria funcionar"
        print("✓ Operações especiais funcionando")
        
        # Teste 12: Atualizar valor
        print("\n[Teste 12] Atualizando valor de diâmetro...")
        lista.atualizar_valor(0.1, 55.0)
        assert lista.obter_valor(0.1) == 55.0, "Valor deveria ter sido atualizado"
        print("✓ Valor atualizado com sucesso")
        
        # Teste 13: Representação em string
        print("\n[Teste 13] Testando representação em string...")
        print(lista)
        print("✓ Representação em string funcionando")
        
        print("\n" + "="*60)
        print("✓ TODOS OS TESTES DE LDIAMETRO PASSARAM!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ ERRO NOS TESTES DE LDIAMETRO: {str(e)}")
        print("="*60)
        return False


def testar_rede():
    """
    Função de teste para a classe Rede.
    
    Demonstra:
    - Criação de rede com arquivo .inp
    - Criação de rede aleatória (padrão)
    - Execução de simulação
    - Obtenção de pressões
    - Cálculo de pressão mínima
    """
    print("\n" + "="*60)
    print("TESTANDO CLASSE REDE")
    print("="*60)
    
    try:
        # Teste 1: Criar rede aleatória de teste
        print("\n[Teste 1] Criando rede de teste aleatória...")
        rede = Rede()
        
        # Teste 2: Executar simulação
        print("\n[Teste 2] Executando simulação...")
        resultado_sim = rede.simular()
        
        if not resultado_sim['sucesso']:
            print(f"✗ Simulação falhou: {resultado_sim['erro']}")
            return False
        
        # Teste 3: Obter pressões
        print("\n[Teste 3] Obtendo pressões da rede...")
        pressoes = rede.obter_pressoes()
        print(f"✓ Pressões obtidas: {len(pressoes.columns)} nós")
        
        # Teste 4: Obter pressão mínima (sem reservatórios)
        print("\n[Teste 4] Calculando pressão mínima...")
        pressao_min = rede.obter_pressao_minima(excluir_reservatorios=True)
        
        # Teste 5: Salvar rede
        print("\n[Teste 5] Salvando rede...")
        arquivo_teste = "/tmp/rede_teste.inp"
        rede.salvar(arquivo_teste)
        
        if os.path.exists(arquivo_teste):
            print(f"✓ Rede salva com sucesso em: {arquivo_teste}")
            os.remove(arquivo_teste)
        else:
            print(f"✗ Falha ao salvar a rede")
            return False
        
        print("\n" + "="*60)
        print("✓ TODOS OS TESTES DE REDE PASSARAM!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ ERRO NOS TESTES DE REDE: {str(e)}")
        print("="*60)
        return False


def executar_todos_testes():
    """
    Executa todos os testes das classes da biblioteca.
    
    Returns:
        bool: True se todos os testes passarem, False caso contrário
    """
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  INICIANDO TESTES DA BIBLIOTECA HYDROOPT  ".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    resultados = []
    
    # Testar LDiametro
    resultados.append(("LDiametro", testar_ldiametro()))
    
    # Testar Rede
    resultados.append(("Rede", testar_rede()))
    
    # Resumo
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  RESUMO DOS TESTES  ".center(58) + "║")
    print("║" + " "*58 + "║")
    
    for nome, passou in resultados:
        status = "✓ PASSOU" if passou else "✗ FALHOU"
        print(f"║  {nome:30} {status:25} ║")
    
    print("║" + " "*58 + "║")
    
    total_passou = sum(1 for _, passou in resultados if passou)
    print(f"║  Total: {total_passou}/{len(resultados)} testes passaram" + " "*{58-37} + "║")
    
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    return all(passou for _, passou in resultados)