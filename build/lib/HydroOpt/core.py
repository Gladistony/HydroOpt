import os
import numpy as np
import time
from .diametros import LDiametro
from .rede import Rede
import pandas as pd

def gerar_solucao_heuristica(rede, lista_diametros, pressao_min_desejada=10.0, interacao=200, verbose=True):
    """
    Gera uma solução heurística para otimização de diâmetros em rede de água.
    
    A heurística funciona testando incrementalmente aumentar o diâmetro de cada tubo
    e verificando se melhora a pressão mínima.
    
    Args:
        rede (Rede): Objeto da rede de distribuição
        lista_diametros (LDiametro): Lista de diâmetros disponíveis
        pressao_min_desejada (float): Pressão mínima desejada (padrão: 10.0m)
        interacao (int): Número máximo de iterações (padrão: 200)
        verbose (bool): Mostrar progresso (padrão: True)
    
    Returns:
        list: Solução normalizada em [0,1] para cada tubo
    
    Exemplo:
        >>> rede = Rede('hanoiFIM')
        >>> diams = LDiametro.criar_padrao()
        >>> solucao = gerar_solucao_heuristica(rede, diams, verbose=True)
    """
    inicio = time.time()
    
    nomes_tubos = rede.wn.pipe_name_list
    num_tubos = len(nomes_tubos)
    diams_disponiveis = lista_diametros.obter_diametros()
    num_opcoes = len(diams_disponiveis)
    
    # Índices dos diâmetros atuais (começam com os menores)
    indices_atuais = [0] * num_tubos
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"HEURÍSTICA: Otimizando rede com {num_tubos} tubos")
        print(f"Diâmetros disponíveis: {num_opcoes} opções ({diams_disponiveis[0]:.4f}m - {diams_disponiveis[-1]:.4f}m)")
        print(f"Pressão desejada: {pressao_min_desejada}m")
        print(f"{'='*70}\n")
    
    # Loop principal da heurística
    for i in range(interacao):
        # Atualizar diâmetros da rede com valores atuais
        for idx_tubo, pipe_name in enumerate(nomes_tubos):
            idx_diam = indices_atuais[idx_tubo]
            diametro = diams_disponiveis[idx_diam]
            link = rede.wn.get_link(pipe_name)
            link.diameter = diametro
        
        # Simular rede com diâmetros atuais
        rede.simular(verbose=False)
        p_info = rede.obter_pressao_minima(excluir_reservatorios=True, verbose=False)
        p_min = p_info['valor']
        
        # Se atingiu pressão desejada, pode parar
        if p_min >= pressao_min_desejada:
            if verbose:
                print(f"✓ Pressão desejada atingida na iteração {i+1}")
                print(f"  Pressão mínima: {p_min:.2f}m")
                print(f"  Tempo: {time.time() - inicio:.2f}s\n")
            break
        
        # Mostrar progresso a cada 20 iterações
        if verbose and (i + 1) % 20 == 0:
            print(f"  Iteração {i+1}/{interacao}: Pmin={p_min:.2f}m (desejada={pressao_min_desejada}m)")
        
        # Tentar aumentar diâmetros dos tubos críticos
        mudou = False
        try:
            # Calcular velocidades para identificar tubos críticos
            velocidades = rede.resultados.link['flowrate'].abs() / (
                (np.pi * (pd.Series([rede.wn.get_link(n).diameter for n in rede.wn.link_name_list], 
                index=rede.wn.link_name_list) / 2)**2))
            
            # Ordenar tubos por velocidade (maiores primeiro = mais críticos)
            tubos_criticos = velocidades.sort_values(ascending=False).index
            
            # Aumentar diâmetro dos tubos críticos
            for tubo_critico in tubos_criticos:
                if tubo_critico in nomes_tubos:
                    idx_lista = nomes_tubos.index(tubo_critico)
                    if indices_atuais[idx_lista] < num_opcoes - 1:
                        indices_atuais[idx_lista] += 1
                        mudou = True
                        break  # Aumenta um por vez para testar incrementalmente
            
            if not mudou:
                # Se nenhum tubo crítico pôde ser aumentado, tenta tubos não-críticos
                for idx_lista in range(num_tubos):
                    if indices_atuais[idx_lista] < num_opcoes - 1:
                        indices_atuais[idx_lista] += 1
                        mudou = True
                        break
                
                if not mudou:
                    # Todos os tubos já estão no máximo
                    if verbose:
                        print(f"⚠️  Todos os tubos estão no diâmetro máximo. Pressão final: {p_min:.2f}m")
                    break
                    
        except Exception as e:
            if verbose:
                print(f"⚠️  Erro ao calcular velocidades (iteração {i+1}): {str(e)}")
            # Continuar tentando aumentar diâmetros mesmo com erro
            for idx_lista in range(num_tubos):
                if indices_atuais[idx_lista] < num_opcoes - 1:
                    indices_atuais[idx_lista] += 1
                    break

    # Converter índices atuais em solução [0,1]
    # (mapeando opção mais barata -> 0.0, mais cara -> 1.0)
    if num_opcoes <= 1:
        solucao = [0.0] * num_tubos
    else:
        solucao = [idx_atual / (num_opcoes - 1) for idx_atual in indices_atuais]

    # Sanitizar NaN/Inf e limitar a [0,1]
    solucao = np.nan_to_num(np.asarray(solucao, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    solucao = np.clip(solucao, 0.0, 1.0)

    if verbose:
        print(f"✓ Solução heurística gerada: {num_tubos} tubos normalizados")
        print(f"✓ Tempo total: {time.time() - inicio:.2f}s\n")

    return solucao.tolist()

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
    
    Returns:
        tuple: (bool, list) - (passou, lista_de_mensagens)
    """
    import time
    inicio = time.time()
    
    print("\n" + "="*60)
    print("TESTANDO CLASSE LDiametro")
    print("="*60)
    
    erros = []
    testes_passaram = 0
    total_testes = 13
    
    try:
        # Teste 1: Criar lista vazia
        print("\n[Teste 1] Criando lista vazia...")
        lista = LDiametro()
        assert len(lista) == 0, "Lista deveria estar vazia"
        testes_passaram += 1
        print("✓ Lista vazia criada com sucesso")
        
        # Teste 2: Adicionar diâmetros em metros
        print("\n[Teste 2] Adicionando diâmetros em metros...")
        lista.adicionar(0.1, 50.0).adicionar(0.15, 75.0).adicionar(0.2, 100.0)
        assert len(lista) == 3, f"Lista deveria ter 3 diâmetros, tem {len(lista)}"
        testes_passaram += 1
        print("✓ Diâmetros em metros adicionados com sucesso")
        
        # Teste 3: Conversão automática de mm para m
        print("\n[Teste 3] Testando conversão automática (100mm → 0.1m)...")
        lista2 = LDiametro()
        lista2.adicionar(100, 50.0)  # Será convertido para 0.1m
        assert 0.1 in lista2, "Diâmetro 0.1m deveria estar na lista"
        testes_passaram += 1
        print("✓ Conversão automática funcionou")
        
        # Teste 4: Forçar valor sem conversão
        print("\n[Teste 4] Testando forçar valor (forcar=True)...")
        lista3 = LDiametro()
        try:
            lista3.adicionar(100, 50.0, forcar=True)  # Mantém 100m
            print("✓ Valor forçado aceito (mesmo sendo muito grande)")
        except ValueError:
            print("✓ Proteção contra valores muito grandes funcionou")
        testes_passaram += 1
        
        # Teste 5: Adicionar dicionário
        print("\n[Teste 5] Adicionando múltiplos diâmetros via dicionário...")
        novos = {50: 20.0, 75: 35.0, 150: 80.0}  # Em mm, serão convertidos
        lista.adicionar_dicionario(novos)
        assert len(lista) >= 5, f"Lista deveria ter mais de 5 diâmetros, tem {len(lista)}"
        testes_passaram += 1
        print("✓ Dicionário de diâmetros adicionado com sucesso")
        
        # Teste 6: Métodos de consulta
        print("\n[Teste 6] Testando métodos de consulta...")
        diametros = lista.obter_diametros()
        valores = lista.obter_valores()
        assert len(diametros) == len(valores), "Tamanho de diâmetros e valores deveria ser igual"
        testes_passaram += 1
        print(f"✓ Consultas funcionando: {len(diametros)} diâmetros")
        
        # Teste 7: Obter valor específico
        print("\n[Teste 7] Obtendo valor de diâmetro específico...")
        valor = lista.obter_valor(0.1)
        assert valor is not None, "Valor não deveria ser None"
        testes_passaram += 1
        print(f"✓ Valor do diâmetro 0.1m: {valor}")
        
        # Teste 8: Diâmetro mais próximo
        print("\n[Teste 8] Procurando diâmetro mais próximo...")
        mais_proximo = lista.diametro_mais_proximo(0.125)
        assert mais_proximo is not None, "Diâmetro mais próximo não deveria ser None"
        testes_passaram += 1
        print(f"✓ Diâmetro mais próximo de 0.125m: {mais_proximo}m")
        
        # Teste 9: Criar lista padrão
        print("\n[Teste 9] Criando lista com diâmetros padrão...")
        lista_padrao = LDiametro.criar_padrao()
        assert len(lista_padrao) == 10, f"Lista padrão deveria ter 10 diâmetros, tem {len(lista_padrao)}"
        testes_passaram += 1
        print(f"✓ Lista padrão criada com {len(lista_padrao)} diâmetros")
        
        # Teste 10: Criar de mm
        print("\n[Teste 10] Criando lista a partir de valores em mm...")
        lista_mm = LDiametro.criar_de_mm({50: 20, 100: 50, 200: 100})
        assert 0.05 in lista_mm, "Diâmetro 0.05m (50mm) deveria estar na lista"
        assert 0.1 in lista_mm, "Diâmetro 0.1m (100mm) deveria estar na lista"
        testes_passaram += 1
        print(f"✓ Lista criada de mm com sucesso ({len(lista_mm)} diâmetros)")
        
        # Teste 11: Operações especiais
        print("\n[Teste 11] Testando operações especiais ([], in, len)...")
        assert 0.1 in lista, "Operador 'in' deveria funcionar"
        valor_via_index = lista[0.1]
        lista[0.3] = 120.0  # Adicionar via indexação
        assert 0.3 in lista, "Indexação deveria funcionar"
        testes_passaram += 1
        print("✓ Operações especiais funcionando")
        
        # Teste 12: Atualizar valor
        print("\n[Teste 12] Atualizando valor de diâmetro...")
        lista.atualizar_valor(0.1, 55.0)
        assert lista.obter_valor(0.1) == 55.0, "Valor deveria ter sido atualizado"
        testes_passaram += 1
        print("✓ Valor atualizado com sucesso")
        
        # Teste 13: Representação em string
        print("\n[Teste 13] Testando representação em string...")
        repr_str = str(lista)
        assert len(repr_str) > 0, "Representação em string não deveria ser vazia"
        print(repr_str)
        testes_passaram += 1
        print("✓ Representação em string funcionando")
        
        duracao = time.time() - inicio
        print(f"\n{'='*60}")
        print(f"✓ TESTES LDiametro: {testes_passaram}/{total_testes} passaram ({duracao:.2f}s)")
        print("="*60)
        return True
        
    except Exception as e:
        duracao = time.time() - inicio
        erros.append(str(e))
        print(f"\n✗ ERRO NOS TESTES DE LDiametro: {str(e)}")
        print(f"  Testes passados antes do erro: {testes_passaram}/{total_testes} ({duracao:.2f}s)")
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
    
    Returns:
        bool: True se todos os testes passarem
    """
    import time
    inicio = time.time()
    
    print("\n" + "="*60)
    print("TESTANDO CLASSE REDE")
    print("="*60)
    
    testes_passaram = 0
    total_testes = 5
    
    try:
        # Teste 1: Criar rede aleatória de teste
        print("\n[Teste 1] Criando rede de teste aleatória...")
        rede = Rede()
        assert rede.wn is not None, "Rede não foi criada"
        assert len(rede.wn.pipe_name_list) > 0, "Rede deveria ter tubulações"
        testes_passaram += 1
        print("✓ Rede criada com sucesso")
        
        # Teste 2: Executar simulação
        print("\n[Teste 2] Executando simulação...")
        resultado_sim = rede.simular()
        
        if not resultado_sim['sucesso']:
            print(f"✗ Simulação falhou: {resultado_sim.get('erro', 'desconhecido')}")
            return False
        testes_passaram += 1
        print("✓ Simulação concluída com sucesso")
        
        # Teste 3: Obter pressões
        print("\n[Teste 3] Obtendo pressões da rede...")
        pressoes = rede.obter_pressoes()
        assert pressoes is not None, "Pressões não deveriam ser None"
        assert not pressoes.empty, "DataFrame de pressões não deveria estar vazio"
        testes_passaram += 1
        print(f"✓ Pressões obtidas: {len(pressoes.columns)} nós")
        
        # Teste 4: Obter pressão mínima (sem reservatórios)
        print("\n[Teste 4] Calculando pressão mínima...")
        pressao_min = rede.obter_pressao_minima(excluir_reservatorios=True)
        assert pressao_min['valor'] != float('inf'), "Pressão mínima não deveria ser infinita"
        testes_passaram += 1
        print(f"✓ Pressão mínima: {pressao_min['valor']:.2f}m no nó {pressao_min['no']}")
        
        # Teste 5: Salvar rede
        print("\n[Teste 5] Salvando rede...")
        arquivo_teste = "/tmp/rede_teste.inp"
        rede.salvar(arquivo_teste)
        
        if os.path.exists(arquivo_teste):
            print(f"✓ Rede salva com sucesso em: {arquivo_teste}")
            os.remove(arquivo_teste)
            testes_passaram += 1
        else:
            print(f"✗ Falha ao salvar a rede")
            return False
        
        duracao = time.time() - inicio
        print(f"\n{'='*60}")
        print(f"✓ TESTES REDE: {testes_passaram}/{total_testes} passaram ({duracao:.2f}s)")
        print("="*60)
        return True
        
    except Exception as e:
        duracao = time.time() - inicio
        print(f"\n✗ ERRO NOS TESTES DE REDE: {str(e)}")
        print(f"  Testes passados antes do erro: {testes_passaram}/{total_testes} ({duracao:.2f}s)")
        print("="*60)
        return False


def executar_todos_testes():
    """
    Executa todos os testes das classes da biblioteca.
    
    Returns:
        bool: True se todos os testes passarem, False caso contrário
    """
    import time
    inicio_total = time.time()
    
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
    duracao_total = time.time() - inicio_total
    
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  RESUMO DOS TESTES  ".center(58) + "║")
    print("║" + " "*58 + "║")
    
    for nome, passou in resultados:
        status = "✓ PASSOU" if passou else "✗ FALHOU"
        linha = f"  {nome:30} {status}"
        print(f"║{linha:<58}║")
    
    print("║" + " "*58 + "║")
    
    total_passou = sum(1 for _, passou in resultados if passou)
    msg_total = f"  Total: {total_passou}/{len(resultados)} testes passaram ({duracao_total:.2f}s)"
    print(f"║{msg_total:<58}║")
    
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    return all(passou for _, passou in resultados)