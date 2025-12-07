"""
Exemplo: Leitura de Nós e Tubulações com adaptação do código fornecido
Compatível com WNTR (utilizado por HydroOpt)
"""

from HydroOpt import Rede
import pandas as pd


def exemplo_leitura_nos_e_tubos():
    """
    Lê dados dos nós (pressão) e tubulações (vazão) da rede.
    Adaptação do código fornecido para WNTR.
    """
    print("\n" + "="*70)
    print("LEITURA DE NÓS E TUBULAÇÕES - VERSÃO WNTR")
    print("="*70)
    
    # Criar e simular rede
    rede = Rede()
    resultado_sim = rede.simular()
    
    if not resultado_sim['sucesso']:
        print(f"Erro na simulação: {resultado_sim['erro']}")
        return
    
    # Acessar o objeto WaterNetworkModel
    wn = rede.wn
    resultados = rede.resultados
    
    # ==============================================================
    # PARTE A: Lendo os Nós (Pressão)
    # ==============================================================
    
    dados_nos = []
    
    # Iterar sobre todos os nós (junctions)
    for node_name in wn.junction_name_list:
        node = wn.get_node(node_name)
        
        # Obter a pressão do nó (média ao longo do tempo)
        if node_name in resultados.node['pressure'].columns:
            pressao = resultados.node['pressure'][node_name].mean()
        else:
            pressao = None
        
        dados_nos.append({
            "ID do Nó": node_name,
            "Cota (m)": node.elevation,
            "Demanda (m³/s)": round(node.base_demand, 4),
            "Pressão (mca)": round(pressao, 2) if pressao else None
        })
    
    # Adicionar também os reservatórios
    for reservoir_name in wn.reservoir_name_list:
        reservoir = wn.get_node(reservoir_name)
        
        if reservoir_name in resultados.node['pressure'].columns:
            pressao = resultados.node['pressure'][reservoir_name].mean()
        else:
            pressao = None
        
        # Reservatório pode não ter atributo elevation, usar head_timeseries ou padrão
        cota = getattr(reservoir, 'elevation', 0.0)
        
        dados_nos.append({
            "ID do Nó": reservoir_name,
            "Cota (m)": cota,
            "Demanda (m³/s)": 0.0,
            "Pressão (mca)": round(pressao, 2) if pressao else None
        })
    
    # Criar DataFrame
    df_nos = pd.DataFrame(dados_nos)
    
    print("\n>>> RESULTADOS DOS NÓS:")
    print(df_nos.to_string(index=False))
    print(f"\nTotal de nós: {len(df_nos)}")
    
    print("\n" + "-"*70 + "\n")
    
    # ==============================================================
    # PARTE B: Lendo as Tubulações (Comprimento e Vazão)
    # ==============================================================
    
    dados_tubos = []
    
    # Iterar sobre todas as tubulações (pipes)
    for pipe_name in wn.pipe_name_list:
        pipe = wn.get_link(pipe_name)
        
        # Obter a vazão (média ao longo do tempo)
        if pipe_name in resultados.link['flowrate'].columns:
            vazao_m3_s = resultados.link['flowrate'][pipe_name].mean()
            vazao_l_s = vazao_m3_s * 1000  # Converter m³/s para L/s
        else:
            vazao_l_s = None
        
        dados_tubos.append({
            "ID Tubo": pipe_name,
            "De (Nó)": pipe.start_node,
            "Para (Nó)": pipe.end_node,
            "Comprimento (m)": pipe.length,
            "Diâmetro (mm)": pipe.diameter * 1000,  # Converter m para mm
            "Diâmetro (m)": round(pipe.diameter, 4),
            "Vazão (L/s)": round(vazao_l_s, 2) if vazao_l_s else None
        })
    
    df_tubos = pd.DataFrame(dados_tubos)
    
    print(">>> RESULTADOS DAS TUBULAÇÕES:")
    print(df_tubos.to_string(index=False))
    print(f"\nTotal de tubulações: {len(df_tubos)}")
    
    print("\n" + "-"*70 + "\n")
    
    # ==============================================================
    # ESTATÍSTICAS
    # ==============================================================
    
    print(">>> ESTATÍSTICAS DOS NÓS:")
    print(f"Pressão mínima: {df_nos['Pressão (mca)'].min():.2f} mca")
    print(f"Pressão máxima: {df_nos['Pressão (mca)'].max():.2f} mca")
    print(f"Pressão média: {df_nos['Pressão (mca)'].mean():.2f} mca")
    print(f"Demanda total: {df_nos['Demanda (m³/s)'].sum():.4f} m³/s ({df_nos['Demanda (m³/s)'].sum()*1000:.2f} L/s)")
    
    print("\n>>> ESTATÍSTICAS DAS TUBULAÇÕES:")
    print(f"Vazão mínima: {df_tubos['Vazão (L/s)'].min():.2f} L/s")
    print(f"Vazão máxima: {df_tubos['Vazão (L/s)'].max():.2f} L/s")
    print(f"Vazão média: {df_tubos['Vazão (L/s)'].mean():.2f} L/s")
    
    print("\n" + "="*70 + "\n")
    
    return df_nos, df_tubos


def main():
    """Executar o exemplo"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " LEITURA DE NÓS E TUBULAÇÕES (ADAPTADO) ".center(68) + "║")
    print("╚" + "="*68 + "╝")
    
    try:
        # Leitura principal
        df_nos, df_tubos = exemplo_leitura_nos_e_tubos()
        
        print("\n" + "="*70)
        print("✓ Leitura de dados concluída com sucesso!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()