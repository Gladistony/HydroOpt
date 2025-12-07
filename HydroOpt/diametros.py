class LDiametro:
    """
    Classe para gerenciar uma lista de diâmetros disponíveis e seus valores/custos.
    
    Permite criar uma biblioteca de diâmetros comerciais com seus respectivos custos
    para uso em otimização de redes hidráulicas.
    
    IMPORTANTE: Diâmetros devem ser informados em METROS.
    """
    
    # Limite para conversão automática (diâmetros > 10m são provavelmente em mm)
    LIMITE_CONVERSAO_MM = 10.0
    
    def __init__(self, diametros=None):
        """
        Inicializa a lista de diâmetros.
        
        Args:
            diametros (dict, optional): Dicionário com diâmetros como chave e valores/custos.
                                       Formato: {diametro: valor}
                                       Exemplo: {0.1: 50.0, 0.15: 75.0, 0.2: 100.0}
                                       Se None, cria uma lista vazia.
        """
        self._diametros = {}
        self._penalidade = 1e5  # valor alto padrão
        
        if diametros is None:
            print("Lista de diâmetros criada vazia.")
        else:
            if not isinstance(diametros, dict):
                raise TypeError("O parâmetro 'diametros' deve ser um dicionário.")
            
            # Adicionar cada diâmetro com validação
            for diametro, valor in diametros.items():
                self.adicionar(diametro, valor, forcar=False)
            
            print(f"Lista de diâmetros criada com {len(self._diametros)} diâmetros.")
        self._atualizar_penalidade()
    
    def adicionar(self, diametro, valor, forcar=False):
        """
        Adiciona um diâmetro à lista.
        
        Args:
            diametro (float): Diâmetro em METROS (valores > 10 serão tratados como mm)
            valor (float): Valor/custo associado ao diâmetro
            forcar (bool): Se True, ignora a conversão automática de mm para m.
                          Use apenas se tiver certeza que o valor está em metros.
        
        Returns:
            LDiametro: Retorna self para permitir encadeamento
            
        Raises:
            ValueError: Se o diâmetro for inválido
        """
        if not isinstance(diametro, (int, float)) or diametro <= 0:
            raise ValueError("O diâmetro deve ser um número positivo.")
        
        if not isinstance(valor, (int, float)) or valor < 0:
            raise ValueError("O valor deve ser um número não-negativo.")
        
        diametro_original = diametro
        
        # Verificar se o diâmetro precisa ser convertido de mm para m
        if not forcar and diametro > self.LIMITE_CONVERSAO_MM:
            diametro_convertido = diametro / 1000.0
            print(f"⚠️  AVISO: Diâmetro {diametro_original} parece estar em milímetros.")
            print(f"   Convertendo automaticamente para {diametro_convertido}m ({diametro_original}mm).")
            print(f"   Para forçar o valor original, use: adicionar({diametro_original}, {valor}, forcar=True)")
            diametro = diametro_convertido
        
        # Validação de segurança: diâmetros muito grandes mesmo após conversão
        if diametro > 5.0:  # 5 metros = 5000mm
            raise ValueError(
                f"Diâmetro {diametro}m parece muito grande para uma tubulação comercial.\n"
                f"Diâmetros comerciais típicos: 0.05m (50mm) a 1.0m (1000mm).\n"
                f"Se você realmente deseja adicionar {diametro}m, use forcar=True."
            )
        
        # Validação de segurança: diâmetros muito pequenos
        if diametro < 0.01 and not forcar:  # menor que 10mm
            raise ValueError(
                f"Diâmetro {diametro}m ({diametro*1000}mm) parece muito pequeno.\n"
                f"Diâmetros comerciais típicos começam em 0.025m (25mm).\n"
                f"Se você realmente deseja adicionar este diâmetro, use forcar=True."
            )
        
        self._diametros[float(diametro)] = float(valor)
        
        # Mostrar em mm se for mais legível
        if diametro < 1.0:
            print(f"✓ Diâmetro {diametro}m ({diametro*1000:.0f}mm) adicionado com valor {valor}.")
        else:
            print(f"✓ Diâmetro {diametro}m adicionado com valor {valor}.")
            self._atualizar_penalidade()
        
        return self
    
    def adicionar_polegadas(self, diametro_polegadas, custo_por_metro):
        """
        Adiciona um diâmetro usando unidade em POLEGADAS.
        
        Converte automaticamente polegadas para metros antes de armazenar.
        1 polegada = 25.4 mm = 0.0254 m
        
        Args:
            diametro_polegadas (float): Diâmetro em POLEGADAS
            custo_por_metro (float): Custo por metro linear de tubulação
        
        Returns:
            LDiametro: Retorna self para permitir encadeamento
            
        Exemplo:
            >>> lista = LDiametro()
            >>> lista.adicionar_polegadas(12, 45.73)  # 12 polegadas
            >>> lista.adicionar_polegadas(16, 70.41)  # 16 polegadas
        """
        if not isinstance(diametro_polegadas, (int, float)) or diametro_polegadas <= 0:
            raise ValueError("O diâmetro em polegadas deve ser um número positivo.")
        
        if not isinstance(custo_por_metro, (int, float)) or custo_por_metro < 0:
            raise ValueError("O custo por metro deve ser um número não-negativo.")
        
        # Conversão: 1 polegada = 0.0254 metros
        diametro_metros = diametro_polegadas * 0.0254
        diametro_mm = diametro_polegadas * 25.4
        
        # Adicionar com conversão já feita
        self._diametros[float(diametro_metros)] = float(custo_por_metro)
        
        print(f"✓ Diâmetro {diametro_polegadas}\" ({diametro_mm:.1f}mm / {diametro_metros:.4f}m) adicionado com custo {custo_por_metro}/m.")
        self._atualizar_penalidade()
        
        return self
    
    def adicionar_dicionario(self, diametros, forcar=False):
        """
        Adiciona múltiplos diâmetros de uma vez a partir de um dicionário.
        
        Args:
            diametros (dict): Dicionário {diametro: valor} para adicionar
            forcar (bool): Se True, ignora conversões automáticas para todos os diâmetros
        
        Returns:
            LDiametro: Retorna self para permitir encadeamento
        """
        if not isinstance(diametros, dict):
            raise TypeError("O parâmetro 'diametros' deve ser um dicionário.")
        
        print(f"\nAdicionando {len(diametros)} diâmetros...")
        sucesso = 0
        falhas = 0
        
        for diametro, valor in diametros.items():
            try:
                self.adicionar(diametro, valor, forcar=forcar)
                sucesso += 1
            except (ValueError, TypeError) as e:
                print(f"✗ Erro ao adicionar diâmetro {diametro}: {e}")
                falhas += 1
        
        print(f"\n{sucesso} diâmetros adicionados com sucesso, {falhas} falhas.")
        
        return self
    
    def remover(self, diametro):
        """
        Remove um diâmetro da lista.
        
        Args:
            diametro (float): Diâmetro a ser removido
        
        Returns:
            bool: True se removido com sucesso, False se não encontrado
        """
        if diametro in self._diametros:
            valor = self._diametros.pop(diametro)
            print(f"Diâmetro {diametro}m (valor: {valor}) removido.")
            self._atualizar_penalidade()
            return True
        else:
            print(f"Diâmetro {diametro}m não encontrado na lista.")
            return False
    
    def obter_valor(self, diametro):
        """
        Obtém o valor associado a um diâmetro.
        
        Args:
            diametro (float): Diâmetro a consultar
        
        Returns:
            float: Valor associado ao diâmetro
        
        Raises:
            KeyError: Se o diâmetro não existir na lista
        """
        if diametro not in self._diametros:
            raise KeyError(f"Diâmetro {diametro}m não encontrado na lista.")
        
        return self._diametros[diametro]
    
    def obter_diametros(self):
        """
        Retorna a lista de diâmetros disponíveis ordenada.
        
        Returns:
            list: Lista de diâmetros em ordem crescente
        """
        return sorted(self._diametros.keys())
    
    def obter_valores(self):
        """
        Retorna a lista de valores correspondentes aos diâmetros ordenados.
        
        Returns:
            list: Lista de valores na mesma ordem dos diâmetros
        """
        diametros_ordenados = self.obter_diametros()
        return [self._diametros[d] for d in diametros_ordenados]
    
    def obter_dicionario(self):
        """
        Retorna uma cópia do dicionário completo de diâmetros e valores.
        
        Returns:
            dict: Dicionário {diâmetro: valor}
        """
        return self._diametros.copy()
    
    def quantidade(self):
        """
        Retorna a quantidade de diâmetros na lista.
        
        Returns:
            int: Número de diâmetros
        """
        return len(self._diametros)
    
    def limpar(self):
        """
        Remove todos os diâmetros da lista.
        """
        self._diametros.clear()
        print("Todos os diâmetros foram removidos.")
        self._atualizar_penalidade()
    
    def atualizar_valor(self, diametro, novo_valor):
        """
        Atualiza o valor de um diâmetro existente.
        
        Args:
            diametro (float): Diâmetro a atualizar
            novo_valor (float): Novo valor/custo
        
        Returns:
            bool: True se atualizado, False se diâmetro não existe
        """
        if diametro not in self._diametros:
            print(f"Diâmetro {diametro}m não encontrado. Use adicionar() para criar novo diâmetro.")
            return False
        
        if not isinstance(novo_valor, (int, float)) or novo_valor < 0:
            raise ValueError("O valor deve ser um número não-negativo.")
        
        valor_antigo = self._diametros[diametro]
        self._diametros[diametro] = float(novo_valor)
        print(f"Diâmetro {diametro}m: valor atualizado de {valor_antigo} para {novo_valor}.")
        self._atualizar_penalidade()
        
        return True
    
    def diametro_mais_proximo(self, diametro_desejado):
        """
        Encontra o diâmetro comercial mais próximo de um valor desejado.
        
        Args:
            diametro_desejado (float): Diâmetro alvo
        
        Returns:
            float: Diâmetro comercial mais próximo
        
        Raises:
            ValueError: Se a lista estiver vazia
        """
        if not self._diametros:
            raise ValueError("A lista de diâmetros está vazia.")
        
        diametros = self.obter_diametros()
        mais_proximo = min(diametros, key=lambda d: abs(d - diametro_desejado))
        
        return mais_proximo
    
    def __str__(self):
        """Representação em string da lista de diâmetros."""
        if not self._diametros:
            return "LDiametro(vazio)"
        
        linhas = ["LDiametro:"]
        for diametro in self.obter_diametros():
            valor = self._diametros[diametro]
            linhas.append(f"  {diametro:.3f}m -> {valor:.2f}")
        
        return "\n".join(linhas)
    
    def __repr__(self):
        """Representação técnica da lista de diâmetros."""
        return f"LDiametro({self._diametros})"
    
    def __len__(self):
        """Retorna a quantidade de diâmetros."""
        return len(self._diametros)
    
    def __contains__(self, diametro):
        """Verifica se um diâmetro existe na lista."""
        return diametro in self._diametros
    
    def __getitem__(self, diametro):
        """Permite acesso via indexação: lista[0.2]"""
        return self._diametros[diametro]
    
    def __setitem__(self, diametro, valor):
        """Permite atribuição via indexação: lista[0.2] = 100.0"""
        # Usar o método adicionar para aplicar todas as validações
        self.adicionar(diametro, valor, forcar=False)
        self._atualizar_penalidade()
    
    @classmethod
    def criar_padrao(cls):
        """
        Cria uma lista com diâmetros comerciais padrão e custos relativos.
        
        Returns:
            LDiametro: Nova instância com diâmetros padrão
        """
        diametros_padrao = {
            0.050: 20.0,   # 50mm
            0.075: 35.0,   # 75mm
            0.100: 50.0,   # 100mm
            0.150: 75.0,   # 150mm
            0.200: 100.0,  # 200mm
            0.250: 130.0,  # 250mm
            0.300: 165.0,  # 300mm
            0.400: 230.0,  # 400mm
            0.500: 300.0,  # 500mm
            0.600: 380.0,  # 600mm
        }
        
        print("Lista de diâmetros padrão criada.")
        return cls(diametros_padrao)
    
    @classmethod
    def criar_de_mm(cls, diametros_mm):
        """
        Cria uma lista de diâmetros a partir de valores em milímetros.
        
        Args:
            diametros_mm (dict): Dicionário {diametro_mm: valor}
                                Exemplo: {100: 50.0, 150: 75.0}
        
        Returns:
            LDiametro: Nova instância com diâmetros convertidos para metros
        """
        if not isinstance(diametros_mm, dict):
            raise TypeError("O parâmetro 'diametros_mm' deve ser um dicionário.")
        
        diametros_m = {d/1000.0: v for d, v in diametros_mm.items()}
        
        print(f"Criando lista com {len(diametros_m)} diâmetros (convertidos de mm para m).")
        return cls(diametros_m)

    # -------------------- Penalidade --------------------
    def _atualizar_penalidade(self):
        """Atualiza a penalidade baseada no maior valor da lista."""
        if not self._diametros:
            self._penalidade = 1e5
        else:
            maior_valor = max(self._diametros.values())
            self._penalidade = max(1e3, maior_valor * 10)

    def obter_penalidade(self):
        """Retorna a penalidade atual calculada."""
        return self._penalidade
