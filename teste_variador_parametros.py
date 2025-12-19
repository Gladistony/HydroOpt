"""
Testes unitários para a classe VariadorDeParametros
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path

try:
    from HydroOpt import Rede, Otimizador, LDiametro, VariadorDeParametros
except ImportError:
    print("Erro ao importar HydroOpt. Certifique-se de estar no diretório raiz do projeto.")
    exit(1)


class TestVariadorDeParametros(unittest.TestCase):
    """Testes para a classe VariadorDeParametros"""
    
    @classmethod
    def setUpClass(cls):
        """Setup que roda uma única vez antes de todos os testes"""
        cls.rede = Rede('hanoiFIM')
        
        cls.diametros = LDiametro()
        cls.diametros.adicionar_polegadas(12, 45.73)
        cls.diametros.adicionar_polegadas(16, 70.40)
        cls.diametros.adicionar_polegadas(20, 98.38)
        cls.diametros.adicionar_polegadas(24, 129.30)
        cls.diametros.adicionar_polegadas(30, 180.80)
        cls.diametros.adicionar_polegadas(40, 278.30)
        
        cls.otimizador = Otimizador(
            rede=cls.rede,
            diametros=cls.diametros,
            epoch=5,  # Reduzido para testes
            pop_size=10,
            usar_paralelismo=False,
            verbose=False
        )
        cls.otimizador.pressao_min_desejada = 30.0
        
        # População inicial
        num_tubos = len(cls.rede.wn.pipe_name_list)
        cls.populacao_inicial = [
            np.random.uniform(0, 1, num_tubos).tolist()
            for _ in range(10)
        ]
    
    def setUp(self):
        """Setup que roda antes de cada teste"""
        self.variador = VariadorDeParametros(self.otimizador, verbose=False)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Cleanup após cada teste"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_inicializacao(self):
        """Testa inicialização do variador"""
        self.assertIsNotNone(self.variador)
        self.assertEqual(len(self.variador.ranges_parametros), 0)
        self.assertIsNone(self.variador.dataframe_resultados)
    
    def test_definir_parametro_valido(self):
        """Testa definição de parâmetro válido"""
        self.variador.definir_parametro('c1', inicial=1.5, final=2.5, passo=0.5)
        
        self.assertIn('c1', self.variador.ranges_parametros)
        self.assertEqual(self.variador.ranges_parametros['c1']['inicial'], 1.5)
        self.assertEqual(self.variador.ranges_parametros['c1']['final'], 2.5)
        self.assertEqual(self.variador.ranges_parametros['c1']['passo'], 0.5)
    
    def test_definir_parametro_passo_invalido(self):
        """Testa erro com passo inválido"""
        with self.assertRaises(ValueError):
            self.variador.definir_parametro('c1', inicial=1.5, final=2.5, passo=-0.5)
        
        with self.assertRaises(ValueError):
            self.variador.definir_parametro('c1', inicial=1.5, final=2.5, passo=0)
    
    def test_definir_parametro_intervalo_invalido(self):
        """Testa erro com intervalo inválido"""
        with self.assertRaises(ValueError):
            self.variador.definir_parametro('c1', inicial=2.5, final=1.5, passo=0.5)
    
    def test_definir_condicoes_iniciais(self):
        """Testa definição de condições iniciais"""
        self.variador.definir_condicoes_iniciais(
            populacao_inicial=self.populacao_inicial,
            verbose_otimizacao=False
        )
        
        self.assertIsNotNone(self.variador.populacao_inicial)
        self.assertEqual(len(self.variador.populacao_inicial), 10)
    
    def test_gerar_combinacoes_univariada(self):
        """Testa geração de combinações para um parâmetro"""
        self.variador.definir_parametro('c1', inicial=1.0, final=2.0, passo=0.5)
        
        combinacoes = self.variador._gerar_combinacoes()
        
        # Esperado: [1.0, 1.5, 2.0] = 3 valores
        self.assertEqual(len(combinacoes), 3)
        self.assertIn('c1', combinacoes[0])
    
    def test_gerar_combinacoes_bivariada(self):
        """Testa geração de combinações para dois parâmetros"""
        self.variador.definir_parametro('c1', inicial=1.0, final=1.5, passo=0.5)
        self.variador.definir_parametro('c2', inicial=1.0, final=1.5, passo=0.5)
        
        combinacoes = self.variador._gerar_combinacoes()
        
        # Esperado: [1.0, 1.5] * [1.0, 1.5] = 4 combinações
        self.assertEqual(len(combinacoes), 4)
        self.assertIn('c1', combinacoes[0])
        self.assertIn('c2', combinacoes[0])
    
    def test_executar_sem_parametros(self):
        """Testa erro ao executar sem parâmetros definidos"""
        with self.assertRaises(ValueError):
            self.variador.executar(metodo='PSO')
    
    def test_executar_com_parametro_simples(self):
        """Testa execução com um parâmetro (pode ser lento)"""
        # Apenas teste se há tempo
        if True:  # Alterar para False se quiser pular teste longo
            self.variador.definir_parametro('c1', inicial=1.5, final=1.5, passo=0.5)
            self.variador.definir_condicoes_iniciais(
                populacao_inicial=self.populacao_inicial,
                verbose_otimizacao=False
            )
            
            df = self.variador.executar(metodo='PSO', diretorio_saida=None)
            
            self.assertIsNotNone(df)
            self.assertGreater(len(df), 0)
            self.assertIn('combinacao_id', df.columns)
            self.assertIn('custo_real_R$', df.columns)
    
    def test_obter_resumo_sem_execucao(self):
        """Testa erro ao obter resumo sem executar"""
        with self.assertRaises(ValueError):
            self.variador.obter_resumo()
    
    def test_obter_melhor_resultado_sem_execucao(self):
        """Testa erro ao obter melhor resultado sem executar"""
        with self.assertRaises(ValueError):
            self.variador.obter_melhor_resultado()
    
    def test_processar_resultados(self):
        """Testa processamento de resultados"""
        # Criar resultados fictícios
        self.variador.ranges_parametros = {'c1': {'inicial': 1.5, 'final': 1.5, 'passo': 0.5}}
        self.variador.resultados = [
            {
                'combinacao_id': 0,
                'parametros': {'c1': 1.5},
                'melhor_custo_fitness': 100.0,
                'custo_real': 50000.0,
                'pressao_minima': 30.0,
                'no_pressao_minima': 'node1',
                'sucesso': True,
                'melhor_solucao': [0.5, 0.5]
            }
        ]
        
        self.variador._processar_resultados()
        
        self.assertIsNotNone(self.variador.dataframe_resultados)
        self.assertEqual(len(self.variador.dataframe_resultados), 1)
        self.assertIn('custo_real_R$', self.variador.dataframe_resultados.columns)
    
    def test_comparar_parametros_univariada(self):
        """Testa comparação univariada de parâmetros"""
        self.variador.ranges_parametros = {'c1': {'inicial': 1.5, 'final': 1.5, 'passo': 0.5}}
        self.variador.resultados = [
            {
                'combinacao_id': 0,
                'parametros': {'c1': 1.5},
                'melhor_custo_fitness': 100.0,
                'custo_real': 50000.0,
                'pressao_minima': 30.0,
                'no_pressao_minima': 'node1',
                'sucesso': True,
                'melhor_solucao': [0.5, 0.5]
            }
        ]
        self.variador._processar_resultados()
        
        comparacao = self.variador.comparar_parametros('c1')
        
        self.assertIsNotNone(comparacao)
        self.assertIn('c1', comparacao.index)
    
    def test_obter_informacoes_sem_execucao(self):
        """Testa obter informações sem executar"""
        info = self.variador.obter_informacoes()
        
        self.assertIn('status', info)
        self.assertEqual(info['status'], 'Nenhuma varredura executada ainda')
    
    def test_obter_informacoes_com_execucao(self):
        """Testa obter informações após execução fictícia"""
        self.variador.resultados = [
            {
                'combinacao_id': 0,
                'parametros': {'c1': 1.5},
                'melhor_custo_fitness': 100.0,
                'custo_real': 50000.0,
                'pressao_minima': 30.0,
                'no_pressao_minima': 'node1',
                'sucesso': True,
                'melhor_solucao': [0.5, 0.5]
            },
            {
                'combinacao_id': 1,
                'parametros': {'c1': 1.0},
                'melhor_custo_fitness': 110.0,
                'custo_real': 55000.0,
                'pressao_minima': 29.0,
                'no_pressao_minima': 'node2',
                'sucesso': True,
                'melhor_solucao': [0.5, 0.5]
            }
        ]
        
        info = self.variador.obter_informacoes()
        
        self.assertEqual(info['total_combinacoes'], 2)
        self.assertEqual(info['sucessos'], 2)
        self.assertEqual(info['falhas'], 0)
        self.assertEqual(info['melhor_custo'], 50000.0)
        self.assertEqual(info['pior_custo'], 55000.0)


class TestVariadorIntegracaoCompleta(unittest.TestCase):
    """Testes de integração mais complexos"""
    
    @classmethod
    def setUpClass(cls):
        """Setup compartilhado para testes de integração"""
        cls.rede = Rede('hanoiFIM')
        
        cls.diametros = LDiametro()
        cls.diametros.adicionar_polegadas(12, 45.73)
        cls.diametros.adicionar_polegadas(16, 70.40)
        cls.diametros.adicionar_polegadas(20, 98.38)
        cls.diametros.adicionar_polegadas(24, 129.30)
        cls.diametros.adicionar_polegadas(30, 180.80)
        cls.diametros.adicionar_polegadas(40, 278.30)
    
    def test_workflow_completo_simples(self):
        """Testa workflow completo simplificado"""
        otimizador = Otimizador(
            rede=self.rede,
            diametros=self.diametros,
            epoch=3,  # Bem reduzido
            pop_size=5,
            usar_paralelismo=False,
            verbose=False
        )
        
        variador = VariadorDeParametros(otimizador, verbose=False)
        variador.definir_parametro('c1', inicial=2.0, final=2.0, passo=0.5)
        
        num_tubos = len(self.rede.wn.pipe_name_list)
        populacao = [np.random.uniform(0, 1, num_tubos).tolist() for _ in range(5)]
        variador.definir_condicoes_iniciais(populacao_inicial=populacao)
        
        # Executar (pode ser lento)
        df = variador.executar(metodo='PSO', diretorio_saida=None)
        
        self.assertGreater(len(df), 0)
        self.assertTrue('sucesso' in df.columns)
        
        # Obter melhor resultado
        melhor = variador.obter_melhor_resultado()
        self.assertIn('parametros', melhor)
        self.assertIn('custo_real', melhor)


def run_tests_rapidos():
    """Executa apenas testes rápidos (sem otimização real)"""
    suite = unittest.TestSuite()
    
    # Adicionar apenas testes unitários rápidos
    suite.addTest(TestVariadorDeParametros('test_inicializacao'))
    suite.addTest(TestVariadorDeParametros('test_definir_parametro_valido'))
    suite.addTest(TestVariadorDeParametros('test_definir_parametro_passo_invalido'))
    suite.addTest(TestVariadorDeParametros('test_gerar_combinacoes_univariada'))
    suite.addTest(TestVariadorDeParametros('test_gerar_combinacoes_bivariada'))
    suite.addTest(TestVariadorDeParametros('test_processar_resultados'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


def run_tests_completos():
    """Executa todos os testes (incluindo otimização real)"""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--rapido':
        # Executar apenas testes rápidos
        run_tests_rapidos()
    else:
        # Executar todos os testes
        run_tests_completos()
