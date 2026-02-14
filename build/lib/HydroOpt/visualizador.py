#!/usr/bin/env python3
"""
Visualizador de Rede (GUI mínima)

Classe `VisualizadorRede` que recebe uma instância de `Rede` (ou um
`wntr.network.WaterNetworkModel`) e fornece `show()` para exibir uma
visualização com pressões por nó e nomes dos tubos.

Comportamento inspirado no código de referência fornecido.
"""
from typing import Optional
import io
import logging

try:
    import wntr
except Exception as e:
    raise ImportError("WNTR é necessário para VisualizadorRede. Instale com: pip install wntr") from e

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from PIL import Image
except Exception as e:
    raise ImportError("Pillow é necessário para VisualizadorRede. Instale com: pip install pillow") from e

import os
import sys
import shutil
import subprocess
import tempfile

logger = logging.getLogger(__name__)


class VisualizadorRede:
    """Visualizador simples de rede hidráulica.

    Args:
        rede: instância de `Rede` (HydroOpt) ou `wntr.network.WaterNetworkModel`.
        ajustar_cota_reservatorio (bool): se True, tenta aumentar a cota do
            reservatório até atingir `pressao_minima_desejada`.
        nome_reservatorio (str|None): nome do reservatório a ajustar (se aplicável).
        incremento_cota (float): incremento por iteração ao ajustar cota.
        pressao_minima_desejada (float): pressão mínima alvo (m).
        max_iteracoes (int): limite de iterações ao ajustar cota.
    """

    def __init__(
        self,
        rede,
        ajustar_cota_reservatorio: bool = False,
        nome_reservatorio: Optional[str] = None,
        incremento_cota: float = 5.0,
        pressao_minima_desejada: float = 10.0,
        max_iteracoes: int = 100,
    ):
        # Aceita tanto Rede (wrapper) quanto WaterNetworkModel
        if hasattr(rede, 'wn'):
            self.wn = rede.wn
        else:
            self.wn = rede

        self.ajustar_cota_reservatorio = ajustar_cota_reservatorio
        self.nome_reservatorio = nome_reservatorio
        self.incremento_cota = float(incremento_cota)
        self.pressao_minima_desejada = float(pressao_minima_desejada)
        self.max_iteracoes = int(max_iteracoes)

    def _adjust_reservoir_if_needed(self):
        """Ajusta a cota do reservatório iterativamente até atingir a pressão mínima.

        Retorna a cota final (float) ou None se não ajustado.
        """
        if not self.ajustar_cota_reservatorio:
            return None

        if self.nome_reservatorio is None:
            # Tentar usar o primeiro reservatório disponível
            res_list = self.wn.reservoir_name_list
            if not res_list:
                logger.warning("Nenhum reservatório encontrado na rede para ajustar cota.")
                return None
            nome = res_list[0]
        else:
            nome = self.nome_reservatorio

        if nome not in self.wn.reservoir_name_list:
            logger.warning(f"Reservatório '{nome}' não encontrado na rede")
            return None

        reservatorio = self.wn.get_node(nome)
        # garantir base_head disponível
        cota_atual = getattr(reservatorio, 'base_head', None)
        if cota_atual is None:
            logger.warning(f"Reservatório '{nome}' não tem atributo base_head")
            return None

        iteracao = 0
        while iteracao < self.max_iteracoes:
            reservatorio.base_head = cota_atual
            try:
                try:
                    sim = wntr.sim.EpanetSimulator(self.wn)
                except Exception:
                    sim = wntr.sim.WNTRSimulator(self.wn)
                results_temp = sim.run_sim()
            except Exception as e:
                logger.error(f"Erro na simulação durante ajuste de cota: {e}")
                break

            # Excluir reservatórios para calcular mínimo
            pressure_df_temp = results_temp.node['pressure']
            reservoirs_temp = set(self.wn.reservoir_name_list)
            junctions_tanks_temp = [n for n in pressure_df_temp.columns if n not in reservoirs_temp]
            pressao_minima_atual = pressure_df_temp[junctions_tanks_temp].min().min()

            logger.debug(f"Iter {iteracao}: cota {cota_atual}, pressao min {pressao_minima_atual}")
            if pressao_minima_atual >= self.pressao_minima_desejada:
                return cota_atual

            cota_atual += self.incremento_cota
            iteracao += 1

        return cota_atual

    def _generate_image(self, results):
        """Gera PIL.Image com a plot da rede mostrando pressões e nomes dos tubos."""
        if plt is None:
            raise RuntimeError("matplotlib não disponível para renderização de visualização")

        # Coordenadas
        coords = self.wn.query_node_attribute('coordinates')
        pos = {n: (xy[0], xy[1]) for n, xy in coords.items() if xy is not None}

        try:
            pressure_df = results.node['pressure']
        except Exception:
            raise RuntimeError("Resultados de pressão não disponíveis para renderização")

        reservoirs = set(self.wn.reservoir_name_list)
        junctions_tanks = [n for n in pressure_df.columns if n not in reservoirs]

        per_node_min = pressure_df[junctions_tanks].min()
        node_min = per_node_min.idxmin()
        val_min = per_node_min.min()
        time_min = pressure_df[node_min].idxmin()

        pressures_at_t = pressure_df.loc[time_min]

        fig, ax = plt.subplots(figsize=(10, 8))

        # Desenha links e rótulos
        for link_name in self.wn.link_name_list:
            link = self.wn.get_link(link_name)
            n1 = link.start_node.name
            n2 = link.end_node.name
            if n1 in pos and n2 in pos:
                x1, y1 = pos[n1]
                x2, y2 = pos[n2]
                ax.plot([x1, x2], [y1, y2], color="#bbbbbb", linewidth=1.0)
                xm, ym = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                ax.text(xm, ym, link_name, fontsize=8, color="#555555", ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))

        # Nós
        junctions = set(self.wn.junction_name_list)
        tanks = set(self.wn.tank_name_list)
        reservoirs = set(self.wn.reservoir_name_list)

        def scatter_nodes(nodes, color, label):
            xs = [pos[n][0] for n in nodes if n in pos]
            ys = [pos[n][1] for n in nodes if n in pos]
            ax.scatter(xs, ys, s=36, c=color, label=label, alpha=0.95, edgecolors='white', linewidths=0.6)

        scatter_nodes(junctions, "#1f77b4", "Junções")
        scatter_nodes(tanks, "#ff7f0e", "Tanques")
        scatter_nodes(reservoirs, "#2ca02c", "Reservatórios")

        # Nomes e pressões
        for n, p in pressures_at_t.items():
            if n in pos:
                x, y = pos[n]
                ax.text(x, y, f"{n}\n{p:.2f} m", fontsize=7, color="#222222", ha='left', va='bottom')

        # Destaque nó com menor pressão
        if node_min in pos:
            x, y = pos[node_min]
            ax.scatter([x], [y], s=120, facecolors='none', edgecolors='red', linewidths=2.0)
            ax.text(x, y, f"MIN {val_min:.2f} m", fontsize=9, color="red", ha='right', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='red', pad=0.5))

        ax.set_title("Rede com pressão por nó e nomes dos tubos")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=160)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    def render(self, inp_path: Optional[str] = None):
        """Executa a simulação e retorna uma `PIL.Image` com a visualização.

        Útil para exibir em notebooks (Colab, Jupyter) usando `IPython.display`.
        """
        if inp_path is not None:
            wn_local = wntr.network.WaterNetworkModel(inp_path)
            self.wn = wn_local

        # Ajustar cota do reservatório (opcional)
        if self.ajustar_cota_reservatorio:
            self._adjust_reservoir_if_needed()

        try:
            try:
                sim = wntr.sim.EpanetSimulator(self.wn)
            except Exception:
                sim = wntr.sim.WNTRSimulator(self.wn)
            results = sim.run_sim()
        except Exception as e:
            raise RuntimeError(f"Falha na simulação: {e}")

        return self._generate_image(results)

    def show(self, inp_path: Optional[str] = None):
        """Exibe a janela com a rede.

        Se `inp_path` for passado, cria um WaterNetworkModel a partir do arquivo;
        caso contrário usa o `wn` já associado.
        """
        if inp_path is not None:
            wn_local = wntr.network.WaterNetworkModel(inp_path)
            self.wn = wn_local

        # Ajustar cota do reservatório (opcional)
        if self.ajustar_cota_reservatorio:
            self._adjust_reservoir_if_needed()

        try:
            try:
                sim = wntr.sim.EpanetSimulator(self.wn)
            except Exception:
                sim = wntr.sim.WNTRSimulator(self.wn)
            results = sim.run_sim()
        except Exception as e:
            raise RuntimeError(f"Falha na simulação: {e}")

        img = self._generate_image(results)

        # Tentar exibir via PIL (usa visualizador do sistema)
        try:
            img.show()
            return
        except Exception:
            # se falhar, seguimos para salvar em arquivo temporário e abrir com o utilitário do SO
            pass

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        try:
            img.save(tmp.name, format='PNG')
        finally:
            tmp.close()

        tmp_path = tmp.name
        opened = False
        try:
            if sys.platform.startswith("linux"):
                if shutil.which("xdg-open"):
                    subprocess.Popen(["xdg-open", tmp_path])
                    opened = True
            elif sys.platform == "darwin":
                subprocess.Popen(["open", tmp_path])
                opened = True
            elif sys.platform.startswith("win"):
                try:
                    os.startfile(tmp_path)  # type: ignore
                    opened = True
                except Exception:
                    opened = False
        except Exception:
            opened = False

        if not opened:
            print(f"Imagem salva em: {tmp_path}")
