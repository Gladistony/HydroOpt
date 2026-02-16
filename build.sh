#!/bin/bash
# filepath: /home/gladistony/Github/HydroOpt/build.sh

VERSION="0.6.1"

echo "🔨 Iniciando build versão $VERSION..."

# Limpar builds anteriores para evitar artefatos
echo "🧹 Removendo artefatos de build anteriores..."
rm -rf dist build *.egg-info HydroOpt.egg-info hydroopt.egg-info

# Atualizar versão em arquivos (regex robusto)
echo "📝 Atualizando versão nos arquivos de configuração..."
sed -i 's/version *= *"[0-9]\+\.[0-9]\+\.[0-9]\+"/version = "'"$VERSION"'"/g' pyproject.toml
sed -i 's/version *= *"[0-9]\+\.[0-9]\+\.[0-9]\+"/version = "'"$VERSION"'"/g' setup.py
sed -i 's/__version__ *= *"[0-9]\+\.[0-9]\+\.[0-9]\+"/__version__ = "'"$VERSION"'"/g' HydroOpt/__init__.py

echo "✅ Versão atualizada para $VERSION"

# Build
# Preferir Python do ambiente virtual local (.venv) quando disponível
if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON=$(command -v python3 || command -v python || echo python)
fi

echo "🔨 Compilando pacotes com $PYTHON..."
# Garantir ferramentas de empacotamento
"$PYTHON" -m pip install --upgrade build setuptools wheel >/dev/null 2>&1 || true

# Usar PEP517 build quando disponível, senão fallback para setup.py
if "$PYTHON" -m build --version >/dev/null 2>&1; then
  "$PYTHON" -m build
else
  "$PYTHON" setup.py sdist bdist_wheel
fi

# Verificar
echo ""
echo "📦 Pacotes criados:"
ls -lh dist/hydroopt-$VERSION* 2>/dev/null || echo "  (nenhum pacote encontrado)"

echo ""
echo "✅ Build concluído com sucesso!"
echo ""
echo "Para publicar no PyPI (use o mesmo Python acima):"
echo "  $PYTHON -m pip install twine"
echo "  $PYTHON -m twine upload dist/hydroopt-$VERSION*"