#!/bin/bash
# filepath: /home/gladistony/Github/HydroOpt/build.sh

VERSION="0.5.9"

echo "🔨 Iniciando build versão $VERSION..."

# Atualizar versão em 3 arquivos
echo "📝 Atualizando versão..."
sed -i "s/version = \"0.5.[0-9]*\"/version = \"$VERSION\"/g" pyproject.toml
sed -i "s/version = \"0.5.[0-9]*\"/version = \"$VERSION\"/g" setup.py
sed -i "s/__version__ = \"0.5.[0-9]*\"/__version__ = \"$VERSION\"/g" HydroOpt/__init__.py

echo "✅ Versão atualizada para $VERSION"

# Build
echo "🔨 Compilando pacotes..."
python3 setup.py sdist bdist_wheel

# Verificar
echo ""
echo "📦 Pacotes criados:"
ls -lh dist/hydroopt-$VERSION*

echo ""
echo "✅ Build concluído com sucesso!"
echo ""
echo "Para publicar no PyPI:"
echo "  python3 -m pip install twine"
echo "  python3 -m twine upload dist/hydroopt-$VERSION*"