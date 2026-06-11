"""
Estimativa do Índice de Gini municipal — Censo 2022 (dados agrupados)
======================================================================
Fonte: SIDRA Tabela 10296 — Moradores em domicílios particulares permanentes
ocupados, por classes de rendimento domiciliar per capita em salários mínimos.
Resultados preliminares da amostra (divulgação out/2025).

Método: Gini de dados agrupados com limites de Gastwirth (1972).
Como as classes de renda são intervalos não sobrepostos, o Gini decompõe-se
exatamente em G = G_entre + soma(p_i * s_i * G_intra_i). Os limites vêm das
hipóteses extremas sobre a distribuição intra-classe:
  - gini_inf: todos no ponto médio da classe (G_intra = 0);
  - gini_sup: massa nos extremos da classe preservando a média
    (G_intra máximo = 0,25*(ls-li)/m para média no ponto médio);
  - classe aberta superior: Pareto com média 2*li (alfa = 2, G_intra = 1/3).
O Gini verdadeiro está entre gini_inf e gini_sup; gini_medio é o ponto
central do intervalo. O valor exato só virá com os microdados da amostra.

Uso:
    pip install requests pandas
    python gini_municipal_2022.py            # todos os municípios
    python gini_municipal_2022.py --uf 35    # só São Paulo (código UF)

Saída: gini_municipal_2022.csv
"""

import argparse
import re
import sys
import time
import unicodedata

import pandas as pd
import requests

TABELA = 10296
ANO = "2022"
SM_2022 = 1212.0  # salário mínimo de referência do Censo 2022 (R$)

META_URL = f"https://servicodados.ibge.gov.br/api/v3/agregados/{TABELA}/metadados"
SIDRA_URL = "https://apisidra.ibge.gov.br/values"

CODIGOS_UF = [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29,
              31, 32, 33, 35, 41, 42, 43, 50, 51, 52, 53]

FRACOES = {"1/4": 0.25, "1/2": 0.5, "3/4": 0.75, "1/8": 0.125}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return s.lower()


def _num(token: str) -> float:
    token = token.strip()
    if token in FRACOES:
        return FRACOES[token]
    try:
        return float(token.replace(",", "."))
    except ValueError:
        raise ValueError(f"Não foi possível converter '{token}' para número. "
                         f"Adicione-o ao dicionário FRACOES se for uma fração.")


def parse_classe(nome: str):
    """Converte o rótulo da classe em (lim_inf, lim_sup) em salários mínimos.

    lim_sup = None para classe aberta superior.
    Retorna None para categorias não-renda (total, sem declaração).
    """
    n = _norm(nome)
    if "total" in n or "sem declaracao" in n:
        return None
    if "sem rendimento" in n:
        return (0.0, 0.0)
    # "Mais de X" (classe aberta)
    m = re.match(r"mais de ([\d/,\.]+)\s*(?:salarios?\b|$)", n)
    if m and " a " not in n and " ate " not in n:
        return (_num(m.group(1)), None)
    # "X ou mais" (classe aberta, formato alternativo)
    m = re.match(r"([\d/,\.]+)\s*ou mais", n)
    if m:
        return (_num(m.group(1)), None)
    # "Mais de X a Y" / "Mais de X até Y"
    m = re.search(r"mais de ([\d/,\.]+)\s*(?:a|ate)\s*([\d/,\.]+)", n)
    if m:
        return (_num(m.group(1)), _num(m.group(2)))
    # "Até X"
    m = re.search(r"ate ([\d/,\.]+)", n)
    if m:
        return (0.0, _num(m.group(1)))
    print(f"  [aviso] classe nao reconhecida, ignorada: {nome!r}", file=sys.stderr)
    return None


def descobrir_estrutura():
    """Lê metadados da tabela e retorna (id_variavel, id_classif_renda,
    {cod_categoria: (nome, (li, ls))}, ids_classifs_para_totalizar)."""
    meta = requests.get(META_URL, timeout=60).json()

    # Variável: moradores (pessoas), evita variáveis de percentual
    id_var = None
    for v in meta["variaveis"]:
        nv = _norm(v["nome"])
        if "percentual" not in nv and "distribuicao" not in nv:
            id_var = v["id"]
            break
    if id_var is None:
        id_var = meta["variaveis"][0]["id"]

    id_renda, categorias = None, {}
    outras = []
    for c in meta["classificacoes"]:
        nc = _norm(c["nome"])
        if "rendimento" in nc and "salario" in nc:
            id_renda = c["id"]
            for cat in c["categorias"]:
                bounds = parse_classe(cat["nome"])
                if bounds is not None:
                    categorias[cat["id"]] = (cat["nome"], bounds)
        else:
            outras.append(c["id"])
    if id_renda is None or not categorias:
        sys.exit("Nao encontrei a classificacao de classes de rendimento. "
                 "Verifique os metadados em: " + META_URL)
    if outras:
        print(f"  [aviso] {len(outras)} classificacao(oes) adicional(is) encontrada(s) "
              f"(ids: {outras}). A consulta omite essas dimensoes; se a API "
              f"retornar cruzamentos, os counts estarao inflados.", file=sys.stderr)
    return id_var, id_renda, categorias, outras


def baixar_uf(uf, id_var, id_renda, cods_renda, outras_classifs):
    """Baixa moradores por classe de renda para todos os municípios de uma UF."""
    classif = f"c{id_renda}/{','.join(str(c) for c in cods_renda)}"
    # demais classificações omitidas -> apisidra retorna o total automaticamente
    url = (f"{SIDRA_URL}/t/{TABELA}/n6/in n3 {uf}"
           f"/v/{id_var}/p/{ANO}/{classif}?formato=json")
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    dados = r.json()
    if len(dados) <= 1:
        return pd.DataFrame()
    df = pd.DataFrame(dados[1:])
    header = dados[0]
    # Identifica colunas dinâmicas; "Código" vem antes de "Nome" na resposta,
    # por isso a busca por nome exclui explicitamente colunas de código.
    col_mun_cod = next(k for k, v in header.items()
                       if "Codigo" in str(v) and "Munic" in str(v))
    col_mun_nome = next(k for k, v in header.items()
                        if str(v).startswith("Munic") and "Codigo" not in str(v))
    col_classe = next(k for k, v in header.items() if "rendimento" in _norm(str(v)))
    out = df[[col_mun_cod, col_mun_nome, col_classe, "V"]].copy()
    out.columns = ["cod_mun", "municipio", "classe", "valor"]
    out["valor"] = pd.to_numeric(out["valor"], errors="coerce")
    return out.dropna(subset=["valor"])


def gini_agrupado(classes):
    """Limites de Gastwirth do Gini a partir de [(li, ls, pop)], em SM.

    Hipóteses: média de classe = ponto médio (classe aberta: 2*li, Pareto
    alfa=2). Retorna (gini_inf, gini_sup) ou None se não houver dados.

    Como as classes não se sobrepõem, vale a decomposição exata
    G = G_entre + soma(p_i * s_i * G_intra_i), em que p_i é a participação
    populacional e s_i a participação na renda da classe i. G_entre é o
    Gini da distribuição com todos na média da classe (= gini_inf).
    """
    dados = []  # (media, pop, g_intra_max)
    for li, ls, pop in classes:
        if pop <= 0:
            continue
        if ls is None:           # classe aberta: Pareto com média 2*li
            media, g_max = 2.0 * li, 1.0 / 3.0
        else:
            media = (li + ls) / 2.0
            # G_intra máximo: 2 pontos nos extremos com média preservada.
            # Para distribuição de 2 massas iguais em li e ls:
            #   G = (ls - li) / (2*(li+ls)) = 0.25*(ls-li)/media
            g_max = 0.25 * (ls - li) / media if media > 0 else 0.0
        dados.append((media, pop, g_max))
    if not dados:
        return None
    dados.sort()
    pop_total = sum(p for _, p, _ in dados)
    renda_total = sum(m * p for m, p, _ in dados)
    if renda_total == 0:
        return (0.0, 0.0)
    # G_entre via Lorenz por trapézios (classes substituídas pelas médias).
    # g_entre acumula sum(p_i*(L_i + L_{i-1})) = 2*Area_Lorenz, logo
    # G = 1 - 2*Area = 1 - g_entre.
    g_entre, P_ant, L_ant = 0.0, 0.0, 0.0
    g_intra = 0.0
    for media, pop, g_max in dados:
        p_i = pop / pop_total
        s_i = (media * pop) / renda_total
        P, L = P_ant + p_i, L_ant + s_i
        g_entre += (P - P_ant) * (L + L_ant)
        g_intra += p_i * s_i * g_max
        P_ant, L_ant = P, L
    g_inf = 1.0 - g_entre
    return (g_inf, min(g_inf + g_intra, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uf", type=int, default=None, help="Codigo IBGE da UF (ex.: 35)")
    ap.add_argument("--saida", default="gini_municipal_2022.csv")
    args = ap.parse_args()

    print("Lendo metadados da tabela 10296...")
    id_var, id_renda, categorias, outras = descobrir_estrutura()
    print(f"  variavel={id_var}, classificacao_renda={id_renda}, "
          f"{len(categorias)} classes de renda")
    for cod, (nome, (li, ls)) in sorted(categorias.items()):
        print(f"    [{cod}] {nome} -> ({li}, {ls})")

    ufs = [args.uf] if args.uf else CODIGOS_UF
    frames = []
    ufs_falhas = []
    for uf in ufs:
        print(f"Baixando UF {uf}...")
        for tentativa in range(3):
            try:
                frames.append(baixar_uf(uf, id_var, id_renda,
                                        list(categorias.keys()), outras))
                break
            except Exception as e:
                print(f"  erro ({e}); retry {tentativa+1}/3", file=sys.stderr)
                time.sleep(5 * (tentativa + 1))
        else:
            print(f"  [erro] UF {uf} falhou permanentemente, ignorada.", file=sys.stderr)
            ufs_falhas.append(uf)
        time.sleep(1)  # cortesia com a API

    if not frames:
        sys.exit("Nenhum dado baixado. Verifique a conexão ou os parâmetros.")
    if ufs_falhas:
        print(f"\n[aviso] {len(ufs_falhas)} UF(s) sem dados: {ufs_falhas}", file=sys.stderr)

    df = pd.concat(frames, ignore_index=True)
    bounds = {nome: b for _, (nome, b) in categorias.items()}

    resultados = []
    for (cod, nome), g in df.groupby(["cod_mun", "municipio"]):
        classes = []
        for _, row in g.iterrows():
            b = bounds.get(row["classe"]) or parse_classe(row["classe"])
            if b is None:
                continue
            classes.append((b[0], b[1], row["valor"]))
        res_gini = gini_agrupado(classes)
        pop = sum(c[2] for c in classes)
        if res_gini is None:
            g_inf = g_sup = g_med = None
        else:
            g_inf, g_sup = res_gini
            g_med = (g_inf + g_sup) / 2.0
        resultados.append({"cod_mun": cod, "municipio": nome,
                           "gini_inf": round(g_inf, 4) if g_inf is not None else None,
                           "gini_sup": round(g_sup, 4) if g_sup is not None else None,
                           "gini_medio": round(g_med, 4) if g_med is not None else None,
                           "populacao_considerada": int(round(pop))})

    res = pd.DataFrame(resultados).sort_values("cod_mun")
    res.to_csv(args.saida, index=False, encoding="utf-8")
    print(f"\n{len(res)} municipios -> {args.saida}")
    print(res[["gini_inf", "gini_medio", "gini_sup"]].describe())


if __name__ == "__main__":
    main()
