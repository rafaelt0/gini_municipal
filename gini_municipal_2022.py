"""
Estimativa do Índice de Gini municipal — Censo 2022 (dados agrupados)
======================================================================
Fonte primária:
  SIDRA Tabela 10296 — Moradores em domicílios particulares permanentes
  ocupados, por classes de rendimento domiciliar per capita em salários
  mínimos. Resultados preliminares da amostra (divulgação out/2025).

Calibração externa:
  PNAD Contínua 2022 (anual) via SIDRA — usado para estimar o parâmetro
  de forma Pareto α da cauda superior de renda por UF. Sem essa calibração
  o script usa α=1,5 (melhor estimativa da literatura para o Brasil),
  que produz média ~15 SM para a classe aberta ">5 SM" — contra α=2 (média
  10 SM) do método original, que subestima sistematicamente o topo.

Método: limites de Gastwirth (1972) via decomposição exata do Gini.
  Para grupos sem sobreposição: G = G_entre + Σ p_i·s_i·G_i (sem resíduo).
  - gini_inf : todos no ponto médio / média calibrada (G_intra_i = 0)
  - gini_sup : massa nos extremos com média preservada (G_intra_i máximo)
    • Classe fechada [li, ls] com média m:
        G_intra_max = (ls–m)·(m–li) / (m·(ls–li))
      Quando m = ponto médio isso simplifica para 0,25·(ls–li)/m.
    • Classe aberta (Pareto com parâmetro α):
        média = α·li/(α–1),  G_intra_max = 1/(2α–1)

Uso:
    pip install requests pandas
    python gini_municipal_2022.py               # todos os municípios
    python gini_municipal_2022.py --uf 35       # só São Paulo
    python gini_municipal_2022.py --alpha 1.5   # forçar α manual (sem PNAD)

Saída: gini_municipal_2022.csv
"""

import argparse
import math
import re
import sys
import time
import unicodedata

import pandas as pd
import requests

# ── Constantes ────────────────────────────────────────────────────────────────
TABELA_CENSO = 10296
ANO_CENSO = "2022"
SM_2022 = 1212.0  # salário mínimo de referência do Censo 2022 (R$)

IBGE_META_URL = "https://servicodados.ibge.gov.br/api/v3/agregados"
SIDRA_URL = "https://apisidra.ibge.gov.br/values"

CODIGOS_UF = [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29,
              31, 32, 33, 35, 41, 42, 43, 50, 51, 52, 53]

FRACOES = {"1/8": 0.125, "1/4": 0.25, "1/2": 0.5, "3/4": 0.75}

# Melhor estimativa de α para o Brasil baseada na literatura
# (PNAD 2022 implica α ≈ 1,4–1,6 segundo Souza 2016 e WID.world).
ALPHA_FALLBACK = 1.5


# ── Utilitários ───────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def _num(token: str) -> float:
    token = token.strip()
    if token in FRACOES:
        return FRACOES[token]
    try:
        return float(token.replace(",", "."))
    except ValueError:
        raise ValueError(
            f"Token '{token}' não reconhecido como número. "
            f"Se for fração, adicione ao dicionário FRACOES."
        )


def parse_classe(nome: str):
    """Converte rótulo de classe em (li, ls) em salários mínimos.

    ls = None para classe aberta. Retorna None para total/sem declaração.
    """
    n = _norm(nome)
    if "total" in n or "sem declaracao" in n:
        return None
    if "sem rendimento" in n:
        return (0.0, 0.0)
    # "Mais de X" — classe aberta (protege contra "mais de X a Y")
    m = re.match(r"mais de ([\d/,\.]+)\s*(?:salarios?\b|$)", n)
    if m and " a " not in n and " ate " not in n:
        return (_num(m.group(1)), None)
    # "X ou mais" — formato alternativo de classe aberta
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
    print(f"  [aviso] classe não reconhecida, ignorada: {nome!r}", file=sys.stderr)
    return None


# ── Estrutura do Censo ─────────────────────────────────────────────────────────

def descobrir_estrutura_censo():
    """Lê metadados da tabela 10296.

    Retorna (id_var, id_classif_renda, {cod: (nome, (li, ls))}, outras_classifs).
    """
    meta = requests.get(
        f"{IBGE_META_URL}/{TABELA_CENSO}/metadados", timeout=60
    ).json()

    id_var = None
    for v in meta["variaveis"]:
        nv = _norm(v["nome"])
        if "percentual" not in nv and "distribuicao" not in nv:
            id_var = v["id"]
            break
    if id_var is None:
        id_var = meta["variaveis"][0]["id"]

    id_renda, categorias, outras = None, {}, []
    for c in meta["classificacoes"]:
        nc = _norm(c["nome"])
        # Aceita se o nome da classificação menciona rendimento OU se suas
        # categorias contêm referências a salário mínimo (nome pode variar).
        cats_candidatas = {}
        for cat in c.get("categorias", []):
            b = parse_classe(cat["nome"])
            if b is not None:
                cats_candidatas[cat["id"]] = (cat["nome"], b)
        tem_renda_no_nome = "rendimento" in nc
        tem_sm_nas_cats = any(
            "salario" in _norm(cat["nome"]) or "sem rendimento" in _norm(cat["nome"])
            for cat in c.get("categorias", [])
        )
        if (tem_renda_no_nome or tem_sm_nas_cats) and len(cats_candidatas) >= 3:
            id_renda = c["id"]
            categorias = cats_candidatas
        else:
            outras.append(c["id"])

    if id_renda is None or not categorias:
        sys.exit(
            "Classificação de classes de rendimento não encontrada. "
            "Verifique: " + f"{IBGE_META_URL}/{TABELA_CENSO}/metadados"
        )
    if outras:
        print(
            f"  [aviso] {len(outras)} classificação(ões) adicional(is) "
            f"(ids: {outras}) omitida(s) da consulta.",
            file=sys.stderr,
        )
    return id_var, id_renda, categorias, outras


def baixar_uf_censo(uf, id_var, id_renda, cods_renda, _outras):
    """Baixa moradores por classe de renda para todos os municípios de uma UF."""
    classif = f"c{id_renda}/{','.join(str(c) for c in cods_renda)}"
    url = (
        f"{SIDRA_URL}/t/{TABELA_CENSO}/n6/in n3 {uf}"
        f"/v/{id_var}/p/{ANO_CENSO}/{classif}?formato=json"
    )
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    dados = r.json()
    if len(dados) <= 1:
        return pd.DataFrame()

    header = dados[0]
    df = pd.DataFrame(dados[1:])

    # "Município (Código)" vem antes de "Município" — excluir Codigo para não
    # confundir as duas colunas.
    col_cod = next(
        k for k, v in header.items() if "Codigo" in str(v) and "Munic" in str(v)
    )
    col_nome = next(
        k for k, v in header.items()
        if str(v).startswith("Munic") and "Codigo" not in str(v)
    )
    col_classe = next(
        k for k, v in header.items() if "rendimento" in _norm(str(v))
    )
    out = df[[col_cod, col_nome, col_classe, "V"]].copy()
    out.columns = ["cod_mun", "municipio", "classe", "valor"]
    out["valor"] = pd.to_numeric(out["valor"], errors="coerce")
    return out.dropna(subset=["valor"])


# ── Calibração PNAD Contínua ───────────────────────────────────────────────────

def _get_json(url, params=None, timeout=60):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _descobrir_tabela_pnad():
    """Procura no SIDRA uma tabela PNAD Contínua com distribuição de
    rendimento domiciliar per capita em SM ao nível UF (n3).

    Retorna (id_tab, id_var, id_classif, {cod: (nome, (li,ls))}) ou None.
    """
    try:
        tabelas = _get_json(IBGE_META_URL, params={"pesquisa": "CN"})
    except Exception as e:
        print(f"  [aviso] PNAD inacessível ({e})", file=sys.stderr)
        return None

    def _score(t):
        n = _norm(t.get("nome", ""))
        return (
            ("rendimento" in n) * 3
            + ("domiciliar" in n) * 2
            + ("per capita" in n or "percapita" in n) * 2
            + ("salario" in n or "minimo" in n) * 2
            + ("pessoa" in n or "morador" in n)
        )

    candidatas = sorted(
        [t for t in tabelas if _score(t) >= 4], key=_score, reverse=True
    )

    for tab in candidatas[:30]:
        try:
            meta = _get_json(f"{IBGE_META_URL}/{tab['id']}/metadados", timeout=30)
        except Exception:
            continue

        # Exige nível UF (N3)
        niveis = []
        for v in meta.get("nivelTerritorial", {}).values():
            if isinstance(v, list):
                niveis += [str(x).upper() for x in v]
        if "N3" not in niveis:
            continue

        # Procura classificação com classes SM
        id_classif, cats = None, {}
        for c in meta.get("classificacoes", []):
            nc = _norm(c.get("nome", ""))
            if "rendimento" in nc and ("salario" in nc or "minimo" in nc):
                for cat in c.get("categorias", []):
                    b = parse_classe(cat["nome"])
                    if b is not None:
                        cats[cat["id"]] = (cat["nome"], b)
                if len(cats) >= 3:
                    id_classif = c["id"]
                    break
        if not id_classif:
            continue

        # Variável de contagem de pessoas
        id_var = None
        for v in meta.get("variaveis", []):
            nv = _norm(v["nome"])
            if any(k in nv for k in ("pessoa", "morador", "populacao", "individuo")):
                id_var = v["id"]
                break
        if id_var is None and meta.get("variaveis"):
            id_var = meta["variaveis"][0]["id"]

        if id_var:
            print(f"  [PNAD] tabela {tab['id']}: {tab['nome'][:70]}")
            return tab["id"], id_var, id_classif, cats

    print("  [aviso] Nenhuma tabela PNAD compatível encontrada.", file=sys.stderr)
    return None


def _baixar_pnad_por_uf(id_tab, id_var, id_classif, cods_pnad):
    """Baixa população por classe de rendimento per capita (SM) por UF do PNAD."""
    # Tenta ano 2022; se falhar, tenta o período mais recente disponível
    for periodo in (ANO_CENSO, "last%201"):
        classif = f"c{id_classif}/{','.join(str(c) for c in cods_pnad)}"
        url = (
            f"{SIDRA_URL}/t/{id_tab}/n3/all"
            f"/v/{id_var}/p/{periodo}/{classif}?formato=json"
        )
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            dados = r.json()
            if len(dados) > 1:
                break
        except Exception:
            dados = []

    if len(dados) <= 1:
        return pd.DataFrame()

    header = dados[0]
    df = pd.DataFrame(dados[1:])

    # Identifica coluna de UF e de classe
    col_uf = next(
        (k for k, v in header.items()
         if any(w in _norm(str(v)) for w in ("unidade", "estado", "uf"))
         and "codigo" in _norm(str(v))),
        None,
    )
    if col_uf is None:
        # fallback: primeira coluna com "Codigo"
        col_uf = next(
            (k for k, v in header.items() if "Codigo" in str(v)), list(header)[0]
        )
    col_classe = next(
        k for k, v in header.items() if "rendimento" in _norm(str(v))
    )
    out = df[[col_uf, col_classe, "V"]].copy()
    out.columns = ["cod_uf", "classe", "valor"]
    out["valor"] = pd.to_numeric(out["valor"], errors="coerce")
    return out.dropna(subset=["valor"])


def _ajustar_alpha_pareto(dist_pnad, li_topo):
    """Estima α de Pareto da cauda superior usando o estimador de Hill
    adaptado para dados agrupados.

    dist_pnad: [(li, ls, pop)] com as classes do PNAD para uma UF.
    Retorna α estimado ou None se dados insuficientes.

    Fundamento: para Pareto(α, x_m=li_topo), P(X > x) = (li_topo/x)^α.
    Dados os pontos de corte x_1 < x_2 < … acima de li_topo e suas
    frequências acumuladas P_k = P(X > x_k | X > li_topo):
        ln(P_k) = –α · ln(x_k / li_topo)
    Ajusta por OLS sem intercepto (passa pela origem por construção).
    """
    # Filtra classes com li >= li_topo (classes da cauda superior)
    cauda = sorted(
        [(li, ls, pop) for li, ls, pop in dist_pnad if li >= li_topo * 0.99],
        key=lambda x: x[0],
    )
    if len(cauda) < 2:
        return None  # precisa de pelo menos 2 pontos para ajustar

    pop_total_cauda = sum(p for _, _, p in cauda)
    if pop_total_cauda <= 0:
        return None

    # Pontos de corte: limite superior de cada classe fechada acima de li_topo
    pontos = []  # (x_k, P_k)
    pop_acumulada = 0.0
    for li, ls, pop in cauda:
        if ls is None:
            break  # classe aberta — não tem corte superior definido
        pop_acumulada += pop
        x_k = ls
        p_k = 1.0 - pop_acumulada / pop_total_cauda  # P(X > x_k | X > li_topo)
        if 0 < p_k < 1 and x_k > li_topo:
            pontos.append((x_k, p_k))

    if not pontos:
        return None

    # OLS sem intercepto: ln(P_k) = –α · ln(x_k/li_topo)
    soma_xy = sum(math.log(x / li_topo) * (-math.log(p)) for x, p in pontos)
    soma_xx = sum(math.log(x / li_topo) ** 2 for x, p in pontos)
    if soma_xx == 0:
        return None
    alpha_est = soma_xy / soma_xx

    if 1.01 < alpha_est < 10.0:
        return alpha_est
    return None


def calibrar_alpha_por_uf(li_topo, ufs, alpha_fallback):
    """Usa PNAD Contínua para estimar α de Pareto por UF.

    Retorna {str(cod_uf): alpha}. Para UFs sem dados usa alpha_fallback.
    """
    print("\nCalibrando parâmetro Pareto via PNAD Contínua...")
    resultado_pnad = _descobrir_tabela_pnad()
    if resultado_pnad is None:
        print(f"  Usando α = {alpha_fallback} (fallback) para todas as UFs.")
        return {str(uf): alpha_fallback for uf in ufs}

    id_tab, id_var, id_classif, cats_pnad = resultado_pnad
    try:
        df_pnad = _baixar_pnad_por_uf(id_tab, id_var, id_classif, list(cats_pnad.keys()))
    except Exception as e:
        print(f"  [aviso] Download PNAD falhou ({e}); usando fallback.", file=sys.stderr)
        return {str(uf): alpha_fallback for uf in ufs}

    if df_pnad.empty:
        print("  [aviso] PNAD sem dados; usando fallback.", file=sys.stderr)
        return {str(uf): alpha_fallback for uf in ufs}

    # Reconstrói bounds do PNAD: tenta por código de categoria primeiro
    nome_para_bounds = {nome: b for _, (nome, b) in cats_pnad.items()}
    cod_para_bounds = {str(cod): b for cod, (_, b) in cats_pnad.items()}

    alphas = {}
    for cod_uf, grupo in df_pnad.groupby("cod_uf"):
        dist = []
        for _, row in grupo.iterrows():
            b = (cod_para_bounds.get(str(row["classe"]))
                 or nome_para_bounds.get(row["classe"])
                 or parse_classe(str(row["classe"])))
            if b is None or row["valor"] <= 0:
                continue
            dist.append((b[0], b[1], row["valor"]))

        alpha_uf = _ajustar_alpha_pareto(dist, li_topo)
        if alpha_uf is None:
            alpha_uf = alpha_fallback
        alphas[str(cod_uf)] = alpha_uf

    n_ok = sum(1 for a in alphas.values() if abs(a - alpha_fallback) > 0.01)
    print(f"  α estimado via PNAD em {n_ok}/{len(alphas)} UFs "
          f"(demais usam fallback {alpha_fallback}).")

    # Garante todas as UFs solicitadas
    for uf in ufs:
        alphas.setdefault(str(uf), alpha_fallback)

    return alphas


# ── Cálculo do Gini ────────────────────────────────────────────────────────────

def gini_agrupado(classes, alpha_topo=None):
    """Limites de Gastwirth do Gini a partir de [(li, ls, pop)], em SM.

    alpha_topo: parâmetro de forma Pareto para a classe aberta superior.
      Se None, usa ALPHA_FALLBACK.

    Decomposição exata (grupos sem sobreposição):
      G = G_entre + Σ p_i · s_i · G_intra_i
    onde:
      G_entre  = Gini com todos no ponto médio / média calibrada  → gini_inf
      G_intra_i = máximo dado a média calibrada e suporte [li, ls] → gini_sup

    Para classe fechada [li, ls] com média m:
      G_intra_max = (ls – m)(m – li) / (m · (ls – li))
    Para classe aberta (Pareto com α):
      média = α · li / (α – 1),  G_intra = 1 / (2α – 1)
    """
    if alpha_topo is None:
        alpha_topo = ALPHA_FALLBACK

    dados = []  # (media, pop, g_intra_max)
    for li, ls, pop in classes:
        if pop <= 0:
            continue
        if ls is None:
            media = alpha_topo * li / (alpha_topo - 1)
            g_max = 1.0 / (2.0 * alpha_topo - 1.0)
        else:
            media = (li + ls) / 2.0
            if media > 0:
                # Fórmula geral: G_intra_max para 2-pontos nos extremos com média m
                # Quando m = ponto médio: simplifica para 0,25·(ls–li)/m
                g_max = (ls - media) * (media - li) / (media * (ls - li))
            else:
                g_max = 0.0
        dados.append((media, pop, g_max))

    if not dados:
        return None

    dados.sort()  # ordena por média crescente (= ordem de renda)
    pop_total = sum(p for _, p, _ in dados)
    renda_total = sum(m * p for m, p, _ in dados)
    if renda_total == 0:
        return (0.0, 0.0)

    # Curva de Lorenz por trapézios.
    # g_entre = Σ p_i·(L_i + L_{i-1}) = 2·Área_Lorenz → G = 1 – g_entre
    g_entre = 0.0
    g_intra = 0.0
    P_ant = L_ant = 0.0
    for media, pop, g_max in dados:
        p_i = pop / pop_total
        s_i = (media * pop) / renda_total
        P = P_ant + p_i
        L = L_ant + s_i
        g_entre += p_i * (L + L_ant)
        g_intra += p_i * s_i * g_max
        P_ant, L_ant = P, L

    g_inf = 1.0 - g_entre
    return (g_inf, min(g_inf + g_intra, 1.0))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Gini municipal 2022 com calibração PNAD Contínua."
    )
    ap.add_argument("--uf", type=int, default=None, help="Código IBGE da UF (ex.: 35)")
    ap.add_argument("--saida", default="gini_municipal_2022.csv")
    ap.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=(
            "Parâmetro Pareto α fixo para a classe aberta "
            "(pula a calibração PNAD). Padrão: calibrar via PNAD ou usar 1,5."
        ),
    )
    args = ap.parse_args()

    # ── 1. Estrutura do Censo ────────────────────────────────────────────────
    print("Lendo metadados da tabela 10296 (Censo 2022)...")
    id_var, id_renda, categorias, outras = descobrir_estrutura_censo()
    print(f"  variável={id_var}, classif_renda={id_renda}, "
          f"{len(categorias)} classes")
    for cod, (nome, (li, ls)) in sorted(categorias.items()):
        print(f"    [{cod}] {nome} → ({li}, {ls})")

    # Identifica limite inferior da classe aberta
    li_topo = max(
        li for _, (_, (li, ls)) in categorias.items() if ls is None
    )

    # ── 2. Calibração PNAD ──────────────────────────────────────────────────
    ufs = [args.uf] if args.uf else CODIGOS_UF

    if args.alpha is not None:
        print(f"\nUsando α = {args.alpha} (fixado via --alpha).")
        alpha_por_uf = {str(uf): args.alpha for uf in ufs}
    else:
        alpha_por_uf = calibrar_alpha_por_uf(li_topo, ufs, ALPHA_FALLBACK)

    alphas_unicos = sorted(set(round(a, 3) for a in alpha_por_uf.values()))
    medias_topo = {a: round(a * li_topo / (a - 1), 2) for a in alphas_unicos}
    print(f"\n  Classe aberta (>{li_topo} SM): "
          f"α ∈ {alphas_unicos[:5]}{'…' if len(alphas_unicos) > 5 else ''}, "
          f"médias → {medias_topo}")

    # ── 3. Download Censo por UF ─────────────────────────────────────────────
    print("\nBaixando dados do Censo 2022...")
    frames = []
    ufs_falhas = []
    for uf in ufs:
        print(f"  UF {uf}...")
        for tentativa in range(3):
            try:
                frames.append(
                    baixar_uf_censo(uf, id_var, id_renda,
                                    list(categorias.keys()), outras)
                )
                break
            except Exception as e:
                print(f"    erro ({e}); retry {tentativa+1}/3", file=sys.stderr)
                time.sleep(5 * (tentativa + 1))
        else:
            print(f"    [erro] UF {uf} falhou permanentemente.", file=sys.stderr)
            ufs_falhas.append(uf)
        time.sleep(1)

    if not frames:
        sys.exit("Nenhum dado baixado. Verifique conexão ou parâmetros.")
    if ufs_falhas:
        print(f"[aviso] UFs sem dados: {ufs_falhas}", file=sys.stderr)

    df = pd.concat(frames, ignore_index=True)

    # Mapeia nome de categoria → bounds (fallback para parse_classe)
    bounds_por_nome = {nome: b for _, (nome, b) in categorias.items()}

    # ── 4. Extrai UF do código de município (2 primeiros dígitos) ──────────
    df["cod_uf"] = df["cod_mun"].astype(str).str[:2]

    # ── 5. Calcula Gini por município ────────────────────────────────────────
    print("\nCalculando Gini por município...")
    resultados = []
    for (cod, nome, cod_uf), grupo in df.groupby(
        ["cod_mun", "municipio", "cod_uf"]
    ):
        alpha_uf = alpha_por_uf.get(str(cod_uf), ALPHA_FALLBACK)

        classes = []
        for _, row in grupo.iterrows():
            b = bounds_por_nome.get(row["classe"]) or parse_classe(row["classe"])
            if b is None:
                continue
            classes.append((b[0], b[1], row["valor"]))

        res = gini_agrupado(classes, alpha_topo=alpha_uf)
        pop = sum(c[2] for c in classes)

        if res is None:
            g_inf = g_sup = g_med = None
        else:
            g_inf, g_sup = res
            g_med = (g_inf + g_sup) / 2.0

        resultados.append({
            "cod_mun": cod,
            "municipio": nome,
            "cod_uf": cod_uf,
            "alpha_pareto": round(alpha_uf, 4),
            "gini_inf": round(g_inf, 4) if g_inf is not None else None,
            "gini_sup": round(g_sup, 4) if g_sup is not None else None,
            "gini_medio": round(g_med, 4) if g_med is not None else None,
            "populacao_considerada": int(round(pop)),
        })

    res = pd.DataFrame(resultados).sort_values("cod_mun")
    res.to_csv(args.saida, index=False, encoding="utf-8")
    print(f"\n{len(res)} municípios → {args.saida}")
    print(res[["gini_inf", "gini_medio", "gini_sup"]].describe().round(4))


if __name__ == "__main__":
    main()
