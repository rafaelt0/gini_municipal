"""
Estimativa do Índice de Gini municipal — Censo 2022 (dados agrupados)
======================================================================
Fonte primária:
  SIDRA Tabela 10296 — Moradores em domicílios particulares permanentes
  ocupados, por classes de rendimento domiciliar per capita em salários
  mínimos. Resultados preliminares da amostra (divulgação out/2025).

Calibração externa:
  PNAD Contínua 2022 (anual) via SIDRA — estimativa do parâmetro de forma
  Pareto α da cauda superior de renda por UF.

Método: limites de Gastwirth (1972) + estimativa pontual lognormal.

  Gastwirth bounds (colunas gini_inf / gini_sup):
    G = G_entre + Σ p_i·s_i·G_i  (decomposição exata, grupos sem sobreposição)
    - gini_inf : médias condicionais lognormais por faixa (quando MLE converge)
                 ou ponto médio da faixa (fallback). G_intra_i = 0.
    - gini_sup : mesma média + dispersão intra-faixa máxima.
      • Faixa fechada [li, ls] com média m: G_max = (ls−m)(m−li) / (m(ls−li))
      • Faixa aberta (Pareto α):            G_max = 1/(2α−1)

  Estimativa pontual lognormal (coluna gini_lognormal):
    Ajuste MLE de lognormal(μ, σ) aos dados agrupados de renda positiva.
    Gini = 1 − 2(1−p_zero)² Φ(−σ/√2)
    onde p_zero é a fração de domicílios sem rendimento.

  Distribuições alternativas (colunas gini_gamma, gini_singh_maddala):
    Além da lognormal, ajusta por MLE Gamma(α, β) e Singh-Maddala(a, b, q).
    Gini misto para qualquer distribuição: G = p_zero + (1−p_zero) · G_+
    • Gamma: G_+ = Γ(α+½) / (√π · Γ(α+1)) — fórmula analítica
    • Singh-Maddala: G_+ via G = (2/μ)·∫x·F(x)·f(x)dx − 1 — integração numérica
    Fórmula geral vale para qualquer distribuição: derivada da curva de Lorenz.

  Comparação por AIC (colunas aic_lognormal, aic_gamma, aic_singh_maddala):
    AIC = 2k − 2·LL  onde k = número de parâmetros (2, 2, 3) e LL = Σᵢ nᵢ log P̂ᵢ.
    Coluna melhor_dist indica a distribuição com menor AIC.

  Intervalos de confiança 95% (colunas gini_ic_low, gini_ic_high, gini_se):
    Método delta aplicado ao estimador MLE lognormal.
    A matriz de covariância assintótica é obtida invertendo a Hessiana
    numérica da log-verossimilhança normalizada avaliada no MLE:
      Cov(μ̂, log σ̂) ≈ (1/n) · H_code⁻¹
    O gradiente de G em relação a (μ, log σ) é:
      dG/d(log σ) = 2(1−p0)² φ(σ/√2) · σ/√2  (dG/dμ = 0)
    IC 95%: Ĝ ± 1,96 · SE,  SE = √[Var(G)]
    Nota: p_zero tratado como quantidade observada (não parâmetro).

  Qualidade do ajuste (coluna ln_mad):
    Desvio absoluto médio entre proporções observadas e preditas pelo
    lognormal por faixa. Valores > 0,04 indicam ajuste insatisfatório.

  Consistência com os bounds (coluna ln_outside_bounds):
    True quando gini_lognormal > gini_sup. Ocorre porque o lognormal
    tem suporte em (0,∞) enquanto os bounds assumem suporte dentro de
    cada faixa. São estimadores com premissas distintas; não implica erro.

  Análise de sensibilidade ao α (padrão: α ∈ {1,2; 1,5; 2,0}):
    Colunas gini_inf_a{α} e gini_sup_a{α} sempre presentes no CSV.

  Rastreamento de exclusões (coluna pct_sem_declaracao):
    Fração de domicílios sem declaração de rendimento, incluída na
    consulta SIDRA e contabilizada separadamente no cálculo.

  Validação externa (coluna gini_2010):
    Gini municipal 2010 do Atlas do Desenvolvimento Humano via IPEADATA
    (série ADH_GINI). Permite comparação histórica e teste de validade.

Uso:
    pip install requests pandas scipy numpy
    python gini_municipal_2022.py                           # todos os municípios
    python gini_municipal_2022.py --uf 35                   # só São Paulo
    python gini_municipal_2022.py --alpha 1.5               # forçar α manual
    python gini_municipal_2022.py --alpha-values 1.2,1.5,2.0,2.5
    python gini_municipal_2022.py --sem-validacao-2010      # pula IPEADATA

Saída: gini_municipal_2022.csv
"""

import argparse
import math
import re
import sys
import time
import unicodedata

import numpy as np
import pandas as pd
import requests
from scipy import integrate as _integrate
from scipy.optimize import minimize
from scipy.special import gammaln as _gammaln
from scipy.stats import gamma as _gamma_dist
from scipy.stats import burr12 as _sm_dist
from scipy.stats import norm as _norm

# ── Constantes ────────────────────────────────────────────────────────────────
TABELA_CENSO = 10296
ANO_CENSO = "2022"
SM_2022 = 1212.0  # salário mínimo de referência do Censo 2022 (R$)

IBGE_META_URL = "https://servicodados.ibge.gov.br/api/v3/agregados"
SIDRA_URL = "https://apisidra.ibge.gov.br/values"
IPEADATA_URL = "http://www.ipeadata.gov.br/api/odata4"

CODIGOS_UF = [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29,
              31, 32, 33, 35, 41, 42, 43, 50, 51, 52, 53]

FRACOES = {"1/8": 0.125, "1/4": 0.25, "1/2": 0.5, "3/4": 0.75}
ALPHA_FALLBACK = 1.5
ALPHA_SENSITIVITY = [1.2, 1.5, 2.0]  # sempre incluídos na saída
LN_MAD_THRESHOLD = 0.04              # acima desse MAD o ajuste lognormal é ruim


# ── Utilitários ───────────────────────────────────────────────────────────────

def _ascii(s: str) -> str:
    """Normaliza string para ASCII minúsculo (comparação de rótulos IBGE)."""
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
    n = _ascii(nome)
    if "total" in n or "sem declaracao" in n:
        return None
    if "sem rendimento" in n:
        return (0.0, 0.0)
    m = re.match(r"mais de ([\d/,\.]+)\s*(?:salarios?\b|$)", n)
    if m and " a " not in n and " ate " not in n:
        return (_num(m.group(1)), None)
    m = re.match(r"([\d/,\.]+)\s*ou mais", n)
    if m:
        return (_num(m.group(1)), None)
    m = re.search(r"mais de ([\d/,\.]+)\s*(?:a|ate)\s*([\d/,\.]+)", n)
    if m:
        return (_num(m.group(1)), _num(m.group(2)))
    m = re.search(r"ate ([\d/,\.]+)", n)
    if m:
        return (0.0, _num(m.group(1)))
    print(f"  [aviso] classe não reconhecida, ignorada: {nome!r}", file=sys.stderr)
    return None


# ── Lognormal: funções auxiliares ─────────────────────────────────────────────

def _bracket_prob(li, ls, mu, sigma):
    """Probabilidade lognormal(μ,σ) de cair na faixa (li, ls)."""
    if ls is None:
        z_lo = (math.log(max(li, 1e-9)) - mu) / sigma
        return 1.0 - _norm.cdf(z_lo)
    elif li <= 0:
        z_hi = (math.log(max(ls, 1e-9)) - mu) / sigma
        return _norm.cdf(z_hi)
    else:
        z_lo = (math.log(li) - mu) / sigma
        z_hi = (math.log(ls) - mu) / sigma
        return _norm.cdf(z_hi) - _norm.cdf(z_lo)


def _neg_ll_lognormal(params, dados, total):
    """Negativa da log-verossimilhança normalizada de lognormal agrupada.

    params = [mu, log_sigma]; total = soma das populações (normalização).
    """
    mu, log_sigma = params
    sigma = math.exp(log_sigma)
    ll = sum(pop * math.log(max(_bracket_prob(li, ls, mu, sigma), 1e-12))
             for li, ls, pop in dados)
    return -ll / total


def ic_delta_lognormal(mu_fit, sigma_fit, p_zero, classes_pos, z=1.96):
    """IC (z*100)% para gini_lognormal pelo método delta com Hessiana numérica.

    Aplica o método delta à parametrização (μ, log σ) do MLE:
      G = 1 − 2(1−p0)² Φ(−σ/√2)
      dG/d(log σ) = 2(1−p0)² φ(σ/√2) · σ/√2
      Var(G) ≈ (1/n) · ∇G^T · H_code⁻¹ · ∇G

    Retorna (ic_low, ic_high, se) ou None se a Hessiana for singular.
    """
    dados = [(li, ls, pop) for li, ls, pop in classes_pos if pop > 0]
    total = sum(p for _, _, p in dados)
    if total <= 0 or len(dados) < 2:
        return None

    params_hat = [mu_fit, math.log(sigma_fit)]
    eps = 1e-5

    # Hessiana numérica 2×2 por diferenças centrais mistas
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            xpp = list(params_hat); xpp[i] += eps; xpp[j] += eps
            xpm = list(params_hat); xpm[i] += eps; xpm[j] -= eps
            xmp = list(params_hat); xmp[i] -= eps; xmp[j] += eps
            xmm = list(params_hat); xmm[i] -= eps; xmm[j] -= eps
            H[i, j] = (_neg_ll_lognormal(xpp, dados, total)
                       - _neg_ll_lognormal(xpm, dados, total)
                       - _neg_ll_lognormal(xmp, dados, total)
                       + _neg_ll_lognormal(xmm, dados, total)) / (4.0 * eps * eps)

    # Rejeita Hessiana não-positiva-definida (sinal de mínimo local espúrio)
    eigvals = np.linalg.eigvalsh(H)
    if eigvals.min() <= 0:
        return None

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None

    # Gradiente de G em relação a (μ, log σ)
    # G = 1 − 2(1−p0)Φ(−σ/√2)  →  dG/d(log σ) = 2(1−p0)φ(σ/√2)·σ/√2
    dG_dlogsigma = (2.0 * (1.0 - p_zero)
                    * _norm.pdf(sigma_fit / math.sqrt(2.0))
                    * sigma_fit / math.sqrt(2.0))
    grad = np.array([0.0, dG_dlogsigma])

    # Cov(θ̂) = (1/n) H_code⁻¹  →  Var(G) = (1/n) grad^T H_code⁻¹ grad
    var_G = float(grad @ H_inv @ grad) / total
    if var_G <= 0:
        return None

    se_G = math.sqrt(var_G)
    gini_val = gini_lognormal_mixed(mu_fit, sigma_fit, p_zero)
    return (max(0.0, gini_val - z * se_G), min(1.0, gini_val + z * se_G), se_G)


# ── Lognormal MLE sobre dados agrupados ───────────────────────────────────────

def fit_lognormal_grouped(classes_pos):
    """Ajusta lognormal(μ, σ) a dados agrupados de renda positiva por MLE multinomial.

    classes_pos: list of (li, ls, pop) excluindo "sem rendimento" (li=ls=0).
    Retorna (mu, sigma, converged) ou None se dados insuficientes ou ajuste falha.
    """
    dados = [(li, ls, pop) for li, ls, pop in classes_pos if pop > 0]
    if len(dados) < 2:
        return None
    total = sum(p for _, _, p in dados)
    if total <= 0:
        return None

    def neg_ll(params):
        return _neg_ll_lognormal(params, dados, total)

    log_mids, weights = [], []
    for li, ls, pop in dados:
        if ls is None:
            mid = li * 1.5
        elif li <= 0:
            mid = max(ls / 3.0, 1e-9)
        else:
            mid = math.sqrt(li * ls)
        if mid > 0:
            log_mids.append(math.log(mid))
            weights.append(pop)

    if not log_mids:
        return None
    w_total = sum(weights)
    mu0 = sum(l * w for l, w in zip(log_mids, weights)) / w_total
    var0 = sum(w * (l - mu0) ** 2 for l, w in zip(log_mids, weights)) / w_total
    sigma0 = max(math.sqrt(var0), 0.3)

    try:
        result = minimize(
            neg_ll,
            [mu0, math.log(sigma0)],
            method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000},
        )
        mu_fit = result.x[0]
        sigma_fit = math.exp(result.x[1])
        if 0.05 < sigma_fit < 6.0:
            return mu_fit, sigma_fit, result.success
    except Exception:
        pass
    return None


def lognormal_fit_mad(classes_pos, mu, sigma):
    """Desvio absoluto médio entre proporções observadas e preditas pelo lognormal.

    Medida de qualidade do ajuste por faixa. MAD > 0,04 indica ajuste ruim.
    """
    dados = [(li, ls, pop) for li, ls, pop in classes_pos if pop > 0]
    if not dados:
        return None
    total = sum(p for _, _, p in dados)
    if total <= 0:
        return None

    desvios = [abs(pop / total - max(_bracket_prob(li, ls, mu, sigma), 1e-12))
               for li, ls, pop in dados]
    return sum(desvios) / len(desvios)


def lognormal_conditional_mean(li, ls, mu, sigma):
    """Média condicional de lognormal(μ, σ) em [li, ls] (ls=None: classe aberta).

    Faixa fechada [a,b]: E[X|a<X≤b] = exp(μ+σ²/2)·[Φ(z_b−σ)−Φ(z_a−σ)] / [Φ(z_b)−Φ(z_a)]
    Classe aberta [a,∞): E[X|X>a]   = exp(μ+σ²/2)·[1−Φ(z_a−σ)] / [1−Φ(z_a)]
    """
    exp_mu_sig = math.exp(mu + sigma ** 2 / 2.0)
    if ls is None:
        z_lo = (math.log(max(li, 1e-9)) - mu) / sigma
        denom = 1.0 - _norm.cdf(z_lo)
        if denom < 1e-12:
            return exp_mu_sig
        return exp_mu_sig * (1.0 - _norm.cdf(z_lo - sigma)) / denom
    elif li <= 0:
        z_hi = (math.log(max(ls, 1e-9)) - mu) / sigma
        denom = _norm.cdf(z_hi)
        if denom < 1e-12:
            return ls / 2.0
        return exp_mu_sig * _norm.cdf(z_hi - sigma) / denom
    else:
        z_lo = (math.log(li) - mu) / sigma
        z_hi = (math.log(ls) - mu) / sigma
        denom = _norm.cdf(z_hi) - _norm.cdf(z_lo)
        if denom < 1e-12:
            return math.sqrt(li * ls)
        num = _norm.cdf(z_hi - sigma) - _norm.cdf(z_lo - sigma)
        return exp_mu_sig * num / denom


def gini_lognormal_mixed(mu, sigma, p_zero):
    """Gini de distribuição mista: fração p_zero com renda zero, resto lognormal(μ,σ).

    Fórmula geral para mistura com massa em zero (derivação via curva de Lorenz):
      G = p_zero + (1−p_zero) · G_lognormal(σ)
        = 1 − 2(1−p_zero) · Φ(−σ/√2)
    Reduz a 2Φ(σ/√2)−1 quando p_zero=0 (Gini do lognormal puro).
    """
    return 1.0 - 2.0 * (1.0 - p_zero) * _norm.cdf(-sigma / math.sqrt(2.0))


def gini_mixed(G_plus, p_zero):
    """Gini de mistura com fração p_zero de zeros e distribuição positiva com Gini G_plus.

    G = p_zero + (1−p_zero) · G_plus
    Derivado da curva de Lorenz: L_mista(p) = L_+(（p−p_zero)/(1−p_zero)) para p > p_zero.
    """
    return p_zero + (1.0 - p_zero) * G_plus


# ── Distribuição Gamma ────────────────────────────────────────────────────────

def gini_gamma_shape(alpha):
    """Gini analítico da Gamma(α, β): G = Γ(α+½) / (√π · Γ(α+1)).

    Invariante ao parâmetro de escala β.
    """
    return math.exp(_gammaln(alpha + 0.5) - 0.5 * math.log(math.pi) - _gammaln(alpha + 1))


def _bracket_prob_gamma(li, ls, alpha, beta):
    """P(li < X ≤ ls) para X ~ Gamma(α, scale=β)."""
    if ls is None:
        return 1.0 - _gamma_dist.cdf(max(li, 1e-9), a=alpha, scale=beta)
    elif li <= 0:
        return _gamma_dist.cdf(ls, a=alpha, scale=beta)
    else:
        return _gamma_dist.cdf(ls, a=alpha, scale=beta) - _gamma_dist.cdf(li, a=alpha, scale=beta)


def fit_gamma_grouped(classes_pos):
    """MLE de Gamma(α, β) para dados agrupados de renda positiva.

    Retorna (alpha, beta, converged, neg_ll_min) ou None se ajuste falha.
    """
    dados = [(li, ls, pop) for li, ls, pop in classes_pos if pop > 0]
    if len(dados) < 2:
        return None
    total = sum(p for _, _, p in dados)
    if total <= 0:
        return None

    def neg_ll(params):
        log_alpha, log_beta = params
        alpha, beta = math.exp(log_alpha), math.exp(log_beta)
        ll = sum(pop * math.log(max(_bracket_prob_gamma(li, ls, alpha, beta), 1e-12))
                 for li, ls, pop in dados)
        return -ll / total

    # Inicialização via método dos momentos nos pontos médios aritméticos
    mids, weights = [], []
    for li, ls, pop in dados:
        if ls is None:
            mid = li * 1.5
        elif li <= 0:
            mid = max(ls / 3.0, 1e-9)
        else:
            mid = (li + ls) / 2.0
        mids.append(mid); weights.append(pop)
    w = sum(weights)
    m1 = sum(m * ww for m, ww in zip(mids, weights)) / w
    m2 = sum(m**2 * ww for m, ww in zip(mids, weights)) / w
    var = max(m2 - m1**2, 1e-6)
    alpha0 = max(0.5, m1**2 / var)
    beta0 = max(1e-3, var / m1)

    try:
        result = minimize(
            neg_ll,
            [math.log(alpha0), math.log(beta0)],
            method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000},
        )
        alpha_fit = math.exp(result.x[0])
        beta_fit = math.exp(result.x[1])
        if 0.05 < alpha_fit < 500.0 and 1e-5 < beta_fit < 1e6:
            return alpha_fit, beta_fit, result.success, result.fun
    except Exception:
        pass
    return None


def mad_grouped(classes_pos, prob_fn):
    """MAD entre proporções observadas e preditas pela função prob_fn(li, ls)."""
    dados = [(li, ls, pop) for li, ls, pop in classes_pos if pop > 0]
    if not dados:
        return None
    total = sum(p for _, _, p in dados)
    if total <= 0:
        return None
    return sum(abs(pop / total - max(prob_fn(li, ls), 1e-12)) for li, ls, pop in dados) / len(dados)


# ── Distribuição Singh-Maddala (Burr XII) ─────────────────────────────────────

def gini_sm_numerical(a, q):
    """Gini da Singh-Maddala F(x)=1−(1+x^a)^{−q} via G = (2/μ)·∫x·F(x)·f(x)dx − 1.

    Invariante ao parâmetro de escala b: calculado com b=1.
    Requer q > 1/a para que a média exista.
    """
    if q <= 1.0 / a:
        return None
    mean = _sm_dist.mean(c=a, d=q, scale=1.0)
    if mean <= 0 or not math.isfinite(mean):
        return None

    def integrand(x):
        return x * _sm_dist.cdf(x, c=a, d=q) * _sm_dist.pdf(x, c=a, d=q)

    try:
        result, _ = _integrate.quad(integrand, 0, mean * 100, limit=150)
        G = 2.0 * result / mean - 1.0
        if 0.0 < G < 1.0:
            return G
    except Exception:
        pass
    return None


def _bracket_prob_sm(li, ls, a, b, q):
    """P(li < X ≤ ls) para X ~ Singh-Maddala(a, b, q) = Burr XII(c=a, d=q, scale=b)."""
    if ls is None:
        return 1.0 - _sm_dist.cdf(max(li, 1e-9), c=a, d=q, scale=b)
    elif li <= 0:
        return _sm_dist.cdf(ls, c=a, d=q, scale=b)
    else:
        return _sm_dist.cdf(ls, c=a, d=q, scale=b) - _sm_dist.cdf(li, c=a, d=q, scale=b)


def fit_sm_grouped(classes_pos):
    """MLE de Singh-Maddala(a, b, q) para dados agrupados de renda positiva.

    Retorna (a, b, q, converged, neg_ll_min) ou None se ajuste falha.
    Exige q > 1/a para existência da média.
    """
    dados = [(li, ls, pop) for li, ls, pop in classes_pos if pop > 0]
    if len(dados) < 3:
        return None
    total = sum(p for _, _, p in dados)
    if total <= 0:
        return None

    def neg_ll(params):
        log_a, log_b, log_q = params
        a, b, q = math.exp(log_a), math.exp(log_b), math.exp(log_q)
        if q <= 1.1 / a:
            return 1e10
        ll = sum(pop * math.log(max(_bracket_prob_sm(li, ls, a, b, q), 1e-12))
                 for li, ls, pop in dados)
        return -ll / total

    try:
        result = minimize(
            neg_ll,
            [math.log(2.0), math.log(1.0), math.log(2.0)],
            method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 3000},
        )
        a_f = math.exp(result.x[0])
        b_f = math.exp(result.x[1])
        q_f = math.exp(result.x[2])
        if (0.2 < a_f < 50.0 and 1e-5 < b_f < 1e6
                and 0.1 < q_f < 100.0 and q_f > 1.1 / a_f):
            return a_f, b_f, q_f, result.success, result.fun
    except Exception:
        pass
    return None


# ── Estrutura do Censo ─────────────────────────────────────────────────────────

def descobrir_estrutura_censo():
    """Lê metadados da tabela 10296.

    Retorna (id_var, id_classif_renda, categorias, outras_classifs, cods_sem_dec).
    cods_sem_dec: IDs de categoria "sem declaração" para inclusão explícita na query.
    """
    meta = requests.get(
        f"{IBGE_META_URL}/{TABELA_CENSO}/metadados", timeout=60
    ).json()

    id_var = None
    for v in meta["variaveis"]:
        nv = _ascii(v["nome"])
        if "percentual" not in nv and "distribuicao" not in nv:
            id_var = v["id"]
            break
    if id_var is None:
        id_var = meta["variaveis"][0]["id"]

    id_renda, categorias, outras, cods_sem_dec = None, {}, [], []
    for c in meta["classificacoes"]:
        nc = _ascii(c["nome"])
        cats_candidatas = {}
        sem_dec_candidatos = []
        for cat in c.get("categorias", []):
            b = parse_classe(cat["nome"])
            if b is not None:
                cats_candidatas[cat["id"]] = (cat["nome"], b)
            elif "sem declaracao" in _ascii(cat["nome"]):
                sem_dec_candidatos.append(cat["id"])

        tem_renda_no_nome = "rendimento" in nc
        tem_sm_nas_cats = any(
            "salario" in _ascii(cat["nome"]) or "sem rendimento" in _ascii(cat["nome"])
            for cat in c.get("categorias", [])
        )
        if (tem_renda_no_nome or tem_sm_nas_cats) and len(cats_candidatas) >= 3:
            id_renda = c["id"]
            categorias = cats_candidatas
            cods_sem_dec = sem_dec_candidatos
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
    if cods_sem_dec:
        print(f"  [info] {len(cods_sem_dec)} categoria(s) 'sem declaração' "
              f"incluída(s) na consulta (ids: {cods_sem_dec}).")
    else:
        print("  [aviso] Nenhuma categoria 'sem declaração' encontrada nos metadados.",
              file=sys.stderr)

    return id_var, id_renda, categorias, outras, cods_sem_dec


def baixar_uf_censo(uf, id_var, id_renda, cods_renda, _outras, cods_sem_dec=None):
    """Baixa moradores por classe de renda para todos os municípios de uma UF.

    Inclui categorias 'sem declaração' na query para rastreamento correto.
    """
    todos_cods = list(cods_renda) + (cods_sem_dec or [])
    classif = f"c{id_renda}/{','.join(str(c) for c in todos_cods)}"
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

    col_cod = next(
        k for k, v in header.items()
        if "codigo" in _ascii(str(v)) and "munic" in _ascii(str(v))
    )
    col_nome = next(
        k for k, v in header.items()
        if _ascii(str(v)).startswith("munic") and "codigo" not in _ascii(str(v))
    )
    col_classe = next(
        k for k, v in header.items()
        if "rendimento" in _ascii(str(v)) and "codigo" not in _ascii(str(v))
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
    """
    try:
        tabelas = _get_json(IBGE_META_URL, params={"pesquisa": "CN"})
    except Exception as e:
        print(f"  [aviso] PNAD inacessível ({e})", file=sys.stderr)
        return None

    def _score(t):
        n = _ascii(t.get("nome", ""))
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

        niveis = []
        for v in meta.get("nivelTerritorial", {}).values():
            if isinstance(v, list):
                niveis += [str(x).upper() for x in v]
        if "N3" not in niveis:
            continue

        id_classif, cats = None, {}
        for c in meta.get("classificacoes", []):
            nc = _ascii(c.get("nome", ""))
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

        id_var = None
        for v in meta.get("variaveis", []):
            nv = _ascii(v["nome"])
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

    col_uf = next(
        (k for k, v in header.items()
         if any(w in _ascii(str(v)) for w in ("unidade", "estado", "uf"))
         and "codigo" in _ascii(str(v))),
        None,
    )
    if col_uf is None:
        col_uf = next(
            (k for k, v in header.items() if "Codigo" in str(v)), list(header)[0]
        )
    col_classe = next(
        k for k, v in header.items() if "rendimento" in _ascii(str(v))
    )
    out = df[[col_uf, col_classe, "V"]].copy()
    out.columns = ["cod_uf", "classe", "valor"]
    out["valor"] = pd.to_numeric(out["valor"], errors="coerce")
    return out.dropna(subset=["valor"])


def _ajustar_alpha_pareto(dist_pnad, li_topo):
    """Estima α de Pareto da cauda superior (OLS no espaço log-sobrevivência)."""
    cauda = sorted(
        [(li, ls, pop) for li, ls, pop in dist_pnad if li >= li_topo * 0.99],
        key=lambda x: x[0],
    )
    if len(cauda) < 2:
        return None

    pop_total_cauda = sum(p for _, _, p in cauda)
    if pop_total_cauda <= 0:
        return None

    pontos = []
    pop_acumulada = 0.0
    for li, ls, pop in cauda:
        if ls is None:
            break
        pop_acumulada += pop
        x_k = ls
        p_k = 1.0 - pop_acumulada / pop_total_cauda
        if 0 < p_k < 1 and x_k > li_topo:
            pontos.append((x_k, p_k))

    if not pontos:
        return None

    soma_xy = sum(math.log(x / li_topo) * (-math.log(p)) for x, p in pontos)
    soma_xx = sum(math.log(x / li_topo) ** 2 for x, p in pontos)
    if soma_xx == 0:
        return None
    alpha_est = soma_xy / soma_xx

    if 1.01 < alpha_est < 10.0:
        return alpha_est
    return None


def calibrar_alpha_por_uf(li_topo, ufs, alpha_fallback):
    """Usa PNAD Contínua para estimar α de Pareto por UF."""
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

    for uf in ufs:
        alphas.setdefault(str(uf), alpha_fallback)

    return alphas


# ── Validação externa — Atlas 2010 via IPEADATA ───────────────────────────────

def buscar_gini_2010_ipea():
    """Busca Gini municipal 2010 do Atlas do Desenvolvimento Humano via IPEADATA.

    Retorna dict {cod_mun_str: gini_2010} ou {} se inacessível.
    """
    print("\nBuscando Gini 2010 do Atlas (IPEADATA)...")
    url = f"{IPEADATA_URL}/ValoresSerie(SERCODIGO='ADH_GINI')"
    try:
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        dados = r.json().get("value", [])
    except Exception as e:
        print(f"  [aviso] IPEADATA inacessível ({e}); validação 2010 omitida.",
              file=sys.stderr)
        return {}

    resultado = {}
    for item in dados:
        valdata = str(item.get("VALDATA", ""))
        if not valdata.startswith("2010"):
            continue
        cod = str(item.get("TERCODIGO", "")).strip()
        val = item.get("VALVALOR")
        if not cod or val is None:
            continue
        try:
            resultado[cod] = float(val)
        except (TypeError, ValueError):
            pass

    if resultado:
        print(f"  {len(resultado)} municípios com Gini 2010 carregados.")
    else:
        print("  [aviso] Nenhum dado 2010 retornado pelo IPEADATA.", file=sys.stderr)

    return resultado


def _match_gini_2010(cod_mun, gini_2010_dict):
    """Casa cod_mun IBGE (7 dígitos) com chaves do dicionário IPEADATA.

    Tenta 7 dígitos direto e depois 6 dígitos (sem dígito verificador).
    """
    s = str(cod_mun)
    if s in gini_2010_dict:
        return gini_2010_dict[s]
    if s[:6] in gini_2010_dict:
        return gini_2010_dict[s[:6]]
    return None


# ── Cálculo do Gini ────────────────────────────────────────────────────────────

def gini_agrupado(classes, alpha_topo=None, medias_override=None):
    """Limites de Gastwirth do Gini a partir de [(li, ls, pop)], em SM.

    alpha_topo: parâmetro de forma Pareto para a classe aberta superior.
    medias_override: dict {(li, ls): mean} — substitui o ponto médio por médias
      condicionais lognormais, reduzindo o viés do bound inferior e apertando
      o bound superior (G_max é maximizado exatamente no ponto médio).
    """
    if alpha_topo is None:
        alpha_topo = ALPHA_FALLBACK
    if medias_override is None:
        medias_override = {}

    dados = []
    for li, ls, pop in classes:
        if pop <= 0:
            continue
        if ls is None:
            media = alpha_topo * li / (alpha_topo - 1)
            g_max = 1.0 / (2.0 * alpha_topo - 1.0)
        else:
            media = medias_override.get((li, ls), (li + ls) / 2.0)
            if media > 0 and ls > li:
                g_max = max(0.0, (ls - media) * (media - li) / (media * (ls - li)))
            else:
                g_max = 0.0
        dados.append((media, pop, g_max))

    if not dados:
        return None

    dados.sort()
    pop_total = sum(p for _, p, _ in dados)
    renda_total = sum(m * p for m, p, _ in dados)
    if renda_total == 0:
        return (0.0, 0.0)

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
        description="Gini municipal 2022 — Gastwirth + lognormal + validação Atlas 2010."
    )
    ap.add_argument("--uf", type=int, default=None, help="Código IBGE da UF (ex.: 35)")
    ap.add_argument("--saida", default="gini_municipal_2022.csv")
    ap.add_argument(
        "--alpha", type=float, default=None,
        help="Parâmetro Pareto α fixo (pula calibração PNAD).",
    )
    ap.add_argument(
        "--alpha-values", default=None,
        help="α adicionais separados por vírgula (ex.: 2.5,3.0). "
             "Valores 1.2, 1.5 e 2.0 já são sempre incluídos.",
    )
    ap.add_argument(
        "--sem-validacao-2010", action="store_true",
        help="Pula busca do Gini 2010 no IPEADATA.",
    )
    args = ap.parse_args()

    alpha_sens = list(ALPHA_SENSITIVITY)
    if args.alpha_values:
        try:
            extras = [float(v.strip()) for v in args.alpha_values.split(",")]
            alpha_sens = sorted(set(alpha_sens + extras))
        except ValueError:
            sys.exit("--alpha-values: forneça valores numéricos separados por vírgula.")

    # ── 1. Estrutura do Censo ────────────────────────────────────────────────
    print("Lendo metadados da tabela 10296 (Censo 2022)...")
    id_var, id_renda, categorias, outras, cods_sem_dec = descobrir_estrutura_censo()
    print(f"  variável={id_var}, classif_renda={id_renda}, "
          f"{len(categorias)} classes de renda")
    for cod, (nome, (li, ls)) in sorted(categorias.items()):
        print(f"    [{cod}] {nome} → ({li}, {ls})")

    li_topo = max(li for _, (_, (li, ls)) in categorias.items() if ls is None)

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
                                    list(categorias.keys()), outras, cods_sem_dec)
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
    bounds_por_nome = {nome: b for _, (nome, b) in categorias.items()}
    df["cod_uf"] = df["cod_mun"].astype(str).str[:2]

    # ── 4. Gini 2010 via IPEADATA ────────────────────────────────────────────
    gini_2010_dict = {}
    if not args.sem_validacao_2010:
        gini_2010_dict = buscar_gini_2010_ipea()

    # ── 5. Calcula Gini por município ────────────────────────────────────────
    print("\nCalculando Gini por município...")
    resultados = []

    for (cod, nome, cod_uf), grupo in df.groupby(
        ["cod_mun", "municipio", "cod_uf"]
    ):
        alpha_uf = alpha_por_uf.get(str(cod_uf), ALPHA_FALLBACK)

        classes = []
        pop_sem_declaracao = 0.0
        for _, row in grupo.iterrows():
            classe_nome = str(row["classe"])
            b = bounds_por_nome.get(classe_nome) or parse_classe(classe_nome)
            if b is None:
                if "sem declaracao" in _ascii(classe_nome):
                    pop_sem_declaracao += row["valor"]
                continue
            classes.append((b[0], b[1], row["valor"]))

        pop_classes = sum(c[2] for c in classes)
        pop_total_mun = pop_classes + pop_sem_declaracao
        pct_sem_dec = pop_sem_declaracao / pop_total_mun if pop_total_mun > 0 else 0.0

        pop_zero = sum(pop for li, ls, pop in classes
                       if li == 0.0 and ls is not None and ls == 0.0)
        classes_pos = [(li, ls, pop) for li, ls, pop in classes
                       if not (li == 0.0 and ls is not None and ls == 0.0)]
        p_zero = pop_zero / pop_classes if pop_classes > 0 else 0.0

        ln_params = fit_lognormal_grouped(classes_pos)

        medias_override = {}
        gini_ln = None
        ln_sigma = None
        ln_converged = False
        ln_mad = None
        ln_outside = None

        gini_ic_low = gini_ic_high = gini_se = None

        aic_ln = None
        if ln_params is not None:
            mu_fit, sigma_fit, converged = ln_params
            ln_sigma = sigma_fit
            ln_converged = converged
            gini_ln = gini_lognormal_mixed(mu_fit, sigma_fit, p_zero)
            ln_mad = lognormal_fit_mad(classes_pos, mu_fit, sigma_fit)
            ic = ic_delta_lognormal(mu_fit, sigma_fit, p_zero, classes_pos)
            if ic is not None:
                gini_ic_low, gini_ic_high, gini_se = ic
            # AIC lognormal (k=2)
            dados_ln = [(li, ls, pop) for li, ls, pop in classes_pos if pop > 0]
            n_ln = sum(p for _, _, p in dados_ln)
            nll_ln = _neg_ll_lognormal([mu_fit, math.log(sigma_fit)], dados_ln, n_ln)
            aic_ln = 4.0 + 2.0 * nll_ln * n_ln
            for li, ls, _ in classes:
                if ls is not None and not (li == 0.0 and ls == 0.0):
                    try:
                        medias_override[(li, ls)] = lognormal_conditional_mean(
                            li, ls, mu_fit, sigma_fit
                        )
                    except Exception:
                        pass

        # ── Gamma ──────────────────────────────────────────────────────────────
        gini_gm = gm_mad = gm_converged = aic_gm = None
        gm_params = fit_gamma_grouped(classes_pos)
        if gm_params is not None:
            alpha_gm, beta_gm, gm_conv, gm_nll = gm_params
            gm_converged = gm_conv
            G_plus_gm = gini_gamma_shape(alpha_gm)
            gini_gm = gini_mixed(G_plus_gm, p_zero)
            gm_mad = mad_grouped(classes_pos,
                                  lambda li, ls: _bracket_prob_gamma(li, ls, alpha_gm, beta_gm))
            n_gm = sum(p for _, _, p in classes_pos if p > 0)
            aic_gm = 4.0 + 2.0 * gm_nll * n_gm  # k=2

        # ── Singh-Maddala ───────────────────────────────────────────────────────
        gini_sm_val = sm_mad = sm_converged = aic_sm = None
        sm_params = fit_sm_grouped(classes_pos)
        if sm_params is not None:
            a_sm, b_sm, q_sm, sm_conv, sm_nll = sm_params
            sm_converged = sm_conv
            G_plus_sm = gini_sm_numerical(a_sm, q_sm)
            if G_plus_sm is not None:
                gini_sm_val = gini_mixed(G_plus_sm, p_zero)
            sm_mad = mad_grouped(classes_pos,
                                  lambda li, ls: _bracket_prob_sm(li, ls, a_sm, b_sm, q_sm))
            n_sm = sum(p for _, _, p in classes_pos if p > 0)
            aic_sm = 6.0 + 2.0 * sm_nll * n_sm  # k=3

        # ── Melhor distribuição por AIC ─────────────────────────────────────────
        aics = {n: v for n, v in [("lognormal", aic_ln), ("gamma", aic_gm),
                                   ("singh_maddala", aic_sm)] if v is not None}
        melhor_dist = min(aics, key=aics.get) if aics else None

        res = gini_agrupado(classes, alpha_topo=alpha_uf, medias_override=medias_override)

        if res is None:
            g_inf = g_sup = None
        else:
            g_inf, g_sup = res
            ln_outside = (gini_ln > g_sup) if gini_ln is not None else None

        # Sensibilidade ao α (padrão + extras)
        sens = {}
        for av in alpha_sens:
            sr = gini_agrupado(classes, alpha_topo=av, medias_override=medias_override)
            if sr is not None:
                label = f"{av:.1f}".replace(".", "_")
                sens[f"gini_inf_a{label}"] = round(sr[0], 4)
                sens[f"gini_sup_a{label}"] = round(sr[1], 4)

        resultados.append({
            "cod_mun": cod,
            "municipio": nome,
            "cod_uf": cod_uf,
            "alpha_pareto": round(alpha_uf, 4),
            "gini_inf": round(g_inf, 4) if g_inf is not None else None,
            "gini_sup": round(g_sup, 4) if g_sup is not None else None,
            "gini_lognormal": round(gini_ln, 4) if gini_ln is not None else None,
            "gini_ic_low": round(gini_ic_low, 4) if gini_ic_low is not None else None,
            "gini_ic_high": round(gini_ic_high, 4) if gini_ic_high is not None else None,
            "gini_se": round(gini_se, 6) if gini_se is not None else None,
            "gini_gamma": round(gini_gm, 4) if gini_gm is not None else None,
            "gini_singh_maddala": round(gini_sm_val, 4) if gini_sm_val is not None else None,
            "melhor_dist": melhor_dist,
            "aic_lognormal": round(aic_ln, 2) if aic_ln is not None else None,
            "aic_gamma": round(aic_gm, 2) if aic_gm is not None else None,
            "aic_singh_maddala": round(aic_sm, 2) if aic_sm is not None else None,
            "ln_sigma": round(ln_sigma, 4) if ln_sigma is not None else None,
            "ln_mad": round(ln_mad, 4) if ln_mad is not None else None,
            "gamma_mad": round(gm_mad, 4) if gm_mad is not None else None,
            "sm_mad": round(sm_mad, 4) if sm_mad is not None else None,
            "ln_converged": ln_converged,
            "gamma_converged": gm_converged,
            "sm_converged": sm_converged,
            "ln_outside_bounds": ln_outside,
            "pct_sem_declaracao": round(pct_sem_dec, 4),
            "populacao_considerada": int(round(pop_classes)),
            "gini_2010": _match_gini_2010(cod, gini_2010_dict),
            **sens,
        })

    res_df = pd.DataFrame(resultados).sort_values("cod_mun")
    res_df.to_csv(args.saida, index=False, encoding="utf-8")
    print(f"\n{len(res_df)} municípios → {args.saida}")

    # ── 6. Sumário para o paper ──────────────────────────────────────────────
    cols_summary = [c for c in ["gini_inf", "gini_sup", "gini_lognormal",
                                "gini_gamma", "gini_singh_maddala"]
                    if c in res_df.columns]
    print("\n" + res_df[cols_summary].describe().round(4).to_string())

    ln_ok = int(res_df["ln_converged"].sum())
    n_outside = int(res_df["ln_outside_bounds"].sum()) if "ln_outside_bounds" in res_df else 0
    n_bad = int((res_df["ln_mad"] > LN_MAD_THRESHOLD).sum()) if "ln_mad" in res_df else 0
    print(f"\nLognormal MLE : {ln_ok}/{len(res_df)} convergiram | "
          f"{n_outside} fora dos bounds ({100*n_outside/len(res_df):.1f}%) | "
          f"{n_bad} MAD > {LN_MAD_THRESHOLD} ({100*n_bad/len(res_df):.1f}%)")

    if "gamma_converged" in res_df.columns:
        gm_ok = int(res_df["gamma_converged"].sum()) if res_df["gamma_converged"].notna().any() else 0
        print(f"Gamma MLE     : {gm_ok}/{len(res_df)} convergiram")
    if "sm_converged" in res_df.columns:
        sm_ok = int(res_df["sm_converged"].sum()) if res_df["sm_converged"].notna().any() else 0
        print(f"Singh-Maddala : {sm_ok}/{len(res_df)} convergiram")
    if "melhor_dist" in res_df.columns:
        contagem = res_df["melhor_dist"].value_counts()
        print(f"\nMelhor dist. (AIC): {contagem.to_dict()}")

    if "pct_sem_declaracao" in res_df.columns:
        n_pos = int((res_df["pct_sem_declaracao"] > 0).sum())
        pct_max = res_df["pct_sem_declaracao"].max() * 100
        print(f"Sem declaração: {n_pos} municípios com >0% | máx {pct_max:.2f}%")

    if "gini_2010" in res_df.columns:
        val = res_df.dropna(subset=["gini_2010", "gini_lognormal"])
        if len(val) >= 100:
            corr = val["gini_lognormal"].corr(val["gini_2010"])
            vies = (val["gini_lognormal"] - val["gini_2010"]).mean()
            mae = (val["gini_lognormal"] - val["gini_2010"]).abs().mean()
            aumento = int((val["gini_lognormal"] > val["gini_2010"]).sum())
            print(f"\nValidação Atlas 2010 ({len(val)} municípios):")
            print(f"  Correlação de Pearson : {corr:.4f}")
            print(f"  Viés médio 2022−2010  : {vies:+.4f}")
            print(f"  MAE                   : {mae:.4f}")
            print(f"  Municípios com Gini ↑ : {aumento} ({100*aumento/len(val):.1f}%)")
        elif len(val) > 0:
            print(f"\n[aviso] Apenas {len(val)} municípios com Gini 2010 — "
                  "verifique o código de território no IPEADATA.")


if __name__ == "__main__":
    main()
