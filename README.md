# Gini Municipal Brasil — Censo 2022

Estimativas do Índice de Gini de rendimento domiciliar per capita para os **5.570 municípios brasileiros**, calculadas a partir dos microdados agrupados do Censo Demográfico 2022 (IBGE, divulgação out/2025).

## Dados

| Arquivo | Descrição |
|---|---|
| `gini_municipal_2022.py` | Script de cálculo (fonte única) |
| `gini_municipal_2022.csv` | Resultado: 5.570 municípios, 20 colunas |

### Colunas do CSV

| Coluna | Descrição |
|---|---|
| `cod_mun` | Código IBGE de 7 dígitos |
| `municipio` | Nome do município |
| `cod_uf` | Código da UF (2 dígitos) |
| `alpha_pareto` | Parâmetro de forma Pareto α da cauda superior |
| `gini_inf` | Limite inferior de Gastwirth (médias condicionais lognormais) |
| `gini_sup` | Limite superior de Gastwirth (dispersão intra-faixa máxima) |
| `gini_lognormal` | Estimativa pontual lognormal MLE |
| `ln_sigma` | Desvio-padrão log (σ) do ajuste lognormal |
| `ln_mad` | MAD: desvio absoluto médio entre proporções observadas e preditas |
| `ln_converged` | `True` se o MLE lognormal convergiu |
| `ln_outside_bounds` | `True` se `gini_lognormal > gini_sup` (ver nota metodológica) |
| `pct_sem_declaracao` | Fração de domicílios sem declaração de rendimento |
| `populacao_considerada` | Domicílios com rendimento declarado usados no cálculo |
| `gini_2010` | Gini 2010 do Atlas do Desenvolvimento Humano (IPEADATA) |
| `gini_inf_a1_2` / `gini_sup_a1_2` | Bounds com α = 1,2 |
| `gini_inf_a1_5` / `gini_sup_a1_5` | Bounds com α = 1,5 |
| `gini_inf_a2_0` / `gini_sup_a2_0` | Bounds com α = 2,0 |

## Metodologia

### Fonte primária

SIDRA Tabela 10296 — *Moradores em domicílios particulares permanentes ocupados por classes de rendimento domiciliar per capita (em salários mínimos)*. Censo Demográfico 2022, resultados da amostra. O salário mínimo de referência é R$ 1.212,00 (vigente em agosto/2022).

As 11 classes de rendimento disponíveis são:

| Classe | Intervalo (SM) |
|---|---|
| Sem rendimento | [0, 0] |
| Até 1/4 SM | (0; 0,25] |
| Mais de 1/4 a 1/2 SM | (0,25; 0,5] |
| Mais de 1/2 a 1 SM | (0,5; 1] |
| Mais de 1 a 2 SM | (1; 2] |
| Mais de 2 a 3 SM | (2; 3] |
| Mais de 3 a 5 SM | (3; 5] |
| Mais de 5 a 10 SM | (5; 10] |
| Mais de 10 a 15 SM | (10; 15] |
| Mais de 15 a 20 SM | (15; 20] |
| Mais de 20 SM | (20; ∞) |

### Limites de Gastwirth (1972)

Para dados agrupados em *K* classes sem sobreposição, o Gini admite uma decomposição exata (Gastwirth, 1972):

$$G = G_{\text{entre}} + \sum_{i=1}^{K} p_i \, s_i \, G_i$$

onde $p_i$ é a proporção populacional, $s_i$ a participação na renda e $G_i$ o Gini intra-faixa do grupo *i*.

**Limite inferior** (`gini_inf`): $G_i = 0$ para toda faixa. A média de cada faixa é calculada como a média condicional lognormal $E[X \mid a < X \leq b]$ quando o MLE converge, ou o ponto médio $(a+b)/2$ como fallback.

**Limite superior** (`gini_sup`): $G_i$ é maximizado sujeito à média observada $m_i$:

- Faixa fechada $[l_i, l_s]$ com média $m_i$:

$$G_i^{\max} = \frac{(l_s - m_i)(m_i - l_i)}{m_i(l_s - l_i)}$$

- Classe aberta $(l_i, \infty)$ com distribuição Pareto($\alpha$):

$$G_i^{\max} = \frac{1}{2\alpha - 1}$$

### Estimativa pontual lognormal

Ajuste MLE de lognormal($\mu$, $\sigma$) aos dados agrupados de renda positiva via maximização da log-verossimilhança multinomial:

$$\ell(\mu, \sigma) = \sum_{i} n_i \log P(l_i < X \leq l_s^{(i)} \mid \mu, \sigma)$$

otimizada com Nelder-Mead. A estimativa do Gini para a distribuição mista (fração $p_0$ com renda zero e fração $1 - p_0$ lognormal) é derivada analiticamente da curva de Lorenz:

$$G = 1 - 2(1 - p_0)^2 \, \Phi\!\left(-\frac{\sigma}{\sqrt{2}}\right)$$

que reduz ao Gini do lognormal puro $2\Phi(\sigma/\sqrt{2}) - 1$ quando $p_0 = 0$.

**Qualidade do ajuste** (`ln_mad`): desvio absoluto médio entre proporções observadas e preditas por faixa. Valores acima de 0,04 indicam ajuste insatisfatório.

**Nota sobre `ln_outside_bounds`**: quando `gini_lognormal > gini_sup`, isso **não** indica erro de cálculo. Os bounds de Gastwirth assumem que toda a massa de cada faixa está contida no intervalo $[l_i, l_s]$, enquanto o lognormal tem suporte em $(0, \infty)$ — premissas distintas que podem produzir estimativas fora dos bounds em 14,1% dos municípios.

### Calibração da cauda Pareto

A classe aberta ">20 SM" requer um parâmetro de forma $\alpha$ para a cauda de Pareto. O script tenta calibrar $\alpha$ por UF via PNAD Contínua (SIDRA): ajuste por mínimos quadrados no espaço log-sobrevivência sobre as faixas superiores disponíveis. Na ausência de dados compatíveis da PNAD, usa-se $\alpha = 1{,}5$ (mediana da literatura para o Brasil). A análise de sensibilidade em $\alpha \in \{1{,}2;\ 1{,}5;\ 2{,}0\}$ está sempre presente no CSV.

### Validação histórica

Os valores de 2022 foram comparados com o Gini municipal 2010 do Atlas do Desenvolvimento Humano (série `ADH_GINI`, IPEADATA):

| Métrica | Valor |
|---|---|
| Municípios com Gini 2010 disponível | 5.564 |
| Correlação de Pearson (2022 vs 2010) | 0,575 |
| Viés médio (2022 − 2010) | −0,056 |
| MAE | 0,063 |
| Municípios com Gini maior em 2022 | 785 (14,1%) |

A correlação moderada (r = 0,575) e o viés negativo consistente indicam redução generalizada da desigualdade entre 2010 e 2022, coerente com a literatura recente sobre convergência de renda no Brasil.

## Resultados

| Estatística | gini_inf | gini_sup | gini_lognormal |
|---|---|---|---|
| Média | 0,421 | 0,457 | 0,439 |
| Desvio-padrão | 0,050 | 0,047 | 0,045 |
| Mínimo | 0,257 | 0,312 | 0,292 |
| Mediana | 0,420 | 0,456 | 0,437 |
| Máximo | 0,713 | 0,737 | 0,825 |

Lognormal convergiu em 5.570/5.570 municípios (100%). Apenas 6 municípios (0,1%) com MAD > 0,04.

## Reprodução

```bash
pip install requests pandas scipy
python gini_municipal_2022.py                        # todos os municípios
python gini_municipal_2022.py --uf 35                # só São Paulo
python gini_municipal_2022.py --alpha 1.5            # α fixo, pula PNAD
python gini_municipal_2022.py --sem-validacao-2010   # pula IPEADATA
```

## Citação

Caso utilize estes dados, cite as fontes primárias:

- **IBGE** (2023). *Censo Demográfico 2022 — Resultados da Amostra*. Tabela 10296, SIDRA.
- **IPEA/PNUD/FJP** (2013). *Atlas do Desenvolvimento Humano no Brasil 2013*. Série ADH_GINI via IPEADATA.

## Limitações

- Os dados agrupados em 11 faixas introduzem imprecisão inerente; os bounds de Gastwirth quantificam esse erro.
- A fração de domicílios sem declaração de rendimento não está disponível como categoria separada na Tabela 10296 (divulgação preliminar); `pct_sem_declaracao` permanece zero até publicação dos microdados completos.
- O parâmetro Pareto $\alpha = 1{,}5$ é uma aproximação para a cauda superior; a sensibilidade ao α deve ser reportada em qualquer análise.
- O Censo 2022 captura rendimentos declarados; renda do capital e transferências informais podem estar sub-representadas, especialmente nos municípios com maior desigualdade.
