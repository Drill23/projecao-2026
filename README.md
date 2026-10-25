# Projeção do 2º turno — Presidente 2026

Página: **https://drill23.github.io/projecao-2026/**

Apuração do TSE com projeção do resultado final do 2º turno (Lula × Flávio Bolsonaro),
atualizada a cada minuto durante a apuração. Projeção não oficial.

## Como funciona

- **Robô** (`scripts/ao_vivo.py`, workflow *Apuração ao vivo*): liga sozinho no dia 25/10 às
  16h47 (Brasília). A cada minuto lê os arquivos públicos da divulgação do TSE (andamento por
  município, resultado por município e configuração de seções), roda o modelo e grava o
  resultado em `projecao.json` no ramo `dados`. Respeita um orçamento de requisições ao TSE
  e reduz o ritmo se receber HTTP 429.
- **Base** (`scripts/preparar_base.py`): resultado do 1º turno de 2026 em cada uma das ~517 mil
  seções do 2º turno (mesmas seções), mais 2022 e 2018 por seção para validação.
- **Modelo** (`projecao/modelo.py`): em cada município, compara o apurado com o 1º turno das
  mesmas seções; estima a fração dos votos de Cury, Renan Santos, Caiado e nanicos que foi
  para cada lado, mais efeitos Brasil → região → UF → município com encolhimento; projeta as
  seções que faltam. Faixa de 90% por sorteio dos parâmetros (aproximação de Laplace) e dos
  efeitos não observados, calibrada na reprise.
- **Reprise** (`scripts/reprise_2022.py`): prevê o 2º turno de 2022 a partir do 1º turno de
  2022, com as urnas chegando na ordem real de chegada (amostra dos arquivos auxiliares do
  1º turno de 2026). Erro da projeção abaixo de 0,2 ponto já com 1% apurado.
- **Página** (`docs/index.html`, publicada no GitHub Pages): lê `projecao.json` do ramo
  `dados` a cada 75 segundos pela API do GitHub; se o limite de 60 leituras por hora acabar,
  usa o arquivo raw (pode atrasar até ~5 min).

## Ligar à mão

Aba **Actions** → *Apuração ao vivo* → **Run workflow**. Cada execução dura até ~5h50; se a
apuração continuar, a próxima execução agendada assume.

Dados: TSE (divulgação de resultados e dados abertos).
