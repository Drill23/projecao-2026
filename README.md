# Projeção da apuração · 2º turno

Robô e página que acompanham a apuração de um 2º turno (presidente ou governador) e mostram, a
cada minuto, a **projeção do resultado final**, a faixa de 90% e a chance de vitória.

- Página: **https://drill23.github.io/projecao-2026/**
- Versão do modelo: **v2**, testada em 55 segundos turnos de presidente e governador (2010–2026)
- Estado: **desligado**. Nada roda sozinho; o robô só liga quando alguém manda.

## Como usar numa próxima eleição

1. **Criar a configuração.** Copie `eleicoes/modelo-presidente.json` (ou
   `modelo-governador.json`, um arquivo por estado), preencha os números dos dois candidatos e
   os códigos de eleição/pleito do 1º e do 2º turno. Os códigos estão em
   `https://resultados.tse.jus.br/oficial/comum/config/ele-c.json`.
2. **Preparar a base** (alguns dias antes do 2º turno, quando o TSE já publicou os dados
   abertos do 1º turno): aba **Actions** → *Preparar base* → **Run workflow** → informe o
   arquivo de configuração.
3. **Ensaiar**: *Ensaio* → **Run workflow** com turno `1`. O robô lê os arquivos do 1º turno
   (que já estão no TSE), roda o modelo três vezes e salva o que viu em `logs/ensaio.txt`, sem
   publicar nada.
4. **No dia**, perto das 16h45: *Apuração ao vivo* → **Run workflow** → configuração e, se
   quiser, a hora de parar. O robô publica na página a cada minuto, se renova sozinho antes do
   limite de 6 horas do GitHub e para 20 minutos depois do fim da apuração.

Opcional na configuração:
- `outros_prior`: para onde se espera que vá cada candidato eliminado (por número), por
  exemplo `{"70": 0.30}` = 30% dos eleitores do 70 para o candidato A;
- `pesquisa`: última pesquisa do 2º turno (`{"pA": 0.52, "sd": 0.025}`), usada só como ponto
  de partida nos primeiros minutos.

## Estrutura

| Pasta | O que tem |
|---|---|
| `projecao/modelo_v2.py` | o modelo v2 |
| `projecao/base.py` | carrega a base de uma eleição e monta o modelo |
| `projecao/modelo.py` | o modelo v1 (o da noite de 25/10/2026), mantido para comparação |
| `scripts/ao_vivo.py` | o robô |
| `scripts/preparar_base.py` | monta a base por seção a partir dos dados abertos do TSE |
| `scripts/historico.py`, `geo.py`, `amostra_chegada.py` | dados históricos, posição dos municípios e horários reais de chegada das urnas |
| `scripts/avaliar.py`, `analisar.py` | reprise dos 2º turnos passados e estatísticas v1 × v2 |
| `eleicoes/` | configurações |
| `historico/` | bases por seção dos 2º turnos de 2010 a 2026 |
| `resultados/` | resultados da avaliação (`analise.txt`, `resumo.json`, `calibracao.json`) |
| `docs/index.html` | a página |

O ramo `dados` guarda só o `projecao.json` que a página lê. A explicação completa do modelo,
com ilustrações e estatísticas, está no repositório **projecao-eleitoral**.

Dados: TSE (divulgação de resultados e dados abertos). Projeção não oficial.
