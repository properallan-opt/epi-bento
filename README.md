# EPI Bento — SACI

Detecção de EPI com BentoML + Ultralytics, usando `model_weights/ppe-v2-100epochs.pt`.
Implementação baseada em `/home/ppiper/smokefire-bento`, com o mesmo transporte
`BaseYOLOService` e saída por pessoa compatível com a lógica de `epi-main.zip`.
Não depende de RabbitMQ, Docker, oclusão ou de um segundo modelo de pessoas.

## Estrutura

```text
.
├── environment.yaml
├── model_weights/
│   └── ppe-v2-100epochs.pt
├── saci-model.yaml
├── saci_yolo.py          # Transportes copiados do exemplo de fogo
├── epi_logic.py          # Associação EPI/pessoa e saída legada
├── service.py            # service:EPIDetection
├── tests/
└── README.md
```

O peso foi movido de `Downloads` para este repositório e é versionado com Git LFS.
Ao clonar, execute `git lfs install` e `git lfs pull`.

## Instalação e execução

```bash
mamba env create -f environment.yaml
mamba activate saci-epi
bentoml serve service:EPIDetection --host 127.0.0.1 --port 3005
curl http://127.0.0.1:3005/readyz
```

Documentação HTTP em `http://127.0.0.1:3005/docs`.

O ambiente é uma especificação portátil, não um lock por arquitetura. Diferentemente
do ambiente do exemplo de fogo, não fixa bibliotecas ARM64/CUDA específicas.
Em Jetson, use os wheels PyTorch/torchvision compatíveis com seu JetPack antes de
servir. A versão Ultralytics foi fixada em `8.4.191`, registrada no checkpoint.

### SACI

O manifesto usa `backend: unix`, ambiente `saci-epi`, serviço `service:EPIDetection`,
porta `3005` e método padrão `recognize`. A porta é diferente da usada pelo fogo.
Para registrar a partir deste repositório Git local (com os arquivos commitados):

```bash
saci add /home/ppiper/Documents/epi-bento
saci start epi
saci infer epi /caminho/imagem.jpg
```

Depois de publicar em um remoto acessível pelo SACI, também é possível usar
`saci add <URL-GIT-DESTE-REPOSITORIO>`.

O manifesto tem timeout de inferência de 30 s e prontidão de 120 s. Em CPUs lentas,
considere aumentar o timeout. O SACI não foi iniciado/registrado automaticamente.

## Modelo e regra de conformidade

Classes do peso (IDs preservados):

| ID | Classe |
|---:|---|
| 0 | helmet |
| 1 | gloves |
| 2 | vest |
| 3 | boots |
| 4 | goggles |
| 5 | none |
| 6 | Person |
| 7 | no_helmet |
| 8 | no_goggle |
| 9 | no_gloves |
| 10 | no_boots |

O próprio peso detecta pessoas; **não** se assume que pessoa seja a classe 0.
A associação mantém a heurística do Docker:

- capacete: centro da caixa dentro dos 38% superiores da pessoa, com margem lateral de 10%;
- colete: centro dentro da região de 20% a 78% da altura, margem lateral de 8%;
- em cada região, seleciona-se o equipamento de maior confiança;
- pessoa conforme = capacete **e** colete associados;
- alerta = pelo menos uma pessoa não conforme;
- sem pessoas = sem alerta, mesmo que existam caixas de EPI.

As coordenadas de associação são arredondadas como no legado. Em pessoas sobrepostas,
um equipamento pode ser associado a mais de uma pessoa, também como no legado.
Ausência de detecção positiva não é prova física de ausência de EPI: oclusão, escala
e limiares podem gerar alertas. Valide os limiares com imagens representativas.

Luvas, botas, óculos e classes negativas são expostos nas detecções brutas, mas **não
alteram a regra de conformidade capacete + colete**. `no_helmet` não conta como capacete.

## Configuração

| Variável | Padrão | Finalidade |
|---|---|---|
| `EPI_MODEL_PATH` | `model_weights/ppe-v2-100epochs.pt` | Relativo à raiz do código ou absoluto |
| `EPI_DEVICE` | `cpu` | Use `0` para GPU CUDA compatível |
| `EPI_IMAGE_SIZE` | `640` | `imgsz` |
| `EPI_CONFIDENCE_THRESHOLD` | `0.25` | Classes adicionais |
| `EPI_PERSON_THRESHOLD` | `0.35` | Pessoa |
| `EPI_HELMET_THRESHOLD` | `0.45` | Capacete |
| `EPI_VEST_THRESHOLD` | `0.45` | Colete |
| `EPI_IOU_THRESHOLD` | `0.70` | NMS IoU |

O serviço usa um worker. Para execução direta, exporte as variáveis antes de iniciar.
No SACI, ajuste `runtime.environment_variables` no manifesto (elas definem os valores
do processo gerenciado). Não há engine TensorRT pré-gerada neste repositório.

Cada chamada aceita `confidence_threshold`, `iou_threshold`, `person_threshold`,
`helmet_threshold` e `vest_threshold` entre 0 e 1. Os limiares de pessoa/capacete/colete
são independentes do limiar geral; a predição usa o menor deles e filtra por classe.

## Entradas (contratos do exemplo SACI)

Todos os métodos retornam o mesmo JSON. Arrays e pixels de raw/SHM são **BGR uint8**,
com shape `[altura, largura, 3]`. Base64 e arquivos são imagens codificadas JPEG/PNG.

| Método | Entrada |
|---|---|
| `recognize` | Arquivo `image` via multipart; limiares como parâmetros opcionais |
| `recognize_array` | `image: np.ndarray`; limiares opcionais |
| `recognize_path` | Payload único `{"image_path": "/caminho/no/servidor.jpg", ...}` |
| `recognize_base64` | Payload único `{"image_base64": "<base64 puro>", ...}` |
| `recognize_raw` | Payload único binário `packet` |
| `recognize_shm` | Payload único `{"image": {"name": "...", "shape": [480,640,3], "dtype": "uint8"}, ...}` |

Nos métodos de payload único, os limiares ficam no mesmo objeto (em raw, no header).
Não envolva esse objeto em uma chave `request`.
`recognize_path` acessa o filesystem do servidor, não o do cliente.

```bash
curl -X POST http://127.0.0.1:3005/recognize \
  -F 'image=@/caminho/imagem.jpg' -F 'helmet_threshold=0.50'
```

Base64 em Python:

```python
import base64
from pathlib import Path
import requests

response = requests.post('http://127.0.0.1:3005/recognize_base64', json={
    'image_base64': base64.b64encode(Path('imagem.jpg').read_bytes()).decode(),
    'helmet_threshold': 0.50,
}, timeout=30)
response.raise_for_status()
print(response.json())
```

### Raw e memória compartilhada

O pacote raw segue exatamente o exemplo de fogo:
`[4 bytes tamanho do JSON em uint32 big-endian][JSON UTF-8][pixels contíguos]`.
O JSON contém `shape`, `dtype` e parâmetros opcionais. Exemplo de montagem:

```python
import json
import struct
import numpy as np

frame = np.zeros((480, 640, 3), dtype=np.uint8)
header = json.dumps({'shape': list(frame.shape), 'dtype': 'uint8',
                     'helmet_threshold': 0.5}).encode()
packet = struct.pack('!I', len(header)) + header + frame.tobytes()
# Enviar packet ao método recognize_raw como application/octet-stream.
```

SHM é POSIX/Linux, local ao servidor (ou namespace IPC compartilhado). O produtor
cria e preenche o segmento e continua responsável por `close`/`unlink`. O serviço
abre apenas para leitura, copia o frame antes da inferência e não registra nem
remove o segmento. O produtor deve manter o segmento válido e os pixels estáveis
até a conclusão da chamada.

## Saída e compatibilidade Docker

Exemplo ilustrativo de pessoa sem colete (campos resumidos):

```json
{
  "detection": true,
  "bbox": [[100.0, 100.0, 200.0, 400.0, 0.8, "Person"], [120.0, 100.0, 180.0, 160.0, 0.9, "helmet"]],
  "detections": [{
    "person_id": 0,
    "coordinates": {"x": 150, "y": 250, "width": 100, "height": 300},
    "person_confidence": "0.8",
    "helmet": true,
    "helmet_confidence": "0.9",
    "vest": false,
    "vest_confidence": "0.0",
    "compliant": false
  }],
  "non_compliant": true,
  "time": 1700000000.0,
  "EPIMaxConfidence": "0.9"
}
```

- `detection`: indicador SACI de **alerta de não conformidade**, igual a `non_compliant`.
- `bbox`: todas as caixas aceitas, `[x1,y1,x2,y2,confiança,label]`, em pixels da imagem
  original. Pode ser não vazio mesmo com `detection=false` (pessoas conformes).
- `detections`: lista **por pessoa**, preservando nomes, tipos e coordenadas do Docker.
  `coordinates.x/y` são o **centro**, não o canto superior esquerdo. Confianças são strings.
- `non_compliant` e `time`: mesmos campos do legado; `time` é Unix timestamp do processamento.
- `EPIDetection`: objeto com `{detections, non_compliant, time}` idêntico ao conteúdo
  antes serializado no header RabbitMQ. Para um consumidor legado, use
  `headers['EPIDetection'] = json.dumps(response['EPIDetection'])`.
- `EPIMaxConfidence`: string com a maior confiança entre pessoas e EPIs associados,
  ou `"0.0"` sem pessoas, como no Docker.
- `raw_detections`: caixas do modelo como `{class_id,label,confidence,bbox_xyxy}`.
  Use este campo para luvas, botas, óculos e classes negativas.
- `counts_by_label`, `detection_count`, `person_count`, `non_compliant_count`: contadores.
  `detection_count` conta caixas brutas aceitas, não apenas pessoas/alertas.

Não são propagados `AngelId`, `CameraId` ou `DateIsp`, nem são publicados headers ou
imagens em filas. Esse roteamento pertence à integração externa; o serviço retorna JSON.
Não salva imagens nem implementa o estágio de oclusão do pipeline Docker.

## Testes

Com o ambiente ativo:

```bash
python -m unittest discover -s tests -v
python -m py_compile service.py saci_yolo.py epi_logic.py
```

Cobertura: conformidade/ausência de EPI, múltiplas pessoas, classes negativas,
regiões espaciais, tipos legados, limiares por classe, ausência de caixas e transportes
array/path/base64/raw/SHM com validações de entrada. Testes de serviço usam resultado
YOLO sintético, sem carregar pesos. O peso real e HTTP foram validados em CPU com
frame preto; isso não substitui avaliação de precisão em cenas reais.
