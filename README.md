# EPI Bento — SACI

Detecção de EPI com BentoML + Ultralytics, usando por padrão
`model_weights/ppe-v2-100epochs.engine` (TensorRT 11.3.0.99, FP32, para a GPU Orin da `flintstone`).
O checkpoint `model_weights/ppe-v2-100epochs.pt` continua disponível para CPU/reexportação.
Implementação baseada em `/home/ppiper/smokefire-bento`, com o mesmo transporte
`BaseYOLOService` e saída por pessoa compatível com a lógica de `epi-main.zip`.
Não depende de RabbitMQ, Docker, oclusão ou de um segundo modelo de pessoas.

## Estrutura

```text
.
├── environment.yaml
├── model_weights/
│   ├── ppe-v2-100epochs.engine
│   ├── ppe-v2-100epochs.engine.json  # Proveniência e validação da exportação
│   └── ppe-v2-100epochs.pt
├── saci-model.yaml
├── saci_yolo.py          # Transportes copiados do exemplo de fogo
├── epi_logic.py          # Associação EPI/pessoa e saída legada
├── service.py            # service:EPIDetection
├── tests/
└── README.md
```

Os pesos `.pt` e `.engine` são versionados com Git LFS.
Ao clonar, execute `git lfs install` e `git lfs pull`.

## Instalação e execução

```bash
mamba env create -f environment.yaml
mamba activate yolo-base-jp7.2.1-b49
bentoml serve service:EPIDetection --host 127.0.0.1 --port 3005
curl http://127.0.0.1:3005/readyz
```

Documentação HTTP em `http://127.0.0.1:3005/docs`.

O `environment.yaml` é o **mesmo lock do smokefire** (`yolo-base-jp7.2.1-b49`,
Python 3.10, torch 2.13.0+cu132, **TensorRT 11.3.0.99** com bindings e libs,
Ultralytics 8.4.95, BentoML 1.4.39). Por ter o mesmo conteúdo, o SACI reutiliza o
ambiente já criado (mesmo hash), sem instalar nada. Ele é específico de
Jetson/JetPack 7.2.1 (ARM64, CUDA 13.2); não é portátil para outras plataformas.
Não altere este arquivo isoladamente: a engine só carrega no TensorRT em que foi gerada.

Para executar em CPU, selecione explicitamente o checkpoint:

```bash
EPI_MODEL_PATH=model_weights/ppe-v2-100epochs.pt EPI_DEVICE=cpu \
  bentoml serve service:EPIDetection --host 127.0.0.1 --port 3005
```

### Exportação TensorRT

Engine gerada na `flintstone` (hostname `jetson-02`, GPU Orin, compute capability 8.7)
com o ambiente `yolo-base-jp7.2.1-b49`: Ultralytics 8.4.95, TensorRT 11.3.0.99 e
PyTorch 2.13.0+cu132.

```python
from ultralytics import YOLO

YOLO('model_weights/ppe-v2-100epochs.pt').export(
    format='engine', imgsz=640, batch=1, half=False, dynamic=False,
    workspace=1, device=0, simplify=False,
)
```

Foi usado `YOLO_AUTOINSTALL=false`, com `onnx`, `protobuf` e `ml_dtypes==0.5.1`
em um diretório à parte (`PYTHONPATH`), sem alterar o ambiente compartilhado.
O workspace é limitado a 1 GiB. A entrada é fixa `[1, 3, 640, 640]`, com NMS no Ultralytics.

**FP32:** o TensorRT 11 é fortemente tipado e removeu a flag FP16 do builder; o FP16
exige o NVIDIA ModelOpt (AutoCast), ausente do ambiente. Por isso a engine é FP32
(~134 MB; ~16 ms de inferência no Orin nos smoke tests, contra ~6,6 ms da versão FP16
anterior em TensorRT 10). Os testes produziram as mesmas 11 caixas/classes.
Para FP16, instale o ModelOpt e reexporte com `quantize=16`.

A exportação precisa de memória de GPU livre (memória unificada no Jetson): com outros
modelos carregados ela falha por falta de memória.
Mantenha `EPI_IMAGE_SIZE=640` ao usar esta engine. Engines TensorRT são específicas
para hardware/runtime: para outra GPU ou outra versão do TensorRT, reexporte o `.pt`.
Consulte `model_weights/ppe-v2-100epochs.engine.json` para hashes e validação.
O PyTorch emite um aviso de suporte à arquitetura Orin; os smoke tests passaram.

### SACI

Esta release usa `version: "2.1.0-jp7.2.1-b49"` no `saci-model.yaml`,
correspondente à tag `v2.1.0-jp7.2.1-b49`. O sufixo `b49` identifica o build do
JetPack, não um contador de builds do modelo.

```bash
saci add https://github.com/properallan-opt/epi-bento --ref v2.1.0-jp7.2.1-b49
saci serve epi@2.1.0-jp7.2.1-b49 --wait 120
saci infer epi /caminho/imagem.jpg
```

O manifesto usa `backend: unix`, ambiente `yolo-base-jp7.2.1-b49` (compartilhado com o smokefire), serviço `service:EPIDetection`,
porta `3005` e método padrão `recognize`. A porta é diferente da usada pelo fogo.
Para registrar a partir deste repositório Git local (com os arquivos commitados):

```bash
saci add /home/ppiper/Documents/epi-bento
saci serve epi
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
- óculos: mesma região do capacete (38% superiores, margem lateral de 10%);
- luvas: centro entre 20% e 85% da altura, margem lateral de 20%;
- botas: centro entre 70% e 100% da altura, margem lateral de 10%;
- em cada região, seleciona-se o equipamento de maior confiança;
- pessoa conforme = capacete **e** colete associados;
- alerta = pelo menos uma pessoa não conforme;
- sem pessoas = sem alerta, mesmo que existam caixas de EPI.

As coordenadas de associação são arredondadas como no legado. Em pessoas sobrepostas,
um equipamento pode ser associado a mais de uma pessoa, também como no legado.
Ausência de detecção positiva não é prova física de ausência de EPI: oclusão, escala
e limiares podem gerar alertas. Valide os limiares com imagens representativas.

Cada pessoa agora inclui `gloves`, `boots` e `goggles` (booleanos), acompanhados de
`gloves_confidence`, `boots_confidence` e `goggles_confidence` (strings, `"0.0"`
quando não associados), tanto em `detections` quanto em `EPIDetection.detections`.
`true` significa pelo menos uma caixa associada: não comprova que ambas as mãos
ou ambos os pés estejam equipados. `false` significa não detectado/associado,
**não ausência física comprovada**. As regiões são heurísticas sem pose/keypoints;
valide especialmente mãos fora da região, oclusões e pessoas sobrepostas.

As novas associações são informativas: **não alteram a regra de conformidade
capacete + colete**, nem o cálculo legado de `EPIMaxConfidence`.
Classes negativas (`no_helmet`, `no_goggle`, `no_gloves`, `no_boots`) e `none`
continuam apenas em `raw_detections`/`bbox`/contadores: não geram booleanos por
pessoa, não contam como EPI positivo e não anulam uma detecção positiva associada.

## Configuração

| Variável | Padrão | Finalidade |
|---|---|---|
| `EPI_MODEL_PATH` | `model_weights/ppe-v2-100epochs.engine` | Relativo à raiz do código ou absoluto; `.pt` para CPU |
| `EPI_DEVICE` | `0` | GPU CUDA; use `cpu` somente com `.pt` |
| `EPI_IMAGE_SIZE` | `640` | `imgsz` |
| `EPI_CONFIDENCE_THRESHOLD` | `0.25` | Classes adicionais |
| `EPI_PERSON_THRESHOLD` | `0.35` | Pessoa |
| `EPI_HELMET_THRESHOLD` | `0.45` | Capacete |
| `EPI_VEST_THRESHOLD` | `0.45` | Colete |
| `EPI_IOU_THRESHOLD` | `0.70` | NMS IoU |

O serviço usa um worker. Para execução direta, exporte as variáveis antes de iniciar.
No SACI, ajuste `runtime.environment_variables` no manifesto (elas definem os valores
do processo gerenciado). O manifesto aponta para a engine na GPU `0`; para CPU,
altere nele `EPI_MODEL_PATH` para o `.pt` e `EPI_DEVICE` para `cpu`.

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
    "gloves": false,
    "gloves_confidence": "0.0",
    "boots": true,
    "boots_confidence": "0.78",
    "goggles": false,
    "goggles_confidence": "0.0",
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
- `EPIDetection`: objeto com `{detections, non_compliant, time}` no formato legado,
  com os campos adicionais de luvas/botas/óculos por pessoa. Para um consumidor legado, use
  `headers['EPIDetection'] = json.dumps(response['EPIDetection'])`.
- `EPIMaxConfidence`: string com a maior confiança entre pessoas, capacetes e coletes
  associados, ou `"0.0"` sem pessoas, como no Docker. Não inclui luvas/botas/óculos.
- `raw_detections`: caixas do modelo como `{class_id,label,confidence,bbox_xyxy}`.
  Inclui também as caixas de luvas, botas, óculos e classes negativas, independentemente
  de sua associação a pessoas.
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
regiões espaciais (incluindo limites), luvas/botas/óculos presentes e ausentes,
seleção por confiança, preservação da conformidade/máximo legado, tipos legados,
limiares por classe, ausência de caixas e transportes
array/path/base64/raw/SHM com validações de entrada. Testes de serviço usam resultado
YOLO sintético, sem carregar pesos. O checkpoint `.pt` e HTTP foram validados em CPU
com frame preto. A engine foi carregada com Ultralytics na `flintstone` e validada com
frames pretos 640×640/480×640 (zero caixas) e a imagem `bus.jpg` do Ultralytics
(11 caixas, incluindo pessoas), exercitando pré-processamento, TensorRT e NMS.
Esses smoke tests não substituem avaliação de precisão de EPI em cenas reais;
o serviço HTTP com a engine não foi validado nesta etapa.
