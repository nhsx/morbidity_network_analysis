# MultiNet - Multi-Morbidity Network Analysis

## Build and visualise multi-morbidity networks to discover significant disease associations.

[![status: experimental](https://github.com/GIScience/badges/raw/master/status/experimental.svg)](https://github.com/GIScience/badges#experimental)

This command line tool provides user-friendly and automated multi-morbidity network analysis.
Detect significant associations are correcting for confounding factors such as Age and Sex.
Includes community detection for und-irected networks.
Option to build directed networks when diagnosis times are available.

***Note: Directed network analysis is still experimental.***

## Table of contents

  * [Installation](#installation)
  * [Configuration](#configuration)
  * [Usage](#usage)
  * [Example output](#example-output)
  * [Contributing](#contributing)
  * [License](#license)
  * [Contact](#contact)

## Installation

```bash
pip install git+https://github.com/nhsx/morbidity_network_analysis.git
```

### Docker

```bash
git clone --depth 1 https://github.com/nhsx/morbidity_network_analysis.git
docker build -t cma .
docker run cma --help
```

To run the following example via Docker we recommended using docker volumes to access local data from the docker container.
The following command mounts the current directory to the directory `/cma/` within the container.
Following this we update the container working directory (`-w /cma/`) and run the commands as normal.

```bash
docker run -v $(pwd):/cma/ -w /cma/ \
  cma process config.yaml
```

## Configuration
MultiNet is configured via a single configuration file in YAML format.
The configuration describes the file-path of the input data and column names of the desired strata and diseases.
All columns provided in the configuration must be present in the input data.
MultiNet can automatically handle gzipped compressed files and file seperator can be configured to any relevant character.
The configuration file shown below is suitable for the example data generated by ```cma simulate``` (see below).

```bash
input: CMA-example.csv
edgeData: CMA-example-processed.csv.gz
networkPlot: exampleNetwork.html
strata:
    - Age
excludeNode:
    - 1
radius: 2
codes:
    code1: time1
    code2: time2
    code3: time3
    code4: time4
seed: 42
```


## Usage
MultiNet can be run from the command line and additional help is provided via ```cma --help```.

### Generate Example Data
The ```simulate``` sub-command generates suitably formatted input data for testing functionality.
It also writes an example config file in YAML format.

```bash
cma simulate --config config.yaml > CMA-example.csv
```

### Stage 1 - Processing input and generate edge weights.
The ```process``` sub-command reads the input data and performs a stratified odds-ratio test (Mantel-Haenszel method) for each pair of morbidities.
The results are written in `.csv` format to the path defined by `edgeData:` in the configuration file.
The default simulated data should take approximately 5 minutes to process.

```bash
cma process config.yaml
```

### Stage 2 - Network Construction and Visualisation
The ```network``` sub-command parses the output of ```cma process``` into a network and generate an interactive visualisation.
The visualisation is written in `.html` format to the path defined by `networkPlot:` in the configuration file.

```bash
cma network config.yaml
```

### Alternative Method - Run Full Workflow
MultiNet can be optionally run in a single command that combines stage 1 and stage 2.

```bash
cma analyse config.yaml
```

However, it is generally advised to run each stage separately.
Optimal network visualisation parameters can be quickly explored without having to repeatedly re-run stage 1, which may require considerable compute time.


## Example output
The example network is designed to test MultiNet functionality and configuration.
The simulated data defines relationship among the nodes according to their numerical values.
Specifically, a given node is more likely to associate with numeric factors (e.g. node 8 -> 4, 2, 1)
MorbiNet can successfully recover these relationships in the network analysis.
Finally node relationships are temporal to mimic diagnosis time - in the simulated data larger numbers always occur before smaller.

![Example Network Output](./README_files/exampleNet.png)

### Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### License

Distributed under the MIT License. _See [LICENSE](./LICENSE) for more information._

### Contact

If you have any other questions please contact the author **[Stephen Richer](https://www.linkedin.com/in/stephenricher/)**
at stephen.richer@proton.me
