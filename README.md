# Learning-Driven Skyline-Based Variable Neighborhood Search for Fast Floorplanning

This repository hosts the source code for the proposed algorithm, along with the corresponding instance data, aggregated results, and detailed solutions. 

## Cite

To cite the contents of this repository, please cite both the paper and this repository.

Below is the BibTex for citing this repository:

```
@misc{chen2025learning,
  title={Learning-Driven Skyline-Based Variable Neighborhood Search for Fast Floorplanning},
  author={Chen, Zhuanggeng and Wang, Sunkanghong and Yi, Haotian and Zhang, Hao and Wei, Lijun and Liu, Qiang},
  journal={...},
  year={...},
  publisher={...},
  doi={...},
  url={https://github.com/ChenZhuangGeng/L_S_VNS_Floorplanning},
  note={Available for download at https://github.com/ChenZhuangGeng/L_S_VNS_Floorplanning},
}
```

## Requirements

This project relies on [Conda](https://docs.conda.io/en/latest/) for environment management. All dependencies, including the Python and PyTorch versions, are listed in the `environment.yml` file.

## Installation

Please follow the steps below to create and activate the required environment:

1. **Clone the repository and enter it**

   ```bash
   git clone https://github.com/ChenZhuangGeng/L_S_VNS_Floorplanning.git
   
   cd L_S_VNS_Floorplanning
   ```

2. **Unzip the core files**

   In the repository's root directory, there is a core compressed file named `L_S_VNS_FP`. You need to unzip it first.

   - **On macOS / Linux / Git Bash (Windows):**

     ```bash
     unzip L_S_VNS_FP.zip
     cd L_S_VNS_FP
     ```

   - **On Windows (File Explorer):**

     - In the cloned `L_S_VNS_Floorplanning` folder, find the `L_S_VNS_FP.zip` file.
     - Right-click -> "Extract All..."
     - Ensure you extract it to the current folder, and then enter that folder.

3. **Create the Conda environment**

   ```bash
   conda env create -f environment.yml
   conda activate dl_pytorch
   ```

4. **Run the project**

   ```bash
   python -m src.run.RunnerForGCN --start 1 --end 301
   ```

## Manual Installation

The main dependencies for this project are as follows. It is recommended to create a `Python 3.10` environment using conda.

1. **Install PyTorch (CUDA 12.1):**

   - Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to get the installation command for your system. This project uses `torch 2.3.1` and `CUDA 12.1`.

2. **Install PyTorch Geometric (PyG):**

   - Follow the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) to install according to your PyTorch and CUDA versions.

3. **Install other main dependencies:**

   ```bash
   pip install numpy scipy scikit-learn lightgbm matplotlib shapely
   ```

## Data Format & Usage

The input data for this project is located in the `L_S_VNS_FP/src/data/EDA_DATA/` directory.

### Directory Structure

The structure of the `EDA_DATA` directory is as follows:

```
EDA_DATA/
├── connect/
│   └── connect_file
│         ├── connect_5.txt
│         ├── connect_10.txt
│         ├── connect_16.txt
│         └── ...
├── sample5/
│   ├── 5-1
│   │   ├── placement_info.txt
│   │   └── placement_info_compress.txt
│   ├── 5-2
│   │   ├── placement_info.txt
│   │   └── placement_info_compress.txt
│   └── ...
├── sample10/
├── sample16/
├── ...
└── sample45/
```

- `sampleX` (e.g., `sample5`, `sample10`) folders: Contain module layout files of different scales (defining module area, boundaries, ports, etc.). The `compress` versions feature a more compact layout compared to the standard versions.

- `connect` folder: Contains the corresponding connection relationship files (defining how modules are connected).

### Input File Format

Module File Format

```
// Module File Format: placement_info.txt

// 1. Layout Area Settings (Area)
//    - Rule: Defines spacing rules for different port types (e.g., SD, GATE, SD_ITO, GATE_ITO)
Area: (XXX, XXX) (XXX, XXX) (XXX, XXX)
Rule: SD (5, 5); GATE (4, 4); SD_GATE (0.5); SD_ITO (0.5); GATE_ITO (0.5)

// 2. Module Definition (Module)
Module: M1                                  // Module M1
Boundary: (0, 0) (90, 0) (90, 90) (0, 90); GATE  // M1's boundary coordinates and type
Port: (0, 45) (5, 45) (5, 50) (0, 50); SD   // M1's first port (index 1), type SD
Port: (45, 0) (45, 5) (50, 5) (50, 0); GATE // M1's second port (index 2), type GATE

Module: M2                                  // Module M2
Boundary: (XXX, XXX) (XXX, XXX) (XXX, XXX) (XXX, XXX); ITO // M2's boundary
Port: (XXX, XXX) (XXX, XXX) (XXX, XXX) (XXX, XXX); SD   // M2's first port
Port: (XXX, XXX) (XXX, XXX) (XXX, XXX) (XXX, XXX); GATE // M2's second port
...
```

Connection File Format

```
Link1:                  // First connection (Net)
M1   M2   M3              // Modules involved in the connection
1    1    3               // M1's 1st port, M2's 1st port, and M3's 3rd port connect
Link2:                  // Second connection
M5   M7
2    3                   // M5's 2nd port connects with M7's 3rd port
Link3:
M3   M5
1    2
...
```

## Saving Results

Performance metrics from the algorithm's execution (e.g., scores, iteration counts, runtime) will be automatically saved.

**Save Path:** The result files are saved by default to the `src/result/` directory (the specific path is specified by the `output_csv_path` parameter in the code).

The file will contain the following columns, and a new row will be appended for each run:

| Column             | Description                                               |
| ------------------ | --------------------------------------------------------- |
| sample name        | Complete test case identifier (extracted from input path) |
| main sample name   | Main category of the test case (e.g., `sample5`)          |
| sub sample name    | Specific name of the test case (e.g., `sample5-1`)        |
| score              | The model's predicted score (GCN Score)                   |
| real score         | The evaluated real score (Real Score)                     |
| vns_iterations     | Number of iterations for the VNS algorithm                |
| evaluation count   | Total number of evaluation function calls                 |
| avg_eval_time      | Average time for a single evaluation                      |
| total_compute_time | Total computation time for the algorithm                  |

## Paper-related Experimental Results

- The experimental results related to the paper are stored in the `L_S_VNS_FP/result/result.xlsx` file.
