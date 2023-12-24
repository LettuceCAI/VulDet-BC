### 文档结构

```Python
|-- VulDet-BC
    |-- README.md
    |-- functions.py    # Model Training Correlation Function
    |-- Model.py    # Models used for all experiments
    |-- processData.py    # Data Processing Functions
    |-- HAN-BSVD.py    # Execution code for HAN-BSVD
    |-- VulDet-BC.py    # Execution code for VulDet-BC
    |-- RQ3.py    # Execution code for RQ3
    |-- RQ4.py    # Execution code for RQ4
    |-- RQ2     # Execution code for RQ2
        |-- lengthEX.py 
        |-- Ltime.py  
    |-- data    # Resulting from data processing
    |-- dataset # Source dataset
    |-- model    # word vector
```

### About processData.py:
#### LinuxData_HANBSVD, WindowsData_HANBSVD and WholeData_HANBSVD for HAN-BSVD.
#### LinuxData_L, WindowsData_L and WholeData_L for RQ2.
#### LinuxData_Con, WindowsData_Con and WholeData_Con are used for Concatenation in RQ3.
