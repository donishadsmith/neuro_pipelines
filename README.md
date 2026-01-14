Neuroimaging pipelines tailored for specific tasks and scan protocols.

**Requires >=Python3.10**

Files related to:
- Converting dataset to BIDS format
- Creating event (timing) files
- Preprocessing data
- Performing first level analyses
- Extracting contrasts

To get the required packages on local workstation. In your preferred terminal:

```bash
pip install -r requirements.txt
```

or

```bash
python -m venv venv

# For Linux
source venv/bin/activate

# For Windows
venv\Scripts\activate

pip install -r requirements.txt
```

On HPC:

```bash
conda create -n venv python=3.10 
source activate venv
pip install -r requirements.txt
```