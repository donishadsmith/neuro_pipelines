Neuroimaging pipelines tailored for specific tasks and scan protocols for a multi-session, pharmacological fMRI study.

**Requires >=Python3.10**

Files related to:
- Converting dataset to BIDS format
- Creating event (timing) files
- Preprocessing data
- Performing first level and second level analyses (GLM and gPPI)
- Additional miscellaneous files

To get the required packages on local workstation. In your preferred terminal:

```bash
git clone https://github.com/donishadsmith/neuro_pipelines
cd neuro_pipelines
pip install -r requirements.txt
```

To use a virtual environment on your local workstation:

```bash
python -m venv venv

# For Linux
source venv/bin/activate

# For Windows
venv\Scripts\activate

pip install -r requirements.txt
```

To create a virtual environment on the HPC:

```bash
conda create -n venv python=3.10
source activate venv
pip install -r requirements.txt
```
