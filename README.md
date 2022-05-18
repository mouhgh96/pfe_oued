#SETUP
###From conda create requirements.txt for pip3
`pip list --format=freeze > requirements.txt`
###create a virtual env and install requirements
`python3 -m venv env
source env/bin/activate
pip install -r requirements.txt`

###Install FrenchLefffLemmatizer
`pip install git+https://github.com/ClaudeCoulombe/FrenchLefffLemmatizer.git`
