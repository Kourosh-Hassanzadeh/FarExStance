# FarExStance

## Environment setup

1. Install python3.12(Linux):

```
wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz

tar -xvf Python-3.12.0.tgz

cd Python-3.12.0

./configure

make

sudo make install
```

2. create virtual env and install requirements:

first clone the project:

```
git clone https://github.com/Kourosh-Hassanzadeh/FarExStance.git

cd FarExStance
```

```
python3.12 -m venv env

source env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

* if you get this error: 
    ```
    WARNING: pip is configured with locations that require TLS/SSL, however the ssl module in Python is not available.
    WARNING: pip is configured with locations that require TLS/SSL, however the ssl module in Python is not available.
    Could not fetch URL https://pypi.org/simple/pip/: There was a problem confirming the ssl certificate: HTTPSConnectionPool(host='pypi.org', port=443): Max retries exceeded with url: /simple/pip/ (Caused by SSLError("Can't connect to HTTPS URL because the SSL module is not available.")) - skipping 
    ```

do this:

```
cd Python-3.12.0

make clean

./configure \
  --enable-optimizations \
  --with-openssl=/usr \
  --with-ensurepip=install

make -j$(nproc)

sudo make altinstall
```


## Reproduce Results:

first we need to train XLM-R:

```
python xlmr_model.py
```

* if you get this error:

  ```
  modulenotfounderror: no module named '_sqlite3'
  ```
  do this:
  ```
  sudo apt isntall libsqlite3-dev
  
  cd Python3.12.0

  ./configure --enable-loadable-sqlite-extensions

  make

  sudo make altinstall
  ```
  
  ## or more lightweight:
  ```
  python xlmr_sampled.py
  ```

  ## reproduce results:

  ```
  python xlmr_inference.py
  ```


* I used GPU for training but it can be trained an CPU too.