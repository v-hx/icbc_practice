# icbc_practice

Simple financial instrument prediction solution using ML regression models.

### Prerequisites

- Tested on Python 3.11
- Virtual environment, see <https://github.com/pyenv/pyenv>
- `pip intstall -r requirements.txt`
- Datasets are not commited, add them and define the paths in `config.py`
- See `config.py` and adjust any other confiuration as required

### Run locally

`python main.py`

### Run in Docker

`docker build . --tag icbc`

`docker run -v ${pwd}/output:/usr/src/app/output icbc`

### Additional notes

- Depending on your operating system, you might require to manually create the `output` directory on your host when running in Docker.
- GridSearchCV is time-consuming, you might want to just predict the target, once you have trained a model. Use the `Regressor.predict_with_model` method
