# Collaborative filtering recommendation engine
Simple Collaborative recommendation engine model for product similarity estimation.

The final model is based on item euclidean distance and can be deployed using the the Docker image.

## Requirements

### Running with Python only
Built with [Python 3.7](https://www.python.org/downloads/release/python-370/), but should work with other > Python 3.5 versions.

Install the requirements:

```bash
pip install -r requirements.txt
```

And run the in a dev enviroment:
```bash
python app/main.py
```
### Running with a docker container
Requires that you have setup Docker and running. 

Simple run the command to build and deploy:

```docker
docker build -t recommender-image . && docker run -p 80:80 --name recommender recommender-image 
```

## Testing the application
There is only one endpoint, **/recommend/**, to obtain the TOP 10 user recommended tags in JSON format.

Do a GET petition to the local URL through browser with the user_id as param:
```
http://localhost:80/recommend/<user_id>
```
Example:
http://localhost:80/recommend/00000055a78bf6735c4a89358fab1de34104c3cb

Or using curl:
```bash
curl http://localhost:80/recommend/00000055a78bf6735c4a89358fab1de34104c3cb
```
You should see a reponse:
```json
{
    "user_predicted_tags":
    [
        "642b8a4788b43df14fccc16ba2c926c1a42a736f",
        "9f5cd26abfc96a97f8ee874d132c526a0fccb382",
        "cf78455a9e99ea73476ba4fe54f395c1ed4205d2",
        "3f736ea31dc289439c2868ef54b0fcb8ea3be3b9",
        "17e899eebd1a5d51fb708c6afff76c03bcf1d635",
        "03dd382ae49f2579667496e58e1686f7d9ad58ce",
        "3e4d8d24daf15692515999d4c8809eac1a3ee55c",
        "8f9cad3197bec0704f6d8f7817158eff7a10d86a",
        "99d651ced09bfba1bf4345bc510e510853750ffc",
        "29d400c6bada3de9543bcd931729848b5a95cdd6"
    ]
}
```
